#!/usr/bin/env python3
"""Validate and compare canonical schema-v8 graphs by explicit node ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


class SchemaError(ValueError):
  pass


I64_MIN, I64_MAX, U64_MAX = -(2**63), 2**63-1, 2**64-1
FLOAT_BITS = re.compile(r"0x[0-9a-f]{16}")
XXH64 = re.compile(r"0x[0-9a-f]{16}")
SHA256 = re.compile(r"[0-9a-f]{64}")

ARG_FIELDS = {
  "none": set(), "const": {"value"}, "device": {"name"},
  "sink": {"name", "opts_to_apply", "applied_opts", "dont_use_locals"},
  "d_type": {"value"}, "index": {"value"}, "name": {"value"},
  "param": {"slot", "dtype", "vmin_vmax", "multiple_of", "name", "address_space", "axis", "device", "volatile"},
  "size": {"value"}, "stage": {"device", "local_axis", "address_space", "removable"}, "axes": {"values"},
  "bool_axes": {"values"}, "pad": {"begin", "end"}, "reduce": {"op", "axes", "num_axes"}, "all_reduce": {"op", "device"},
  "range": {"axis", "renumbered", "axis_type"}, "constants": {"values"}, "define_var": {"name", "min", "max"},
  "call": {"grad_tag", "metadata", "name", "precompile", "precompile_backward"},
  "wmma": {"dims", "dtype_in", "dtype_out", "device", "threads", "upcast_a", "upcast_b", "upcast_c"},
  "source": {"code"}, "binary": {"length"}, "ins": {"opcode", "attributes"}, "hints": {"values"},
  "code": {"value"}, "custom_function": {"kind_name"},
  "program": {"name", "global_size", "local_size", "vars", "globals", "outs", "ins", "target"},
}
SCALARS = {"void", "weakint", "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
           "weakfloat", "fp8e4m3", "fp8e4m3fnuz", "fp8e5m2", "fp8e5m2fnuz", "float16", "bfloat16", "float32",
           "float64", "index"}
ARG_OPS = {
  "const": {"CONST"}, "device": {"GETADDR", "COPY"}, "sink": {"SINK"}, "d_type": {"CAST", "BITCAST"},
  "index": {"MSELECT", "MULTI", "GETTUPLE"}, "name": {"SPECIAL"}, "param": {"PARAM", "BUFFER"},
  "size": {"SLICE"}, "stage": {"STAGE"}, "axes": {"PERMUTE"}, "bool_axes": {"FLIP"}, "pad": {"PAD"},
  "reduce": {"REDUCE", "REDUCE_AXIS"}, "all_reduce": {"ALLREDUCE"}, "range": {"RANGE"},
  "constants": {"VCONST"}, "define_var": {"DEFINE_VAR"}, "call": {"CALL", "FUNCTION"}, "wmma": {"WMMA"},
  "source": {"SOURCE"}, "binary": {"BINARY"}, "ins": {"INS"}, "hints": {"CONTIGUOUS"},
  "code": {"CUSTOM", "CUSTOMI"}, "custom_function": {"CUSTOM_FUNCTION"}, "program": {"PROGRAM"},
}
NONE_OPS = {
  "NOOP", "SINK", "GROUP", "INDEX", "MSTACK", "STACK", "RESHAPE", "EXPAND", "SHRINK", "IF", "ENDIF",
  "END", "BARRIER", "BIND", "TUPLE", "LINEAR", "DETACH", "CONTIGUOUS_BACKWARD", "AFTER", "PRECAST", "LOAD", "STORE",
  "ADD", "MUL", "SUB", "FLOORMOD", "CMOD", "MAX", "POW", "FLOORDIV", "CDIV", "FDIV", "CMPLT", "CMPLE",
  "CMPEQ", "CMPNE", "CMPGT", "CMPGE", "AND", "OR", "XOR", "SHL", "SHR", "THREEFRY", "WHERE", "MULACC",
  "EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL", "NEG", "TRUNC", "NOT", "ABS", "RSQRT", "EXP", "LOG", "COS",
  "TAN", "FLOOR", "CEIL", "ROUND", "SIGN", "ERF", "SQUARE",
}
OPS = NONE_OPS | set().union(*ARG_OPS.values())
REDUCE_OPS = {"ADD", "MUL", "MAX", "MIN"}
AXIS_TYPES = {"DEVICE", "GLOBAL", "WARP", "LOCAL", "WEAK", "LOOP", "GROUP_REDUCE", "REDUCE", "UPCAST",
              "UNROLL", "THREAD", "PLACEHOLDER"}
OPT_OPS = {"TC", "UPCAST", "UNROLL", "LOCAL", "THREAD", "GROUP", "GROUPTOP", "NOLOCALS", "PADTO", "SWAP"}
ADDRESS_SPACES = {"global", "local", "register"}


def require(condition: bool, path: str, message: str) -> None:
  if not condition: raise SchemaError(f"{path}: {message}")


def exact_keys(value: Any, expected: set[str], path: str) -> None:
  require(isinstance(value, dict), path, "expected object")
  actual = set(value)
  require(actual == expected, path, f"fields must be {sorted(expected)}, got {sorted(actual)}")


def is_int(value: Any) -> bool: return isinstance(value, int) and not isinstance(value, bool)


def object_value(value: Any, path: str) -> dict[str, Any]:
  require(isinstance(value, dict), path, "expected object")
  return value


def list_value(value: Any, path: str) -> list[Any]:
  require(isinstance(value, list), path, "expected list")
  return value


def string_value(value: Any, path: str, *, nonempty: bool = False) -> str:
  require(isinstance(value, str) and (not nonempty or bool(value)), path,
          "expected nonempty string" if nonempty else "expected string")
  return value


def bool_value(value: Any, path: str) -> bool:
  require(type(value) is bool, path, "expected bool")
  return value


def int_value(value: Any, path: str, minimum: int, maximum: int) -> int:
  require(is_int(value) and minimum <= value <= maximum, path,
          f"expected integer in [{minimum}, {maximum}]")
  return value


def i64_value(value: Any, path: str) -> int: return int_value(value, path, I64_MIN, I64_MAX)
def u64_value(value: Any, path: str) -> int: return int_value(value, path, 0, U64_MAX)
def slot_value(value: Any, path: str) -> int: return int_value(value, path, -1, U64_MAX)


def enum_value(value: Any, choices: set[str], path: str, label: str) -> str:
  value = string_value(value, path)
  require(value in choices, path, f"unknown {label} {value!r}")
  return value


def optional_string(value: Any, path: str) -> None:
  require(value is None or isinstance(value, str), path, "expected string or null")


def integer_list(value: Any, path: str, *, signed: bool = False, nonempty: bool = False) -> list[int]:
  values = list_value(value, path)
  require(not nonempty or bool(values), path, "expected nonempty list")
  for index, item in enumerate(values):
    (i64_value if signed else u64_value)(item, f"{path}[{index}]")
  return values


def string_list(value: Any, path: str) -> list[str]:
  values = list_value(value, path)
  for index, item in enumerate(values): string_value(item, f"{path}[{index}]")
  return values


def validate_dtype(value: Any, path: str) -> None:
  value = object_value(value, path)
  require(isinstance(value.get("kind"), str), path, "expected tagged dtype")
  kind = value["kind"]
  fields = {
    "scalar": {"kind", "name"}, "vector": {"kind", "scalar", "count"},
    "pointer": {"kind", "base", "address_space", "size", "count"}, "image": {"kind", "image_kind", "shape"},
  }
  require(kind in fields, path, f"unknown dtype kind {kind!r}")
  exact_keys(value, fields[kind], path)
  if kind == "scalar": enum_value(value["name"], SCALARS, path + ".name", "scalar dtype")
  if kind == "vector":
    enum_value(value["scalar"], SCALARS, path + ".scalar", "vector scalar")
    require(u64_value(value["count"], path + ".count") > 0, path + ".count", "expected positive integer")
  if kind == "pointer":
    validate_dtype(value["base"], path + ".base")
    enum_value(value["address_space"], ADDRESS_SPACES, path + ".address_space", "address space")
    if value["size"] is not None: u64_value(value["size"], path + ".size")
    require(u64_value(value["count"], path + ".count") > 0, path + ".count", "expected positive integer")
  if kind == "image":
    enum_value(value["image_kind"], {"half", "float"}, path + ".image_kind", "image kind")
    integer_list(value["shape"], path + ".shape")


def validate_const(value: Any, path: str) -> None:
  value = object_value(value, path)
  kind = enum_value(value.get("kind"), {"invalid", "int", "uint", "float", "bool"}, path + ".kind", "constant kind")
  expected = {"kind"} if kind == "invalid" else {"kind", "bits" if kind == "float" else "value"}
  exact_keys(value, expected, path)
  if kind == "int": i64_value(value["value"], path + ".value")
  if kind == "uint": u64_value(value["value"], path + ".value")
  if kind == "bool": bool_value(value["value"], path + ".value")
  if kind == "float":
    bits = string_value(value["bits"], path + ".bits")
    require(FLOAT_BITS.fullmatch(bits) is not None, path + ".bits", "expected 64-bit lowercase hex payload")


def validate_axis(value: Any, path: str) -> None:
  exact_keys(value, {"path", "renumbered"}, path)
  integer_list(value["path"], path + ".path", nonempty=True)
  bool_value(value["renumbered"], path + ".renumbered")


def validate_opt_arg(value: Any, op: str, path: str) -> None:
  value = object_value(value, path)
  require(len(value) == 1, path, "expected one tagged optimization argument")
  kind = next(iter(value), None)
  expected = "TensorCore" if op == "TC" else "Swap" if op == "SWAP" else "Int"
  require(kind == expected, path, f"{op} requires {expected} argument")
  payload = value[kind]
  if kind == "Int": u64_value(payload, path + ".Int")
  elif kind == "Swap":
    exact_keys(payload, {"other_axis"}, path + ".Swap")
    u64_value(payload["other_axis"], path + ".Swap.other_axis")
  else:
    exact_keys(payload, {"tc_select", "opt_level", "use_tc"}, path + ".TensorCore")
    int_value(payload["tc_select"], path + ".TensorCore.tc_select", -(2**31), 2**31-1)
    u64_value(payload["opt_level"], path + ".TensorCore.opt_level")
    u64_value(payload["use_tc"], path + ".TensorCore.use_tc")


def validate_opt(value: Any, path: str) -> None:
  exact_keys(value, {"op", "axis", "arg"}, path)
  op = enum_value(value["op"], OPT_OPS, path + ".op", "optimization")
  if value["axis"] is not None: u64_value(value["axis"], path + ".axis")
  if op == "NOLOCALS": require(value["axis"] is None, path + ".axis", "NOLOCALS axis must be null")
  validate_opt_arg(value["arg"], op, path + ".arg")


def validate_opts(value: Any, path: str) -> None:
  for index, item in enumerate(list_value(value, path)): validate_opt(item, f"{path}[{index}]")


def validate_program_value(value: Any, path: str) -> int | None:
  value = object_value(value, path)
  kind = enum_value(value.get("kind"), {"int", "uint", "float", "node"}, path + ".kind", "PROGRAM value kind")
  field = "bits" if kind == "float" else "node" if kind == "node" else "value"
  exact_keys(value, {"kind", field}, path)
  if kind == "int": i64_value(value[field], path + ".value")
  elif kind == "uint": u64_value(value[field], path + ".value")
  elif kind == "node": return u64_value(value[field], path + ".node")
  else:
    bits = string_value(value[field], path + ".bits")
    require(FLOAT_BITS.fullmatch(bits) is not None, path + ".bits", "expected 64-bit lowercase hex payload")
  return None


def validate_arg(value: Any, op: str, path: str) -> list[int]:
  require(isinstance(value, dict) and isinstance(value.get("kind"), str), path, "expected tagged arg")
  kind = value["kind"]
  require(kind in ARG_FIELDS, path + ".kind", f"unknown arg kind {kind!r}")
  exact_keys(value, {"kind"} | ARG_FIELDS[kind], path)
  allowed_ops = NONE_OPS if kind == "none" else ARG_OPS[kind]
  require(op in allowed_ops, path + ".kind", f"arg kind {kind!r} is invalid for op {op!r}")
  refs: list[int] = []
  if kind == "const": validate_const(value["value"], path + ".value")
  if kind == "d_type": validate_dtype(value["value"], path + ".value")
  if kind in {"device", "name", "source", "code", "custom_function"}:
    field = "name" if kind == "device" else "value" if kind in {"name", "code"} else "code" if kind == "source" else "kind_name"
    string_value(value[field], path + f".{field}")
  if kind in {"index", "size"}: u64_value(value["value"], path + ".value")
  if kind == "param":
    validate_dtype(value["dtype"], path + ".dtype")
    slot_value(value["slot"], path + ".slot")
    if value["vmin_vmax"] is not None:
      bounds = list_value(value["vmin_vmax"], path + ".vmin_vmax")
      require(len(bounds) == 2, path + ".vmin_vmax", "expected two constants")
      for index, bound in enumerate(bounds): validate_const(bound, f"{path}.vmin_vmax[{index}]")
    if value["multiple_of"] is not None: u64_value(value["multiple_of"], path + ".multiple_of")
    optional_string(value["name"], path + ".name")
    if value["address_space"] is not None:
      enum_value(value["address_space"], ADDRESS_SPACES, path + ".address_space", "address space")
    if value["axis"] is not None: u64_value(value["axis"], path + ".axis")
    optional_string(value["device"], path + ".device")
    bool_value(value["volatile"], path + ".volatile")
  if kind == "sink":
    optional_string(value["name"], path + ".name")
    if value["opts_to_apply"] is not None: validate_opts(value["opts_to_apply"], path + ".opts_to_apply")
    validate_opts(value["applied_opts"], path + ".applied_opts")
    bool_value(value["dont_use_locals"], path + ".dont_use_locals")
  if kind == "stage" and value["local_axis"] is not None: validate_axis(value["local_axis"], path + ".local_axis")
  if kind == "stage":
    optional_string(value["device"], path + ".device")
    enum_value(value["address_space"], ADDRESS_SPACES, path + ".address_space", "address space")
    bool_value(value["removable"], path + ".removable")
  if kind == "axes": integer_list(value["values"], path + ".values")
  if kind == "bool_axes":
    for index, item in enumerate(list_value(value["values"], path + ".values")): bool_value(item, f"{path}.values[{index}]")
  if kind == "pad":
    begin, end = integer_list(value["begin"], path + ".begin"), integer_list(value["end"], path + ".end")
    require(len(begin) == len(end), path, "begin/end rank mismatch")
  if kind == "constants":
    require(isinstance(value["values"], list), path + ".values", "expected list")
    for index, item in enumerate(value["values"]): validate_const(item, f"{path}.values[{index}]")
  if kind == "define_var":
    string_value(value["name"], path + ".name", nonempty=True)
    minimum, maximum = i64_value(value["min"], path + ".min"), i64_value(value["max"], path + ".max")
    require(minimum <= maximum, path, "variable minimum exceeds maximum")
  if kind == "range":
    integer_list(value["axis"], path + ".axis", nonempty=True)
    bool_value(value["renumbered"], path + ".renumbered")
    enum_value(value["axis_type"], AXIS_TYPES, path + ".axis_type", "axis type")
  if kind == "reduce":
    enum_value(value["op"], REDUCE_OPS, path + ".op", "reduction")
    if op == "REDUCE_AXIS":
      integer_list(value["axes"], path + ".axes")
      require(value["num_axes"] is None, path + ".num_axes", "REDUCE_AXIS num_axes must be null")
    else:
      require(value["axes"] is None, path + ".axes", "REDUCE axes must be null")
      u64_value(value["num_axes"], path + ".num_axes")
  if kind == "all_reduce":
    enum_value(value["op"], REDUCE_OPS, path + ".op", "reduction")
    string_value(value["device"], path + ".device", nonempty=True)
  if kind == "call":
    require(value["grad_tag"] is None, path + ".grad_tag", "schema v8 requires null")
    require(string_list(value["metadata"], path + ".metadata") == [],
            path + ".metadata", "schema v8 requires an empty CALL metadata list")
    optional_string(value["name"], path + ".name")
    bool_value(value["precompile"], path + ".precompile")
    bool_value(value["precompile_backward"], path + ".precompile_backward")
  if kind == "wmma":
    validate_dtype(value["dtype_in"], path + ".dtype_in"); validate_dtype(value["dtype_out"], path + ".dtype_out")
    require(len(integer_list(value["dims"], path + ".dims")) == 3, path + ".dims", "expected three integers")
    string_value(value["device"], path + ".device", nonempty=True)
    u64_value(value["threads"], path + ".threads")
    for field in ("upcast_a", "upcast_b", "upcast_c"):
      for index, extent in enumerate(list_value(value[field], path + f".{field}")):
        extent_path = f"{path}.{field}[{index}]"
        exact_keys(extent, {"axis", "extent"}, extent_path); validate_axis(extent["axis"], extent_path + ".axis")
        u64_value(extent["extent"], extent_path + ".extent")
  if kind == "program":
    for field in ("global_size", "local_size"):
      values = value[field]
      if values is None: continue
      values = list_value(values, path + f".{field}")
      require(len(values) == 3, path + f".{field}", "expected three launch dimensions")
      for index, item in enumerate(values):
        ref = validate_program_value(item, f"{path}.{field}[{index}]")
        if ref is not None: refs.append(ref)
    require(value["global_size"] is not None, path + ".global_size", "global_size must not be null")
    integer_list(value["vars"], path + ".vars")
    refs.extend(value["vars"])
    for field in ("globals", "outs", "ins"):
      integer_list(value[field], path + f".{field}")
    string_value(value["name"], path + ".name")
    string_value(value["target"], path + ".target", nonempty=True)
  if kind == "ins":
    string_value(value["opcode"], path + ".opcode", nonempty=True)
    for index, item in enumerate(list_value(value["attributes"], path + ".attributes")):
      item_path = f"{path}.attributes[{index}]"
      require(isinstance(item, list) and len(item) == 2, item_path, "expected string pair")
      string_value(item[0], item_path + "[0]"); string_value(item[1], item_path + "[1]")
  if kind == "binary": u64_value(value["length"], path + ".length")
  if kind == "hints":
    for index, hint in enumerate(list_value(value["values"], path + ".values")):
      hint_path = f"{path}.values[{index}]"
      exact_keys(hint, {"op", "axis", "arg"}, hint_path)
      string_value(hint["op"], hint_path + ".op", nonempty=True)
      if hint["axis"] is not None: u64_value(hint["axis"], hint_path + ".axis")
      if hint["arg"] is not None: i64_value(hint["arg"], hint_path + ".arg")
  if kind == "custom_function": enum_value(value["kind_name"], {"EncDec", "Graph"}, path + ".kind_name", "custom function")
  return refs


def validate_graph(graph: Any, name: str) -> dict[int, dict[str, Any]]:
  path = f"{name}:$"
  graph = object_value(graph, path)
  base_fields = {"schema_version", "stage", "roots", "nodes"}
  require(frozenset(graph) in {frozenset(base_fields), frozenset(base_fields | {"verbose"})}, path,
          f"fields must be {sorted(base_fields)} with optional verbose, got {sorted(graph)}")
  require(u64_value(graph["schema_version"], path + ".schema_version") == 8,
          path + ".schema_version", "expected schema version 8")
  string_value(graph["stage"], path + ".stage", nonempty=True)
  integer_list(graph["roots"], path + ".roots")
  list_value(graph["nodes"], path + ".nodes")
  node_map: dict[int, dict[str, Any]] = {}
  metadata_refs: dict[int, list[int]] = {}
  for position, node in enumerate(graph["nodes"]):
    node_path = f"{path}.nodes[{position}]"
    exact_keys(node, {"id", "op", "dtype", "shape", "arg", "src"}, node_path)
    u64_value(node["id"], node_path + ".id")
    require(node["id"] not in node_map, node_path + ".id", "duplicate node ID")
    enum_value(node["op"], OPS, node_path + ".op", "operation")
    validate_dtype(node["dtype"], node_path + ".dtype")
    require(node["shape"] is None or isinstance(node["shape"], list), node_path + ".shape", "expected list or null")
    shape_refs: list[int] = []
    for index, dim in enumerate(node["shape"] or []):
      dim_path = f"{node_path}.shape[{index}]"
      dim = object_value(dim, dim_path)
      dim_kind = enum_value(dim.get("kind"), {"const", "symbolic", "infer"}, dim_path + ".kind", "shape dimension kind")
      exact_keys(dim, {"kind"} if dim_kind == "infer" else {"kind", "node" if dim_kind == "symbolic" else "value"}, dim_path)
      if dim_kind == "const": u64_value(dim["value"], dim_path + ".value")
      if dim_kind == "symbolic": shape_refs.append(u64_value(dim["node"], dim_path + ".node"))
    metadata_refs[node["id"]] = shape_refs + validate_arg(node["arg"], node["op"], node_path + ".arg")
    integer_list(node["src"], node_path + ".src")
    node_map[node["id"]] = node
  require(set(node_map) == set(range(len(node_map))), path + ".nodes", "node IDs must be dense 0..N-1")
  for node_id, node in node_map.items():
    for ref in node["src"] + metadata_refs[node_id]:
      require(ref in node_map, f"{path}.nodes[id={node_id}]", f"reference to missing node {ref}")
      require(ref < node_id, f"{path}.nodes[id={node_id}]", f"reference {ref} is not dependency-first")
  for root in graph["roots"]: require(root in node_map, path + ".roots", f"missing root node {root}")
  for node_id, node in node_map.items():
    if node["arg"]["kind"] == "program":
      for ref in node["arg"]["vars"]:
        require(node_map[ref]["op"] == "PARAM", f"{path}.nodes[id={node_id}].arg.vars",
                f"PROGRAM variable {ref} must refer to PARAM")
        require(node_map[ref]["arg"]["kind"] == "param" and node_map[ref]["arg"]["address_space"] is None,
                f"{path}.nodes[id={node_id}].arg.vars", f"PROGRAM variable {ref} must refer to an ALU PARAM")
  if "verbose" in graph: validate_verbose(graph["verbose"], node_map, path + ".verbose")
  return node_map


def validate_verbose(value: Any, node_map: dict[int, dict[str, Any]], path: str) -> None:
  entries = list_value(value, path)
  require(len(entries) == len(node_map), path, "verbose table must contain one entry per node")
  seen: set[int] = set()
  for index, entry in enumerate(entries):
    entry_path = f"{path}[{index}]"
    entry = object_value(entry, entry_path)
    common = {"id", "tag", "backend_dtype"}
    rust_fields = common | {"runtime_id"}
    python_fields = common | {"object_id"}
    if "origin" in entry: rust_fields.add("origin")
    if "content_xxh64" in entry: rust_fields.add("content_xxh64")
    if "content_sha256" in entry: python_fields.add("content_sha256")
    require(frozenset(entry) in {frozenset(rust_fields), frozenset(python_fields)}, entry_path,
            "expected Rust or Python verbose node fields")
    node_id = u64_value(entry["id"], entry_path + ".id")
    require(node_id in node_map and node_id not in seen, entry_path + ".id", "invalid or duplicate verbose node ID")
    seen.add(node_id)
    string_value(entry["tag"], entry_path + ".tag"); string_value(entry["backend_dtype"], entry_path + ".backend_dtype")
    identity = "runtime_id" if "runtime_id" in entry else "object_id"
    u64_value(entry[identity], entry_path + f".{identity}")
    if "origin" in entry: string_value(entry["origin"], entry_path + ".origin", nonempty=True)
    if "content_xxh64" in entry:
      require(XXH64.fullmatch(string_value(entry["content_xxh64"], entry_path + ".content_xxh64")) is not None,
              entry_path + ".content_xxh64", "expected lowercase xxh64 payload")
    if "content_sha256" in entry:
      require(SHA256.fullmatch(string_value(entry["content_sha256"], entry_path + ".content_sha256")) is not None,
              entry_path + ".content_sha256", "expected lowercase SHA-256 payload")


def binding_variables(nodes: dict[int, dict[str, Any]]) -> list[tuple[str, int | None, str, dict[str, Any]]]:
  variables = []
  for node in nodes.values():
    arg = node["arg"]
    if node["op"] == "PARAM" and arg["kind"] == "param" and arg["address_space"] is None and arg["name"] is not None:
      variables.append(("param", arg["slot"], arg["name"], node["dtype"]))
    elif node["op"] == "DEFINE_VAR" and arg["kind"] == "define_var":
      variables.append(("define_var", None, arg["name"], node["dtype"]))
  return variables


def validate_schedule(schedule: Any, name: str) -> None:
  path = f"{name}:$"
  exact_keys(schedule, {"schema_version", "stage", "items", "output_slots"}, path)
  require(u64_value(schedule["schema_version"], path + ".schema_version") == 8,
          path + ".schema_version", "expected schema version 8")
  require(schedule["stage"] == "scheduled", path + ".stage", "schedule document stage must be 'scheduled'")
  list_value(schedule["items"], path + ".items")
  callable_indices: list[int] = []
  for position, item in enumerate(schedule["items"]):
    item_path = f"{path}.items[{position}]"
    exact_keys(item, {"order", "callable_index", "ast", "buffers", "output_slots", "dependencies", "bindings"}, item_path)
    require(u64_value(item["order"], item_path + ".order") == position, item_path + ".order", "must equal schedule item position")
    callable_indices.append(u64_value(item["callable_index"], item_path + ".callable_index"))
    ast_nodes = validate_graph(item["ast"], item_path + ".ast")
    require(item["ast"]["stage"] == "kernel_ast", item_path + ".ast.stage", "expected kernel_ast")
    list_value(item["buffers"], item_path + ".buffers")
    for buffer_index, buffer in enumerate(item["buffers"]):
      buffer_path = f"{item_path}.buffers[{buffer_index}]"
      exact_keys(buffer, {"argument_index", "global_slot", "buffer_slot", "origin"}, buffer_path)
      require(u64_value(buffer["argument_index"], buffer_path + ".argument_index") == buffer_index,
              buffer_path + ".argument_index", "must equal buffer position")
      require(u64_value(buffer["global_slot"], buffer_path + ".global_slot") == buffer_index,
              buffer_path + ".global_slot", "must equal argument position")
      slot_value(buffer["buffer_slot"], buffer_path + ".buffer_slot")
      enum_value(buffer["origin"], {"PARAM", "BUFFER"}, buffer_path + ".origin", "buffer origin")
    integer_list(item["output_slots"], item_path + ".output_slots")
    require(item["output_slots"] == sorted(set(item["output_slots"])), item_path + ".output_slots", "must be sorted and unique")
    integer_list(item["dependencies"], item_path + ".dependencies")
    require(all(dep < position for dep in item["dependencies"]), item_path + ".dependencies",
            "dependencies must refer to earlier schedule items")
    require(item["dependencies"] == sorted(set(item["dependencies"])), item_path + ".dependencies", "must be sorted and unique")
    list_value(item["bindings"], item_path + ".bindings")
    identities = []
    variables = binding_variables(ast_nodes)
    for binding_index, binding in enumerate(item["bindings"]):
      binding_path = f"{item_path}.bindings[{binding_index}]"
      exact_keys(binding, {"kind", "slot", "name", "dtype", "value", "schedule_loop"}, binding_path)
      kind = enum_value(binding["kind"], {"param", "define_var"}, binding_path + ".kind", "binding kind")
      if kind == "param": slot = slot_value(binding["slot"], binding_path + ".slot")
      else:
        require(binding["slot"] is None, binding_path + ".slot", "DEFINE_VAR binding slot must be null")
        slot = None
      binding_name = string_value(binding["name"], binding_path + ".name", nonempty=True)
      validate_dtype(binding["dtype"], binding_path + ".dtype")
      i64_value(binding["value"], binding_path + ".value")
      bool_value(binding["schedule_loop"], binding_path + ".schedule_loop")
      identity = (binding_name, kind, slot)
      require((kind, slot, binding_name, binding["dtype"]) in variables, binding_path,
              "binding identity does not match a variable in the kernel AST")
      identities.append(identity)
    require(identities == sorted(set(identities)), item_path + ".bindings", "bindings must be sorted and unique")
  if callable_indices:
    require(set(callable_indices) == set(range(max(callable_indices) + 1)), path + ".items",
            "callable indices must form a dense descriptor range")
  list_value(schedule["output_slots"], path + ".output_slots")
  for output_index, output in enumerate(schedule["output_slots"]):
    output_path = f"{path}.output_slots[{output_index}]"
    exact_keys(output, {"item", "buffer"}, output_path)
    item = u64_value(output["item"], output_path + ".item")
    require(item < len(schedule["items"]), output_path + ".item", "invalid item")
    buffers = schedule["items"][item]["buffers"]
    buffer = u64_value(output["buffer"], output_path + ".buffer")
    require(buffer < len(buffers), output_path + ".buffer", "invalid buffer")


def validate_document(document: Any, name: str) -> str:
  require(isinstance(document, dict), f"{name}:$", "expected object")
  if set(document) in ({"schema_version", "stage", "roots", "nodes"}, {"schema_version", "stage", "roots", "nodes", "verbose"}):
    validate_graph(document, name)
    require("verbose" not in document, f"{name}:$.verbose", "verbose documents are diagnostics, not parity inputs")
    return "graph"
  if set(document) == {"schema_version", "stage", "items", "output_slots"}:
    validate_schedule(document, name)
    return "schedule"
  raise SchemaError(f"{name}:$: expected canonical graph or schedule document fields, got {sorted(document)}")


def canonical_sha256(document: Any) -> str:
  normalized = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
  return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def all_differences(left: Any, right: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any, Any]]:
  if type(left) is not type(right): return [(path, left, right)]
  if isinstance(left, dict):
    differences = []
    for key in sorted(set(left) | set(right)):
      if key not in right: differences.append((path + (key,), left[key], "<missing>"))
      elif key not in left: differences.append((path + (key,), "<missing>", right[key]))
      else: differences.extend(all_differences(left[key], right[key], path + (key,)))
    return differences
  if isinstance(left, list):
    differences = []
    for index, (left_item, right_item) in enumerate(zip(left, right)):
      differences.extend(all_differences(left_item, right_item, path + (index,)))
    for index in range(min(len(left), len(right)), max(len(left), len(right))):
      differences.append((path + (index,), left[index] if index < len(left) else "<missing>",
                          right[index] if index < len(right) else "<missing>"))
    return differences
  return [] if left == right else [(path, left, right)]


def source_chain(graph: dict[str, Any], nodes: dict[int, dict[str, Any]], target: int) -> list[int]:
  queue = deque((root, [root]) for root in graph["roots"])
  seen: set[int] = set()
  while queue:
    node_id, chain = queue.popleft()
    if node_id == target: return chain
    if node_id in seen: continue
    seen.add(node_id)
    queue.extend((source, chain + [source]) for source in nodes[node_id]["src"])
  return []


def format_path(path: tuple[Any, ...]) -> str:
  return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path)


def remap_refs(value: Any, mapping: dict[int, int], path: tuple[Any, ...] = ()) -> Any:
  if isinstance(value, dict):
    return {key: remap_refs(item, mapping, path + (key,)) for key, item in value.items()}
  if isinstance(value, list):
    if path and path[-1] in {"src", "roots", "vars"}:
      return [mapping.get(item, f"unaligned:{item}") if is_int(item) else item for item in value]
    return [remap_refs(item, mapping, path + (index,)) for index, item in enumerate(value)]
  if is_int(value) and path and path[-1] == "id": return mapping.get(value, f"unaligned:{value}")
  if is_int(value) and path and path[-1] == "node":
    return mapping.get(value, f"unaligned:{value}")
  return value


def report(left: dict[str, Any], right: dict[str, Any], left_name: str, right_name: str) -> str | None:
  left_kind, right_kind = validate_document(left, left_name), validate_document(right, right_name)
  if left == right: return None
  hashes = [
    "canonical document sha256:",
    f"  {left_name}: {canonical_sha256(left)}",
    f"  {right_name}: {canonical_sha256(right)}",
  ]
  if left_kind != right_kind:
    return "\n".join(hashes + [f"canonical document kind mismatch: {left_kind} != {right_kind}"])
  if left_kind == "schedule":
    differences = all_differences(left, right)
    lines = hashes + [f"canonical schedule mismatch at stage {left['stage']} ({len(differences)} field differences):"]
    for path, left_value, right_value in differences:
      lines.append(f"  {format_path(path)}")
      lines.append(f"    {left_name}: {left_value!r}")
      lines.append(f"    {right_name}: {right_value!r}")
    return "\n".join(lines)

  left_nodes, right_nodes = validate_graph(left, left_name), validate_graph(right, right_name)
  for field in ("schema_version", "stage"):
    if differences := all_differences(left[field], right[field], (field,)):
      path, left_value, right_value = differences[0]
      return "\n".join(hashes + [
        f"canonical mismatch: {format_path(path)}", f"  {left_name}: {left_value!r}", f"  {right_name}: {right_value!r}",
      ])

  left_ops = [left_nodes[index]["op"] for index in sorted(left_nodes)]
  right_ops = [right_nodes[index]["op"] for index in sorted(right_nodes)]
  matcher = SequenceMatcher(a=left_ops, b=right_ops, autojunk=False)
  opcodes = matcher.get_opcodes()
  mapping = {right_id: left_id for tag, left_start, left_end, right_start, right_end in opcodes if tag == "equal"
             for left_id, right_id in zip(range(left_start, left_end), range(right_start, right_end))}
  pairs = [(left_id, right_id) for right_id, left_id in sorted(mapping.items())]

  structural: list[str] = []
  if change := next((opcode for opcode in opcodes if opcode[0] != "equal"), None):
    tag, left_start, left_end, right_start, right_end = change
    structural.append(f"canonical structural mismatch at stage {left['stage']} ({tag}):")
    structural.append(f"  {left_name} ids {left_start}:{left_end}: {left_ops[left_start:left_end]!r}")
    structural.append(f"  {right_name} ids {right_start}:{right_end}: {right_ops[right_start:right_end]!r}")

  remapped_roots = [mapping.get(root, f"unaligned:{root}") for root in right["roots"]]
  root_differences = all_differences(left["roots"], remapped_roots, ("roots",))
  if not structural and root_differences:
    path, left_value, right_value = root_differences[0]
    structural.append(f"canonical structural mismatch at stage {left['stage']}: {format_path(path)}")
    structural.append(f"  {left_name}: {left_value!r}")
    structural.append(f"  {right_name}: {right_value!r}")

  aligned_differences: list[tuple[int, int, tuple[Any, ...], Any, Any]] = []
  for left_id, right_id in pairs:
    right_node = remap_refs(right_nodes[right_id], mapping)
    differences = all_differences(left_nodes[left_id], right_node)
    if not structural:
      source_difference = next((item for item in differences if item[0] and item[0][0] == "src"), None)
      if source_difference is not None:
        path, left_value, right_value = source_difference
        structural.append(f"canonical structural mismatch at stage {left['stage']}: nodes {left_id}/{right_id} {format_path(path)}")
        structural.append(f"  {left_name}: {left_value!r}")
        structural.append(f"  {right_name}: {right_value!r}")
    aligned_differences.extend((left_id, right_id, *difference) for difference in differences)

  if not structural and not root_differences and not aligned_differences: return None
  lines = hashes + (structural or [f"canonical field mismatch at stage {left['stage']}:"])
  all_fields = [(None, None, *difference) for difference in root_differences] + aligned_differences
  if all_fields:
    lines.append(f"aligned field differences ({len(all_fields)}):")
  for left_id, right_id, path, left_value, right_value in all_fields:
    location = "graph" if left_id is None else f"nodes {left_id}/{right_id} {left_nodes[left_id]['op']}"
    lines.append(f"  {location} {format_path(path)}")
    lines.append(f"    {left_name}: {left_value!r}")
    lines.append(f"    {right_name}: {right_value!r}")
  if aligned_differences:
    left_id, right_id = aligned_differences[0][0], aligned_differences[0][1]
    for label, graph, nodes, node_id in ((left_name, left, left_nodes, left_id), (right_name, right, right_nodes, right_id)):
      chain = source_chain(graph, nodes, node_id)
      if chain: lines.append(f"  first {label} source chain: " + " -> ".join(f"{item}:{nodes[item]['op']}" for item in chain))
  return "\n".join(lines)


def test_graph() -> dict[str, Any]:
  scalar = {"kind": "scalar", "name": "float32"}
  none = {"kind": "none"}
  return {"schema_version": 8, "stage": "gated", "roots": [5], "nodes": [
    {"id": 0, "op": "PARAM", "dtype": scalar, "shape": [], "arg": {"kind": "param", "slot": 0, "dtype": scalar,
     "vmin_vmax": None, "multiple_of": None, "name": None, "address_space": "global", "axis": None, "device": None, "volatile": False}, "src": []},
    {"id": 1, "op": "CONST", "dtype": scalar, "shape": [], "arg": {"kind": "const", "value": {"kind": "float", "bits": "0x0000000000000000"}}, "src": []},
    {"id": 2, "op": "CONST", "dtype": {"kind": "scalar", "name": "bool"}, "shape": [], "arg": {"kind": "const", "value": {"kind": "bool", "value": True}}, "src": []},
    {"id": 3, "op": "INDEX", "dtype": scalar, "shape": [], "arg": none, "src": [0, 1]},
    {"id": 4, "op": "STORE", "dtype": {"kind": "scalar", "name": "void"}, "shape": None, "arg": none, "src": [3, 1, 2]},
    {"id": 5, "op": "PROGRAM", "dtype": {"kind": "scalar", "name": "void"}, "shape": None,
     "arg": {"kind": "program", "name": "test", "global_size": [{"kind": "int", "value": 1}, {"kind": "int", "value": 1}, {"kind": "int", "value": 1}],
             "local_size": None, "vars": [], "globals": [0], "outs": [0], "ins": [], "target": "CPU"}, "src": [4]},
  ]}


def test_schedule() -> dict[str, Any]:
  graph = json.loads(json.dumps(test_graph()))
  graph["stage"] = "kernel_ast"
  graph["nodes"][0]["arg"].update(name="schedule_n", address_space=None)
  return {"schema_version": 8, "stage": "scheduled", "items": [{
    "order": 0, "callable_index": 0, "ast": graph, "buffers": [], "output_slots": [], "dependencies": [],
    "bindings": [{
      "kind": "param", "slot": 0, "name": "schedule_n", "dtype": {"kind": "scalar", "name": "float32"},
      "value": 4, "schedule_loop": False,
    }],
  }], "output_slots": []}


def assert_invalid_identical(document: dict[str, Any], expected: str) -> None:
  try: report(document, json.loads(json.dumps(document)), "first", "second")
  except SchemaError as error: assert expected in str(error), f"missing {expected!r} in {error}"
  else: raise AssertionError("identical malformed documents must fail before parity")


def self_test() -> None:
  base = test_graph()
  cases = {
    "dtype": (lambda graph: graph["nodes"][3].update(dtype={"kind": "scalar", "name": "float16"}), ".dtype"),
    "source order": (lambda graph: graph["nodes"][3].update(src=[1, 0]), ".src[0]"),
    "gate": (lambda graph: graph["nodes"][4].update(src=[3, 1, 1]), ".src[2]"),
    "launch dimension": (lambda graph: graph["nodes"][5]["arg"]["global_size"][0].update(value=2), ".arg.global_size[0].value"),
  }
  for name, (mutate, expected) in cases.items():
    changed = json.loads(json.dumps(base)); mutate(changed)
    output = report(base, changed, "rust", "python")
    assert output is not None and expected in output, f"{name} diagnostic missing: {output}"
  shuffled = json.loads(json.dumps(base)); shuffled["nodes"].reverse()
  assert report(base, shuffled, "first", "second") is None
  changed = json.loads(json.dumps(base))
  changed["nodes"][3]["dtype"] = {"kind": "scalar", "name": "float16"}
  changed["nodes"][4]["src"][2] = 1
  output = report(base, changed, "rust", "python") or ""
  assert "aligned field differences (2)" in output and ".dtype.name" in output and ".src[2]" in output
  malformed = json.loads(json.dumps(base)); malformed["nodes"][4]["id"] = 8
  try: validate_graph(malformed, "malformed")
  except SchemaError as error: assert "dense" in str(error)
  else: raise AssertionError("sparse IDs must fail")
  malformed = json.loads(json.dumps(base)); malformed["nodes"][3]["shape"] = [{"kind": "symbolic", "node": None}]
  try: validate_graph(malformed, "malformed")
  except SchemaError as error: assert "expected integer" in str(error)
  else: raise AssertionError("node:null must fail")
  adversarial: list[tuple[str, dict[str, Any], str]] = []
  malformed = json.loads(json.dumps(base)); malformed["nodes"][2]["arg"]["value"]["value"] = "true"
  adversarial.append(("string bool", malformed, "expected bool"))
  malformed = json.loads(json.dumps(base)); malformed["nodes"][3]["shape"] = [{"kind": "const", "value": -1}]
  adversarial.append(("negative shape extent", malformed, "expected integer in [0"))
  for label, axis, axis_type, expected in (
      ("string axis entry", ["0"], "WEAK", "expected integer"),
      ("unknown enum", [0], "NOT_AN_AXIS", "unknown axis type")):
    malformed = json.loads(json.dumps(base))
    malformed["nodes"][3].update(op="RANGE", arg={"kind": "range", "axis": axis, "renumbered": False,
                                                       "axis_type": axis_type}, src=[1])
    adversarial.append((label, malformed, expected))
  malformed = json.loads(json.dumps(base)); malformed["nodes"][3]["shape"] = [{"kind": "symbolic", "node": 99}]
  adversarial.append(("invalid node reference", malformed, "reference to missing node 99"))
  malformed = json.loads(json.dumps(base)); malformed["nodes"][5]["arg"]["global_size"].pop()
  adversarial.append(("malformed ProgramInfo", malformed, "expected three launch dimensions"))
  malformed = test_schedule(); malformed["items"][0]["bindings"][0]["value"] = "4"
  adversarial.append(("malformed schedule binding", malformed, "expected integer"))
  for label, malformed, expected in adversarial:
    assert_invalid_identical(malformed, expected)
  shorter = json.loads(json.dumps(base)); shorter["nodes"].pop(2)
  for index, node in enumerate(shorter["nodes"]): node["id"] = index
  shorter["roots"] = [4]; shorter["nodes"][2]["src"] = [0, 1]; shorter["nodes"][3]["src"] = [2, 1, 1]; shorter["nodes"][4]["src"] = [3]
  expected_signature = report(shorter, base, "left", "right") or ""
  assert "structural mismatch" in expected_signature
  for name, mutate in {
    "unaligned dtype": lambda graph: graph["nodes"][2].update(dtype={"kind": "scalar", "name": "int32"}),
    "unaligned operation": lambda graph: graph["nodes"][2].update(op="NOOP", arg={"kind": "none"}),
  }.items():
    changed = json.loads(json.dumps(base)); mutate(changed)
    changed_signature = report(shorter, changed, "left", "right") or ""
    assert changed_signature != expected_signature and canonical_sha256(changed) in changed_signature, \
      f"{name} must invalidate the complete-document expected signature"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("left", nargs="?", type=Path); parser.add_argument("right", nargs="?", type=Path)
  parser.add_argument("--left-name", default="rust"); parser.add_argument("--right-name", default="python")
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.self_test: self_test(); return
  if args.left is None or args.right is None: parser.error("left and right JSON files are required")
  left, right = json.loads(args.left.read_text()), json.loads(args.right.read_text())
  try: difference = report(left, right, args.left_name, args.right_name)
  except SchemaError as error:
    print(f"invalid canonical document: {error}"); raise SystemExit(2) from error
  if difference is not None: print(difference); raise SystemExit(1)


if __name__ == "__main__": main()
