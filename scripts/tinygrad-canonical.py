#!/usr/bin/env python3
"""Emit canonical UOp JSON from the pinned Tinygrad reference checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TINYGRAD = ROOT / "submodules" / "new_new_tinygrad"
TARGET_COMMIT = "8c8b43de62515abe6c820b1de5aa26b30f48e43a"
NO_ARG_OPS = {
  "NOOP", "SINK", "GROUP", "INDEX", "MSTACK", "STACK", "RESHAPE", "EXPAND", "PAD", "SHRINK", "IF", "ENDIF", "END",
  "BARRIER", "BIND", "TUPLE", "LINEAR", "DETACH", "CONTIGUOUS_BACKWARD", "AFTER", "LOAD", "STORE", "ADD", "MUL", "SUB",
  "FLOORMOD", "CMOD", "MAX", "POW", "FLOORDIV", "CDIV", "FDIV", "CMPLT", "CMPEQ", "CMPNE", "AND", "OR", "XOR",
  "SHL", "SHR", "THREEFRY", "WHERE", "MULACC", "EXP2", "LOG2", "SIN", "SQRT", "RECIPROCAL", "NEG", "TRUNC",
}
sys.path.insert(0, str(TINYGRAD))

from tinygrad.dtype import DType, Invalid, InvalidType, dtypes  # noqa: E402
from tinygrad.uop import Ops  # noqa: E402
from tinygrad.uop.ops import AxisType, CallInfo, KernelInfo, ParamArg, ProgramInfo, UOp, shape_to_shape_arg  # noqa: E402
from tinygrad.helpers import ansistrip  # noqa: E402


def verify_target() -> None:
  actual = subprocess.check_output(["git", "-C", str(TINYGRAD), "rev-parse", "HEAD"], text=True).strip()
  if actual != TARGET_COMMIT:
    raise RuntimeError(f"Tinygrad reference is {actual}, expected {TARGET_COMMIT}")


class CanonicalSerializationError(TypeError):
  """The graph cannot be represented by canonical schema v8 without loss."""


class UnsupportedDivergenceError(CanonicalSerializationError):
  """Pinned Tinygrad metadata has no Svod representation."""


def canonical_dtype(dtype: DType) -> dict[str, Any]:
  names = {
    dtypes.void: "void", dtypes.weakint: "weakint", dtypes.bool: "bool",
    dtypes.int8: "int8", dtypes.uint8: "uint8", dtypes.int16: "int16", dtypes.uint16: "uint16",
    dtypes.int32: "int32", dtypes.uint32: "uint32", dtypes.int64: "int64", dtypes.uint64: "uint64",
    dtypes.weakfloat: "weakfloat", dtypes.fp8e4m3: "fp8e4m3", dtypes.fp8e4m3fnuz: "fp8e4m3fnuz",
    dtypes.fp8e5m2: "fp8e5m2", dtypes.fp8e5m2fnuz: "fp8e5m2fnuz",
    dtypes.float16: "float16", dtypes.bfloat16: "bfloat16", dtypes.float32: "float32", dtypes.float64: "float64",
  }
  if dtype not in names:
    raise UnsupportedDivergenceError(
      "pinned Tinygrad has no vector/pointer/image DType classes; those values are represented by STACK, uint64 addresses, and PARAM shapes"
    )
  return {"kind": "scalar", "name": names[dtype]}


def canonical_const(value: Any, dtype) -> dict[str, Any]:
  if isinstance(value, InvalidType): return {"kind": "invalid"}
  if isinstance(value, bool): return {"kind": "bool", "value": value}
  if isinstance(value, float):
    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return {"kind": "float", "bits": f"0x{bits:016x}"}
  if isinstance(value, int):
    kind = "uint" if dtypes.is_unsigned(dtype) else "int"
    return {"kind": kind, "value": value}
  raise CanonicalSerializationError(f"unsupported constant type: {type(value).__name__}")


def canonical_device(value: Any) -> str:
  if isinstance(value, tuple):
    raise UnsupportedDivergenceError("tuple devices have no Svod DeviceSpec representation")
  if not isinstance(value, str): raise CanonicalSerializationError(f"unsupported canonical device type: {type(value).__name__}")
  return value.upper() if value.lower() == "cpu" else value


def canonical_target(value: Any) -> str:
  from tinygrad.helpers import Target
  if isinstance(value, tuple): raise UnsupportedDivergenceError("tuple targets have no Svod ProgramInfo target representation")
  if not isinstance(value, Target): raise CanonicalSerializationError(f"PROGRAM target must be Target, got {type(value).__name__}")
  unsupported = {field: getattr(value, field) for field in ("renderer", "arch", "interface", "indices") if getattr(value, field)}
  if unsupported:
    raise UnsupportedDivergenceError(f"Target fields have no Svod DeviceSpec representation: {sorted(unsupported)}")
  if not value.device: raise UnsupportedDivergenceError("empty Tinygrad Target has no explicit Svod DeviceSpec")
  return canonical_device(value.device)


def canonical_opt(opt: Any) -> dict[str, Any]:
  op = opt.op.name
  if op == "TC":
    tc_select, opt_level, use_tc = opt.arg
    arg = {"TensorCore": {"tc_select": tc_select, "opt_level": opt_level, "use_tc": use_tc}}
  elif op == "SWAP": arg = {"Swap": {"other_axis": opt.arg}}
  else: arg = {"Int": 0 if opt.arg is None else opt.arg}
  return {"op": op, "axis": opt.axis, "arg": arg}


def canonical_program_value(value: Any, ids: dict[UOp, int]) -> dict[str, Any]:
  if isinstance(value, UOp):
    if value.op is not Ops.CONST:
      if value not in ids: raise CanonicalSerializationError("symbolic PROGRAM value is absent from canonical topology")
      return {"kind": "node", "node": ids[value]}
    value = value.arg
  if isinstance(value, bool): raise CanonicalSerializationError("bool is not a valid PROGRAM launch dimension")
  if isinstance(value, int):
    if value < -(2**63) or value > 2**64-1:
      raise UnsupportedDivergenceError("PROGRAM integer launch dimension is outside Svod's i64/u64 range")
    return {"kind": "int" if value <= 2**63-1 else "uint", "value": value}
  if isinstance(value, float):
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return {"kind": "float", "bits": f"0x{bits:016x}"}
  raise CanonicalSerializationError(f"unsupported PROGRAM launch dimension type: {type(value).__name__}")


def canonical_axis_extent(value: Any) -> dict[str, Any]:
  if not isinstance(value, tuple) or len(value) != 2 or not all(isinstance(item, int) for item in value):
    raise CanonicalSerializationError("WMMA upcast axes must contain (axis, extent) integer pairs")
  axis, extent = value
  return {"axis": {"path": [axis], "renumbered": True}, "extent": extent}


def canonical_arg(node: UOp, ids: dict[UOp, int], stage: str, verbose: bool) -> dict[str, Any]:
  if node.op is Ops.CONST: return {"kind": "const", "value": canonical_const(node.arg, node.dtype)}
  if node.op in {Ops.PARAM, Ops.BUFFER} and isinstance(node.arg, ParamArg):
    return {
      "kind": "param",
      "slot": node.arg.slot,
      "dtype": canonical_dtype(node.arg.dtype),
      "vmin_vmax": None if node.arg.vmin_vmax is None else [canonical_const(value, node.arg.dtype) for value in node.arg.vmin_vmax],
      "multiple_of": node.arg.multiple_of,
      "name": node.arg.name,
      "address_space": None if node.arg.addrspace is None or node.arg.addrspace.name == "ALU" else node.arg.addrspace.name.lower(),
      "axis": node.arg.axis,
      "device": None if node.arg.device is None else canonical_device(node.arg.device),
      "volatile": node.arg.volatile,
    }
  if node.op in {Ops.CAST, Ops.BITCAST}: return {"kind": "d_type", "value": canonical_dtype(node.arg)}
  if node.op in {Ops.MSELECT, Ops.GETTUPLE}: return {"kind": "index", "value": node.arg}
  if node.op is Ops.SPECIAL: return {"kind": "name", "value": node.arg}
  if node.op in {Ops.GETADDR, Ops.COPY}: return {"kind": "device", "name": canonical_device(node.arg)}
  if node.op is Ops.SLICE: return {"kind": "size", "value": node.arg}
  if node.op is Ops.RANGE:
    return {"kind": "range", "axis": list(node.arg[:-1]), "renumbered": stage != "rangeified", "axis_type": node.arg[-1].name}
  if node.op is Ops.STAGE:
    local_axis = node.arg.device if node.arg.addrspace.name == "LOCAL" and isinstance(node.arg.device, int) else None
    return {
      "kind": "stage",
      "device": None if local_axis is not None or node.arg.device is None else canonical_device(node.arg.device),
      "local_axis": None if local_axis is None else {"path": [local_axis], "renumbered": True},
      "address_space": node.arg.addrspace.name.lower(),
      "removable": node.arg.removable,
    }
  if node.op is Ops.PERMUTE: return {"kind": "axes", "values": list(node.arg)}
  if node.op is Ops.FLIP: return {"kind": "bool_axes", "values": list(node.arg)}
  if node.op is Ops.PAD:
    begin = node.src[1].as_shape
    output = node.src[2].as_shape
    source = node.src[0].shape
    if not all(isinstance(value, int) for value in begin + output + source):
      raise UnsupportedDivergenceError("symbolic PAD extents are not common to the pinned Svod/Tinygrad representations")
    end = tuple(out - size - start for start, out, size in zip(begin, output, source))
    if any(value < 0 for value in begin + end):
      raise UnsupportedDivergenceError("negative PAD extents are not representable by Svod")
    return {"kind": "pad", "begin": list(begin), "end": list(end)}
  if node.op is Ops.REDUCE:
    return {"kind": "reduce", "op": node.arg[0].name, "axes": None, "num_axes": node.arg[1]}
  if node.op is Ops.ALLREDUCE:
    return {"kind": "all_reduce", "op": node.arg[0].name, "device": canonical_device(node.arg[1])}
  if node.op is Ops.WMMA:
    if not isinstance(node.arg, tuple) or len(node.arg) != 5: raise CanonicalSerializationError("WMMA arg must have five fields")
    dims, dtype_in, device, threads, upcast_axes = node.arg
    if not isinstance(dims, tuple) or len(dims) != 3 or not all(isinstance(item, int) for item in dims):
      raise CanonicalSerializationError("WMMA dims must be three integers")
    if upcast_axes is None: upcast_axes = ((), (), ())
    if not isinstance(upcast_axes, tuple) or len(upcast_axes) != 3:
      raise CanonicalSerializationError("WMMA upcast metadata must have A/B/C axes")
    return {
      "kind": "wmma", "dims": list(dims), "dtype_in": canonical_dtype(dtype_in),
      "dtype_out": canonical_dtype(node.dtype), "device": canonical_device(device), "threads": threads,
      "upcast_a": [canonical_axis_extent(value) for value in upcast_axes[0]],
      "upcast_b": [canonical_axis_extent(value) for value in upcast_axes[1]],
      "upcast_c": [canonical_axis_extent(value) for value in upcast_axes[2]],
    }
  if node.op in {Ops.CALL, Ops.FUNCTION} and isinstance(node.arg, CallInfo):
    if node.arg.grad_fxn is not None: raise CanonicalSerializationError("callable grad_fxn has no stable cross-language identity")
    if node.arg.aux: raise UnsupportedDivergenceError("CallInfo aux has no Svod CallInfo representation")
    return {
      "kind": "call", "grad_tag": None, "metadata": [], "name": node.arg.name,
      "precompile": node.arg.precompile, "precompile_backward": node.arg.precompile_backward,
    }
  if node.op is Ops.SINK and isinstance(node.arg, KernelInfo):
    unsupported = {
      "axis_types": node.arg.axis_types, "estimates": node.arg.estimates, "beam": node.arg.beam,
    }
    if any(value not in ((), False, None, 0) for value in unsupported.values()):
      raise UnsupportedDivergenceError(
        f"KernelInfo fields have no Svod equivalent: {[key for key, value in unsupported.items() if value not in ((), False, None, 0)]}"
      )
    return {
      "kind": "sink", "name": None if stage in {"kernel_ast", "scheduled"} and node.arg.name == "test" else ansistrip(node.arg.name),
      "opts_to_apply": None if node.arg.opts_to_apply is None else [canonical_opt(opt) for opt in node.arg.opts_to_apply],
      "applied_opts": [canonical_opt(opt) for opt in node.arg.applied_opts],
      "dont_use_locals": node.arg.dont_use_locals,
    }
  if node.op is Ops.PROGRAM and isinstance(node.arg, ProgramInfo):
    if len(node.arg.global_size) != 3 or (node.arg.local_size is not None and len(node.arg.local_size) != 3):
      raise UnsupportedDivergenceError("Svod ProgramInfo requires exactly three global/local launch dimensions")
    return {
      "kind": "program", "name": node.arg.name,
      "global_size": [canonical_program_value(value, ids) for value in node.arg.global_size],
      "local_size": None if node.arg.local_size is None else [canonical_program_value(value, ids) for value in node.arg.local_size],
      "vars": [ids[value] for value in node.arg.vars], "globals": list(node.arg.globals),
      "outs": list(node.arg.outs), "ins": list(node.arg.ins), "target": canonical_target(node.arg.target),
    }
  if node.op is Ops.CONTIGUOUS:
    if node.arg is not None and any(opt.arg is not None and not isinstance(opt.arg, int) for opt in node.arg):
      raise UnsupportedDivergenceError("CONTIGUOUS tuple hints have no Svod ContiguousHint representation")
    values = [] if node.arg is None else [{"op": opt.op.name, "axis": opt.axis, "arg": opt.arg} for opt in node.arg]
    return {"kind": "hints", "values": values}
  if node.op in {Ops.CUSTOM, Ops.CUSTOMI}: return {"kind": "code", "value": node.arg}
  if node.op is Ops.CUSTOM_FUNCTION:
    names = {"encdec": "EncDec", "graph": "Graph"}
    if node.arg not in names: raise UnsupportedDivergenceError(f"CUSTOM_FUNCTION {node.arg!r} has no Svod variant")
    return {"kind": "custom_function", "kind_name": names[node.arg]}
  if node.op is Ops.SOURCE: return {"kind": "source", "code": node.arg}
  if node.op is Ops.BINARY:
    if verbose: return {"kind": "binary", "length": len(node.arg)}
    raise CanonicalSerializationError("BINARY content is diagnostics-only; use verbose canonical serialization")
  if node.op is Ops.INS:
    if isinstance(node.arg, str): opcode = node.arg
    elif hasattr(node.arg, "name") and isinstance(node.arg.name, str): opcode = node.arg.name
    else: raise CanonicalSerializationError(f"unsupported INS opcode type: {type(node.arg).__name__}")
    return {"kind": "ins", "opcode": opcode, "attributes": []}
  if node.arg is not None:
    raise CanonicalSerializationError(f"unsupported {node.op.name} metadata type: {type(node.arg).__name__}")
  if node.arg is None and node.op.name in NO_ARG_OPS: return {"kind": "none"}
  if node.arg is None: raise UnsupportedDivergenceError(f"{node.op.name} has no Svod canonical operation")
  raise AssertionError("unreachable")


def canonical_dependencies(node: UOp) -> tuple[UOp, ...]:
  dependencies = list(canonical_sources(node))
  if node._shape is not None: dependencies.extend(dim for dim in node.shape if isinstance(dim, UOp))
  if node.op is Ops.PROGRAM and isinstance(node.arg, ProgramInfo):
    dependencies.extend(value for value in node.arg.global_size if isinstance(value, UOp) and value.op is not Ops.CONST)
    if node.arg.local_size is not None:
      dependencies.extend(value for value in node.arg.local_size if isinstance(value, UOp) and value.op is not Ops.CONST)
    dependencies.extend(node.arg.vars)
  return tuple(dependencies)


def canonical_sources(node: UOp) -> tuple[UOp, ...]:
  # PAD's metadata UOps encode output extents in Tinygrad and end padding in
  # Svod. canonical_arg retains the equivalent logical begin/end values.
  return node.src[:1] if node.op is Ops.PAD else node.src


def canonical_toposort(roots: tuple[UOp, ...]) -> list[UOp]:
  topo: list[UOp] = []
  visited: set[UOp] = set()
  active: set[UOp] = set()
  stack = [(root, False) for root in reversed(roots)]
  while stack:
    node, processed = stack.pop()
    if node in visited: continue
    if processed:
      active.remove(node)
      visited.add(node)
      topo.append(node)
      continue
    if node in active: raise CanonicalSerializationError("cycle through shape or PROGRAM metadata")
    active.add(node)
    stack.append((node, True))
    stack.extend((dependency, False) for dependency in reversed(canonical_dependencies(node)) if dependency not in visited)
  return topo


def canonical_graph(stage: str, roots: Iterable[UOp], verbose: bool = False) -> dict[str, Any]:
  roots = tuple(roots)
  topo = canonical_toposort(roots)
  ids = {node: index for index, node in enumerate(topo)}

  nodes = []
  for node_id, node in enumerate(topo):
    shape = None if node._shape is None else []
    for dim in () if node._shape is None else node.shape:
      shape.append({"kind": "symbolic", "node": ids[dim]} if isinstance(dim, UOp) else {"kind": "const", "value": dim})
    nodes.append({
      "id": node_id,
      "op": node.op.name,
      "dtype": canonical_dtype(node.dtype),
      "shape": shape,
      "arg": canonical_arg(node, ids, stage, verbose),
      "src": [ids[source] for source in canonical_sources(node)],
    })

  graph = {"schema_version": 8, "stage": stage, "roots": [ids[root] for root in roots], "nodes": nodes}
  if verbose:
    graph["verbose"] = [
      {
        "id": ids[node], "object_id": id(node), "tag": repr(node.tag), "backend_dtype": repr(node.dtype),
        **({"content_sha256": hashlib.sha256(node.arg).hexdigest()} if node.op is Ops.BINARY else {}),
      } for node in topo
    ]
  return graph


def canonical_binding(variable: UOp, value: int, schedule_loop: bool) -> dict[str, Any]:
  if variable.op is not Ops.PARAM or not isinstance(variable.arg, ParamArg) or variable.arg.name is None:
    raise CanonicalSerializationError("schedule binding must identify a named PARAM")
  if not isinstance(value, int) or isinstance(value, bool) or value < -(2**63) or value > 2**63-1:
    raise CanonicalSerializationError("schedule binding value must fit i64")
  return {
    "kind": "param", "slot": variable.arg.slot, "name": variable.arg.name, "dtype": canonical_dtype(variable.dtype),
    "value": value, "schedule_loop": schedule_loop,
  }


def canonical_schedule(kernel_graph: UOp, linear: UOp, var_vals: dict[str, int]) -> dict[str, Any]:
  if linear.op is not Ops.LINEAR: raise CanonicalSerializationError("Tinygrad create_linear_with_vars must return LINEAR")
  original_calls = [node for node in kernel_graph.toposort() if node.op is Ops.CALL]
  descriptor_by_body = {call.src[0]: index for index, call in enumerate(original_calls)}
  original_by_body = {call.src[0]: call for call in original_calls}

  def dependency_bodies(call: UOp) -> set[UOp]:
    dependencies: set[UOp] = set()
    seen: set[UOp] = set()

    def visit(node: UOp) -> None:
      if node in seen: return
      seen.add(node)
      if node.op is Ops.CALL:
        dependencies.add(node.src[0])
        return
      if node.op is Ops.END and node.src and node.src[0].op is Ops.CALL:
        dependencies.add(node.src[0].src[0])
        return
      if node.op is Ops.AFTER:
        for source in node.src[1:]: visit(source)
        visit(node.src[0])
        return
      for source in node.src: visit(source)

    for source in call.src[1:]: visit(source)
    return dependencies

  def output_slots(ast: UOp) -> list[int]:
    slots = {
      node.src[0].buf_uop.arg.slot for node in ast.toposort()
      if node.op is Ops.STORE and isinstance(node.src[0].buf_uop.arg, ParamArg)
    }
    return sorted(slots)

  latest_item_by_body: dict[UOp, int] = {}
  scheduled_buffers: list[list[UOp]] = []
  scheduled_original_buffers: list[list[UOp]] = []
  items = []
  for order, call in enumerate(linear.src):
    if call.op is not Ops.CALL or not call.src:
      raise CanonicalSerializationError(f"LINEAR item {order} must be CALL with an AST")
    ast = call.src[0]
    if ast not in descriptor_by_body:
      raise CanonicalSerializationError(f"LINEAR item {order} AST is absent from scheduler input")
    buffers = [source.buf_uop for source in call.src[1:]]
    buffer_descriptors = []
    for argument_index, buffer in enumerate(buffers):
      if buffer.op not in {Ops.PARAM, Ops.BUFFER} or not isinstance(buffer.arg, ParamArg):
        raise CanonicalSerializationError(f"LINEAR buffer {argument_index} must resolve to PARAM or BUFFER")
      buffer_descriptors.append({
        "argument_index": argument_index,
        "global_slot": argument_index,
        "buffer_slot": buffer.arg.slot,
        "origin": buffer.op.name,
      })
    dependencies = sorted({
      latest_item_by_body[body] for body in dependency_bodies(original_by_body[ast]) if body in latest_item_by_body
    })
    bindings = [canonical_binding(variable, var_vals[variable.expr], False)
                for variable in ast.variables() if variable.expr in var_vals]
    bindings.sort(key=lambda binding: (binding["name"], binding["kind"], binding["slot"]))
    items.append({
      "order": order,
      "callable_index": descriptor_by_body[ast],
      "ast": canonical_graph("kernel_ast", (ast,)),
      "buffers": buffer_descriptors,
      "output_slots": output_slots(ast),
      "dependencies": dependencies,
      "bindings": bindings,
    })
    scheduled_buffers.append(buffers)
    scheduled_original_buffers.append([source.buf_uop for source in original_by_body[ast].src[1:] if source.op is not Ops.BIND])
    latest_item_by_body[ast] = order

  roots = tuple(root for root in kernel_graph.src if root.op is not Ops.BIND) if kernel_graph.op is Ops.SINK else (kernel_graph,)
  schedule_outputs = []
  for output in roots:
    output_buffer = output.buf_uop
    for item in range(len(scheduled_original_buffers)-1, -1, -1):
      if output_buffer in scheduled_original_buffers[item]:
        schedule_outputs.append({"item": item, "buffer": scheduled_original_buffers[item].index(output_buffer)})
        break
    else: raise CanonicalSerializationError("scheduler output is absent from LINEAR buffer arguments")
  return {"schema_version": 8, "stage": "scheduled", "items": items, "output_slots": schedule_outputs}


def fixture(name: str) -> UOp:
  if name == "weak_int_add": return UOp.const(7) + UOp.const(2, dtypes.int32)
  if name == "weak_float_neg_zero": return UOp.const(-0.0) + UOp.const(1.0, dtypes.float32)
  if name == "invalid_where":
    return UOp.const(True).where(UOp.const(1.0, dtypes.float16), UOp.invalid())
  if name == "scalar_stack":
    return UOp(Ops.STACK, src=(UOp.const(1, dtypes.int32), UOp.const(2, dtypes.int32)))
  if name == "shaped_stack":
    return UOp.stack(UOp.stack(UOp.const(1, dtypes.int32), UOp.const(2, dtypes.int32)),
                     UOp.stack(UOp.const(3, dtypes.int32), UOp.const(4, dtypes.int32)))
  if name == "buffer":
    from tinygrad.dtype import AddrSpace
    return UOp(Ops.BUFFER, src=(shape_to_shape_arg((8,)),),
               arg=ParamArg(4, dtypes.float32, device="CPU", addrspace=AddrSpace.GLOBAL))
  if name in {"scalar_load", "gated_load"}:
    from tinygrad.dtype import AddrSpace
    param = UOp(Ops.PARAM, src=(shape_to_shape_arg((16,)),), arg=ParamArg(0, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    index = UOp.const(3)
    indexed = param.index(index)
    if name == "scalar_load": return indexed.load()
    valid = index < UOp.const(5)
    return indexed.load(UOp.const(0.0, dtypes.float32), valid)
  if name in {"range_split_outer", "range_split_inner", "range_split_nested"}:
    paths = {
      "range_split_outer": ((5, 0), 4),
      "range_split_inner": ((5, 1), 2),
      "range_split_nested": ((5, 1, 0), 3),
    }
    path, end = paths[name]
    return UOp(Ops.RANGE, dtypes.weakint, src=(UOp.const(end),), arg=path+(AxisType.WEAK,))
  if name == "scalar_store":
    from tinygrad.dtype import AddrSpace
    output = UOp(Ops.PARAM, src=(shape_to_shape_arg((8,)),), arg=ParamArg(0, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    return output.index(UOp.const(2)).store(UOp.const(3.0, dtypes.float32))
  if name == "mixed_valid_load":
    from tinygrad.dtype import AddrSpace
    input_ = UOp(Ops.PARAM, src=(shape_to_shape_arg((8,)),), arg=ParamArg(0, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    index = UOp.const(3)
    valid = UOp(Ops.CMPLT, src=(index, UOp.const(4)))
    return UOp.stack(input_.index(UOp.const(2)).load(), input_.index(index.valid(valid)).load())
  if name == "copy":
    from tinygrad.dtype import AddrSpace
    input_ = UOp(Ops.PARAM, src=(shape_to_shape_arg((8,)),),
                 arg=ParamArg(0, dtypes.float32, device="CPU", addrspace=AddrSpace.GLOBAL))
    return UOp(Ops.COPY, src=(input_,), arg="CUDA:0")
  if name == "allreduce":
    from tinygrad.dtype import AddrSpace
    input_ = UOp(Ops.PARAM, src=(shape_to_shape_arg((8,)),), arg=ParamArg(0, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    return UOp(Ops.ALLREDUCE, src=(input_,), arg=(Ops.ADD, "CPU"))
  if name == "multi_output_call":
    body = UOp(Ops.TUPLE, src=(UOp.const(1, dtypes.int32), UOp.const(2, dtypes.int32)))
    return UOp(Ops.CALL, src=(body,), arg=CallInfo(name="pair"))
  if name == "padded_reduction":
    source = UOp.stack(UOp.const(1.0, dtypes.float32), UOp.const(2.0, dtypes.float32), UOp.const(3.0, dtypes.float32))
    return source.pad(((1, 2),))._rop(Ops.ADD, (0,))
  if name == "local_wmma_staging":
    from tinygrad.dtype import AddrSpace
    from tinygrad.schedule.indexing import BufferizeOpts
    a = UOp.stack(UOp.const(1.0, dtypes.float16)).bufferize(arg=BufferizeOpts(None, AddrSpace.LOCAL))
    b = UOp.stack(UOp.const(2.0, dtypes.float16)).bufferize(arg=BufferizeOpts(None, AddrSpace.LOCAL))
    return UOp.wmma(a, b, UOp.stack(UOp.const(0.0, dtypes.float32)), (16, 8, 16), "CPU", 1)
  if name == "symbolic_function":
    from tinygrad.dtype import AddrSpace
    formal_dim = UOp.param(1, dtypes.int32, shape=(), vmin_vmax=(1, 8), multiple_of=1, name="n", addrspace=AddrSpace.ALU)
    formal_extent = formal_dim * formal_dim.const_like(2) + formal_dim.const_like(1)
    formal = UOp.param(0, dtypes.float32, shape=(formal_extent,), addrspace=AddrSpace.GLOBAL)
    body = UOp.maketuple(formal, formal.sqrt())

    actual_dim = UOp.variable("m", 1, 8, dtypes.int32)
    actual_extent = actual_dim * actual_dim.const_like(2) + actual_dim.const_like(1)
    actual = UOp.param(7, dtypes.float32, shape=(actual_extent,), addrspace=AddrSpace.GLOBAL)
    return body.call(actual, actual_dim, name="symbolic").gettuple(1)
  if name == "program_info":
    from dataclasses import replace
    from tinygrad.dtype import AddrSpace
    from tinygrad.helpers import Target
    input_ = UOp(Ops.PARAM, src=(shape_to_shape_arg((16,)),), arg=ParamArg(0, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    output = UOp(Ops.PARAM, src=(shape_to_shape_arg((16,)),), arg=ParamArg(2, dtypes.float32, addrspace=AddrSpace.GLOBAL))
    index = UOp.const(0)
    load = UOp(Ops.LOAD, src=(UOp(Ops.INDEX, src=(input_, index)),))
    store = UOp(Ops.STORE, src=(UOp(Ops.INDEX, src=(output, index)), load))
    variable = UOp.variable("n", 1, 16, dtypes.int32)
    variable = variable.replace(arg=replace(variable.arg, slot=1))
    sink = UOp.sink(store, variable, arg=KernelInfo(name="non_default"))
    info = replace(ProgramInfo.from_sink(sink, Target(device="CUDA:1")), global_size=(variable, 2, 1), local_size=(4, 1, 1))
    return UOp(Ops.PROGRAM, src=(sink,), arg=info)
  raise ValueError(f"unknown fixture {name!r}")


def production_stage(stage: str, body_index: int = 0, multi_output: bool = False) -> UOp | dict[str, Any]:
  """Run the stage fixture cumulatively through pinned Tinygrad production APIs."""
  import importlib
  from tinygrad.dtype import AddrSpace
  from tinygrad.helpers import Target
  from tinygrad.renderer import Renderer

  rangeify_mod = importlib.import_module("tinygrad.schedule.rangeify")
  schedule_mod = importlib.import_module("tinygrad.schedule")
  codegen_mod = importlib.import_module("tinygrad.codegen")

  input_ = UOp(Ops.PARAM, src=(shape_to_shape_arg((64,)),),
               arg=ParamArg(0, dtypes.float32, device="CPU", addrspace=AddrSpace.GLOBAL))
  schedule_bound = None
  add = input_ + input_.const_like(1.0)
  if not multi_output:
    schedule_variable = UOp.variable("schedule_n", 1, 8, dtypes.int32)
    schedule_bound = schedule_variable.bind(4)
    add = add + schedule_bound.cast(dtypes.float32)
  add = add.contiguous()
  tensor = UOp.sink(add, (input_ * input_.const_like(2.0)).contiguous()) if multi_output else \
    UOp.sink((add * add.const_like(2.0)).contiguous())
  if stage == "tensor": return tensor

  captures: dict[str, UOp] = {}
  original_graph_rewrite = rangeify_mod.graph_rewrite
  original_run_rangeify = rangeify_mod.run_rangeify

  def capture_rangeify_rewrite(root, matcher, *args, **kwargs):
    result = original_graph_rewrite(root, matcher, *args, **kwargs)
    if kwargs.get("name") == "limit buffers": captures["rangeified"] = result
    return result

  def capture_run_rangeify(*args, **kwargs):
    result = original_run_rangeify(*args, **kwargs)
    captures["range_assignment"] = result[0]
    return result

  rangeify_mod.graph_rewrite = capture_rangeify_rewrite
  rangeify_mod.run_rangeify = capture_run_rangeify
  try:
    kernel_graph = rangeify_mod.get_kernel_graph(tensor)
  finally:
    rangeify_mod.graph_rewrite = original_graph_rewrite
    rangeify_mod.run_rangeify = original_run_rangeify

  if stage == "rangeified": return captures["rangeified"]
  if multi_output:
    calls = [node for node in kernel_graph.toposort() if node.op is Ops.CALL]
    if len(calls) < 2: raise RuntimeError("multi-output production fixture must callify both outputs")
    return kernel_graph
  if stage == "kernel_ast": return kernel_graph
  if stage == "scheduled":
    assert schedule_bound is not None
    scheduling_sink = UOp.sink(*tensor.src, schedule_bound)
    linear, var_vals = schedule_mod.create_linear_with_vars(scheduling_sink)
    return canonical_schedule(kernel_graph, linear, var_vals)

  bodies = [node.src[0] for node in kernel_graph.toposort() if node.op is Ops.CALL]
  if body_index >= len(bodies): raise RuntimeError(f"kernel index {body_index} is absent from production fixture")
  ast = bodies[body_index]
  renderer = Renderer(Target("CPU"))

  original_codegen_rewrite = codegen_mod.graph_rewrite

  def capture_codegen_rewrite(root, matcher, *args, **kwargs):
    name = kwargs.get("name")
    if name == "postopt symbolic":
      captures["optimized"] = root
      captures["postrange"] = root
    result = original_codegen_rewrite(root, matcher, *args, **kwargs)
    if name == "expander": captures["expanded"] = result
    if name == "add images": captures["coalesced"] = result
    if name == "move gates from index": captures["gated"] = result
    return result

  codegen_mod.graph_rewrite = capture_codegen_rewrite
  try:
    final_sink = codegen_mod.full_rewrite_to_sink(ast, renderer)
  finally:
    codegen_mod.graph_rewrite = original_codegen_rewrite

  if stage in captures: return captures[stage]
  info = ProgramInfo.from_sink(final_sink, renderer.target)
  program = UOp(Ops.PROGRAM, src=(final_sink,), arg=info)
  if stage == "program": return program
  if stage == "linearized": return codegen_mod.do_linearize(renderer, program, final_sink)
  raise ValueError(f"unknown production stage {stage!r}")


def _evid02_dtype(dtype: DType) -> str:
  if dtype is dtypes.half: return "float16"
  if dtype is dtypes.float: return "float32"
  return dtype.name


def _evid02_arg(node: UOp) -> Any:
  if node.op is Ops.CONST:
    kind = "bool" if isinstance(node.arg, bool) else "int" if isinstance(node.arg, int) else "float"
    return {"kind": kind, "value": node.arg}
  if node.op is Ops.PARAM and isinstance(node.arg, ParamArg): return {"slot": node.arg.slot}
  if node.op is Ops.SPECIAL: return {"name": node.arg}
  if node.op is Ops.WMMA:
    dims, dtype_in, device, threads, upcast_axes = node.arg
    return {"dims": list(dims), "input_dtype": _evid02_dtype(dtype_in), "device": device,
            "threads": threads, "upcast_axes": upcast_axes}
  return None


def _evid02_graph(name: str, root: UOp) -> dict[str, Any]:
  nodes = list(root.toposort())
  ids = {node: index for index,node in enumerate(nodes)}
  table = []
  for node_id,node in enumerate(nodes):
    shape = None if node._shape is None else list(node.shape)
    if shape is not None and not all(isinstance(extent, int) for extent in shape):
      raise CanonicalSerializationError("EVID-02 requires constant node shapes")
    table.append({"id": node_id, "op": node.op.name, "dtype": _evid02_dtype(node.dtype), "shape": shape,
                  "src": [ids[source] for source in node.src], "arg": _evid02_arg(node)})
  return {"name": name, "root": ids[root], "nodes": table}


def evid02_safety() -> dict[str, Any]:
  """Serialize pinned production source graphs without deriving safety claims."""
  import tinygrad.codegen as codegen_mod
  from dataclasses import replace
  from tinygrad import Tensor
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer

  # The frontend graph is target-independent. Force its lazy buffers onto NULL
  # so evidence generation never probes or opens host GPU device nodes; the
  # production optimizer below still receives the explicit gfx1151 renderer.
  with Context(DEV="NULL"):
    a, b = Tensor.empty(5, 16, dtype=dtypes.half), Tensor.empty(16, 16, dtype=dtypes.half)
    ast = a.matmul(b, dtype=dtypes.float).schedule_linear().src[-1].src[0]
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 2, 1)),)))
  renderer = AMDLLVMRenderer(Target("AMD", arch="gfx1151"))
  captures: dict[str, UOp] = {}
  original_graph_rewrite = codegen_mod.graph_rewrite
  def capture_final_rewrite(root, matcher, *args, **kwargs):
    result = original_graph_rewrite(root, matcher, *args, **kwargs)
    if kwargs.get("name") == "final rewrite": captures["late-final-rewrite"] = result
    return result
  codegen_mod.graph_rewrite = capture_final_rewrite
  try: sink = codegen_mod.full_rewrite_to_sink(ast, renderer)
  finally: codegen_mod.graph_rewrite = original_graph_rewrite
  final_rewrite = captures.get("late-final-rewrite")
  if final_rewrite is None: raise RuntimeError("pinned full_rewrite_to_sink did not expose final rewrite")
  program = UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink, renderer.target))
  linear = codegen_mod.do_linearize(renderer, program, sink).src[1]
  return {"schema_version": 2, "evidence": "EVID-02", "reference": TARGET_COMMIT,
          "fixture": {"m": 5, "k": 16, "n": 16, "input_dtype": "float16", "accumulator_dtype": "float32", "target": "gfx1151"},
          "stages": [_evid02_graph("late-final-rewrite", final_rewrite), _evid02_graph("linearized", linear)]}


def self_test() -> None:
  from dataclasses import replace
  from tinygrad.dtype import AddrSpace
  from tinygrad.helpers import Target

  symbolic = UOp.variable("shape_n", 1, 8, dtypes.int32)
  shaped = UOp(Ops.NOOP, dtypes.float32)
  shaped.__dict__["_RECURSIVE_PROPERTY__shape"] = (symbolic,)
  graph = canonical_graph("self-test", (shaped,))
  symbolic_id = next(node["id"] for node in graph["nodes"] if node["op"] == "PARAM")
  assert graph["nodes"][-1]["shape"] == [{"kind": "symbolic", "node": symbolic_id}]
  assert all(dim.get("node") is not None for node in graph["nodes"] for dim in node["shape"] or [] if dim["kind"] == "symbolic")

  param = UOp(Ops.PARAM, src=(shape_to_shape_arg((4,)),), arg=ParamArg(0, dtypes.float32, device=("CPU", "CUDA:0")))
  try: canonical_graph("self-test", (param,))
  except UnsupportedDivergenceError: pass
  else: raise AssertionError("tuple devices must be rejected")

  sink = UOp.sink(arg=KernelInfo(axis_types=(AxisType.GLOBAL,)))
  try: canonical_graph("self-test", (sink,))
  except UnsupportedDivergenceError: pass
  else: raise AssertionError("unsupported KernelInfo fields must be rejected")

  implicit = UOp.sink(arg=KernelInfo())
  assert canonical_graph("kernel_ast", (implicit,))["nodes"][-1]["arg"]["name"] is None
  assert canonical_graph("tensor", (implicit,))["nodes"][-1]["arg"]["name"] == "test"
  named = UOp.sink(arg=KernelInfo(name="\x1b[31mexplicit\x1b[0m"))
  assert canonical_graph("kernel_ast", (named,))["nodes"][-1]["arg"]["name"] == "explicit"

  program = UOp(Ops.PROGRAM, src=(UOp.sink(),), arg=ProgramInfo(target=Target("CPU", renderer="CLANG")))
  try: canonical_graph("self-test", (program,))
  except UnsupportedDivergenceError: pass
  else: raise AssertionError("unsupported Target fields must be rejected")

  variable = UOp.variable("launch_n", 1, 8, dtypes.int32)
  variable = variable.replace(arg=replace(variable.arg, slot=0))
  program = UOp(Ops.PROGRAM, src=(UOp.sink(),), arg=ProgramInfo(global_size=(variable, 1, 1), vars=(variable,), target=Target("CPU")))
  graph = canonical_graph("self-test", (program,))
  assert graph["nodes"][-1]["arg"]["global_size"][0]["kind"] == "node"

  binary = UOp(Ops.BINARY, arg=b"canonical")
  try: canonical_graph("self-test", (binary,))
  except CanonicalSerializationError: pass
  else: raise AssertionError("non-verbose BINARY must be rejected")
  assert "content_sha256" in canonical_graph("self-test", (binary,), verbose=True)["verbose"][0]

  from tinygrad.schedule.indexing import BufferizeOpts
  padded = UOp.stack(UOp.const(1.0), UOp.const(2.0), UOp.const(3.0)).pad(((1, 2),))
  pad = canonical_graph("self-test", (padded,))["nodes"][-1]
  assert pad["arg"] == {"kind": "pad", "begin": [1], "end": [2]} and len(pad["src"]) == 1
  staged = UOp.const(1.0, dtypes.float32).bufferize(arg=BufferizeOpts("CPU", AddrSpace.GLOBAL, False))
  assert canonical_graph("self-test", (staged,))["nodes"][-1]["arg"] == {
    "kind": "stage", "device": "CPU", "local_axis": None, "address_space": "global", "removable": False,
  }
  ins = UOp(Ops.INS, dtypes.int32, src=(UOp.const(1, dtypes.int32),), arg="mock.mov")
  assert canonical_graph("self-test", (ins,))["nodes"][-1]["arg"]["opcode"] == "mock.mov"
  call = UOp(Ops.CALL, src=(UOp.custom_function("graph"),), arg=CallInfo(name="named", precompile=True))
  assert canonical_graph("self-test", (call,))["nodes"][-1]["arg"]["metadata"] == []
  aux_call = UOp(Ops.CALL, src=(UOp.custom_function("graph"),), arg=CallInfo(name="named", aux=("a", "b")))
  try: canonical_graph("self-test", (aux_call,))
  except UnsupportedDivergenceError: pass
  else: raise AssertionError("CALL aux strings must be rejected")
  a, b, c = (UOp.stack(UOp.const(value, dtypes.float16)) for value in (1.0, 2.0, 0.0))
  wmma = UOp.wmma(a, b, c, (16, 8, 16), "CPU", 1, (((3, 4),), ((4, 2),), ((5, 8),)))
  assert canonical_graph("self-test", (wmma,))["nodes"][-1]["arg"]["upcast_a"][0]["axis"]["path"] == [3]
  schedule = production_stage("scheduled")
  assert isinstance(schedule, dict) and schedule["stage"] == "scheduled" and len(schedule["items"]) == 2
  assert schedule["items"][0]["ast"]["stage"] == "kernel_ast"
  assert schedule["items"][0]["bindings"] == [{
    "kind": "param", "slot": -1, "name": "schedule_n", "dtype": {"kind": "scalar", "name": "int32"},
    "value": 4, "schedule_loop": False,
  }]
  assert schedule["items"][1]["bindings"] == [] and schedule["items"][1]["dependencies"] == [0]
  assert schedule["output_slots"] == [{"item": 1, "buffer": 0}]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("fixture", nargs="?", choices=("weak_int_add", "weak_float_neg_zero", "invalid_where", "scalar_stack", "shaped_stack",
                                                     "buffer", "scalar_load", "gated_load", "scalar_store", "mixed_valid_load", "copy", "allreduce",
                                                     "multi_output_call", "symbolic_function", "padded_reduction", "local_wmma_staging",
                                                     "range_split_outer", "range_split_inner", "range_split_nested", "program_info"))
  parser.add_argument("--stage", default="tensor")
  parser.add_argument("--production-stage", choices=("tensor", "scheduled", "rangeified", "kernel_ast", "optimized", "postrange",
                                                        "expanded", "coalesced", "gated", "linearized", "program"))
  parser.add_argument("--production-multi-output", action="store_true")
  parser.add_argument("--kernel-index", type=int, default=0)
  parser.add_argument("--verbose", action="store_true")
  parser.add_argument("--self-test", action="store_true")
  parser.add_argument("--evid02-safety", action="store_true")
  args = parser.parse_args()
  verify_target()
  if args.self_test:
    self_test()
    return
  if args.evid02_safety:
    root, stage = evid02_safety(), "evid02-safety"
  elif args.production_multi_output:
    root, stage = production_stage("kernel_ast", multi_output=True), "kernel_ast"
  elif args.production_stage is not None:
    root, stage = production_stage(args.production_stage, args.kernel_index), args.production_stage
  else:
    if args.fixture is None: parser.error("fixture is required")
    root, stage = fixture(args.fixture), args.stage
  document = root if isinstance(root, dict) else canonical_graph(stage, (root,), verbose=args.verbose)
  json.dump(document, sys.stdout, indent=2)
  print()


if __name__ == "__main__": main()
