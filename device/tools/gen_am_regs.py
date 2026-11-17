#!/usr/bin/env python3
"""One-shot generator for the AM driver's register tables.

Parses tinygrad's `autogen/am/regs.py` (a self-contained set of per-IP-version
dicts) and emits a VENDORED Rust source file `device/src/amd/am/regs_gen.rs`.
This is run MANUALLY when adding/updating an arch — the cargo build never
touches tinygrad (which is a sometimes-absent submodule). Re-run + commit the
output.

    python3 device/tools/gen_am_regs.py \
        [path/to/tinygrad/runtime/autogen/am] \
        [device/src/amd/am/regs_gen.rs]

Each tinygrad register is `name -> (offset, segment, {field: (lo, hi)})`; we
emit `RegDef { name, offset, segment, fields: &[RegField{name, lo, hi}] }` plus
an `AM_REG_MODULES` table of `(prefix, (maj,min,rev), &[RegDef])` that
`regs.rs::select` resolves by the discovered `ip_ver`, and the `PROFILER_REGS`
list that trims the profiler-only modules.
"""
import importlib.util
import os
import sys

# (rust_static_name, regs.py module attr, prefix, (maj, min, rev), allowlist)
# gfx1151 / Strix Halo, gfx1201 / RDNA4 (GC only, for the PM4 profilers) and
# gfx942 / MI300X (Aqua Vanjaram) target sets. The AM driver resolves its
# registers by name at runtime, so its modules are kept whole (`None`); a
# module only the profilers read keeps the `PROFILER_REGS` of its GC major.
# Widen this list + re-run to add another arch.
MODULES = [
    ("GC_11_5_0", "gc_11_5_0", "gc", (11, 5, 0), None),
    ("GC_12_0_0", "gc_12_0_0", "gc", (12, 0, 0), 12),
    ("MMHUB_3_3_0", "mmhub_3_3_0", "mmhub", (3, 3, 0), None),
    ("MP_14_0_2", "mp_14_0_2", "mp", (14, 0, 2), None),
    ("NBIO_7_11_0", "nbio_7_11_0", "nbio", (7, 11, 0), None),
    ("HDP_6_0_0", "hdp_6_0_0", "hdp", (6, 0, 0), None),
    ("OSSSYS_6_0_0", "osssys_6_0_0", "osssys", (6, 0, 0), None),
    ("GC_9_4_3", "gc_9_4_3", "gc", (9, 4, 3), None),
    ("MMHUB_1_8_0", "mmhub_1_8_0", "mmhub", (1, 8, 0), None),
    ("MP_13_0_0", "mp_13_0_0", "mp", (13, 0, 0), None),
    ("MP_11_0_0", "mp_11_0_0", "mp", (11, 0, 0), None),
    ("NBIO_7_9_0", "nbio_7_9_0", "nbio", (7, 9, 0), None),
    ("HDP_4_4_2", "hdp_4_4_2", "hdp", (4, 4, 2), None),
    ("OSSSYS_4_4_2", "osssys_4_4_2", "osssys", (4, 4, 2), None),
    ("SDMA_4_4_2", "sdma_4_4_2", "sdma", (4, 4, 2), None),
]

GFX11, GFX12, BOTH = (11,), (12,), (11, 12)
# Every GC register the PMC and SQTT profilers write or read: (name, its fields
# they name, the GC majors whose tables carry it). A register whose fields
# differ between majors takes one entry per field set.
PROFILER_REGS = [
    ("regGRBM_GFX_INDEX", ("se_index", "sa_index", "instance_index", "se_broadcast_writes",
                           "sa_broadcast_writes", "instance_broadcast_writes"), BOTH),
    ("regSPI_CONFIG_CNTL", ("ps_pkr_priority_cntl", "exp_priority_order", "gpr_write_priority",
                            "enable_sqg_bop_events", "enable_sqg_top_events"), BOTH),
    ("regCOMPUTE_THREAD_TRACE_ENABLE", ("thread_trace_enable",), BOTH),
    ("regSQ_THREAD_TRACE_BUF0_SIZE", ("size",), BOTH),
    ("regSQ_THREAD_TRACE_BUF0_SIZE", ("base_hi",), GFX11),
    ("regSQ_THREAD_TRACE_BUF0_BASE", (), GFX11),
    ("regSQ_THREAD_TRACE_BUF0_BASE_LO", (), GFX12),
    ("regSQ_THREAD_TRACE_BUF0_BASE_HI", (), GFX12),
    ("regSQ_THREAD_TRACE_MASK", ("wtype_include", "simd_sel", "wgp_sel", "sa_sel"), BOTH),
    ("regSQ_THREAD_TRACE_TOKEN_MASK", ("reg_include", "token_exclude", "bop_events_token_include"), BOTH),
    ("regSQ_THREAD_TRACE_TOKEN_MASK", ("exclude_barrier_wait",), GFX12),
    ("regSQ_THREAD_TRACE_CTRL", ("draw_event_en", "spi_stall_en", "sq_stall_en", "reg_at_hwm",
                                 "hiwater", "util_timer", "mode"), BOTH),
    ("regSQ_THREAD_TRACE_CTRL", ("rt_freq",), GFX11),
    ("regSQ_THREAD_TRACE_STATUS", ("finish_pending", "busy"), BOTH),
    ("regSQ_THREAD_TRACE_WPTR", (), BOTH),
    *[(f"regCOMPUTE_STATIC_THREAD_MGMT_SE{se}", (), BOTH) for se in range(8)],
    ("regCP_PERFMON_CNTL", ("perfmon_state", "perfmon_sample_enable"), GFX11),
    ("regCP_PERFMON_CNTL_1", ("perfmon_state", "perfmon_sample_enable"), GFX12),
    ("regSQ_PERFCOUNTER_CTRL", ("cs_en", "ps_en", "gs_en", "hs_en"), BOTH),
    ("regSQ_PERFCOUNTER_CTRL2", ("force_en", "vmid_en"), BOTH),
    ("regCOMPUTE_PERFCOUNT_ENABLE", ("perfcount_enable",), BOTH),
    # Counter `i` selects at the even SELECT `2i` and reads at LO `i`; gfx12 has eight LOs.
    *[(f"regSQ_PERFCOUNTER{2 * i}_SELECT", ("perf_sel",), BOTH) for i in range(8)],
    *[(f"regSQ_PERFCOUNTER{i}_LO", (), BOTH) for i in range(8)],
]


def load_regs(autogen_dir: str):
    path = os.path.join(autogen_dir, "regs.py")
    spec = importlib.util.spec_from_file_location("am_regs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    autogen_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        repo, "submodules/new_new_tinygrad/tinygrad/runtime/autogen/am")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        repo, "device/src/amd/am/regs_gen.rs")

    regs = load_regs(autogen_dir)
    lines = [
        "// @generated by device/tools/gen_am_regs.py from tinygrad autogen/am/regs.py.",
        "// DO NOT EDIT BY HAND. Re-run the generator (manually) to update.",
        "// Vendored so the build never depends on the tinygrad submodule.",
        "",
    ]
    total_regs = 0
    for static_name, attr, _prefix, _ver, major in MODULES:
        table = getattr(regs, attr)
        if major is not None:
            keep = {name for name, _, majors in PROFILER_REGS if major in majors}
            missing = keep - table.keys()
            assert not missing, f"{attr} lacks profiler registers {sorted(missing)}"
            table = {name: reg for name, reg in table.items() if name in keep}
        lines.append(f"/// `{attr}` — {len(table)} registers.")
        lines.append(f"pub static {static_name}: &[RegDef] = &[")
        for name, (offset, segment, fields) in table.items():
            fs = "".join(
                f"RegField{{name:{f!r},lo:{lo},hi:{hi}}}," for f, (lo, hi) in fields.items()
            ).replace("'", '"')
            lines.append(
                f'    RegDef{{name:{name!r},offset:{offset},segment:{segment},fields:&[{fs}]}},'.replace("'", '"')
            )
        lines.append("];")
        lines.append("")
        total_regs += len(table)

    lines.append("/// The GC registers the PMC and SQTT profilers name: `(name, fields, GC majors)`.")
    lines.append("pub static PROFILER_REGS: &[ProfilerReg] = &[")
    for name, fields, majors in PROFILER_REGS:
        fs = "".join(f'"{f}",' for f in fields)
        lines.append(f'    ("{name}", &[{fs}], &[{", ".join(map(str, majors))}]),')
    lines.append("];")
    lines.append("")

    lines.append("/// All vendored register modules, resolved by `regs::select(prefix, ip_ver)`.")
    lines.append("pub static AM_REG_MODULES: &[RegModule] = &[")
    for static_name, _attr, prefix, (maj, mn, rev), _major in MODULES:
        lines.append(f'    ("{prefix}", ({maj}, {mn}, {rev}), {static_name}),')
    lines.append("];")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}: {len(MODULES)} modules, {total_regs} registers")


if __name__ == "__main__":
    main()
