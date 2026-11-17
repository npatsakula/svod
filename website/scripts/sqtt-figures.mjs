#!/usr/bin/env node
// Regenerates the profiling page's figure data (src/data/sqtt/*.json) from the committed SQTT
// fixtures: replays rdna4_nearest_split and compares rdna4_nearest_joined against it with
// tk_bench, then keeps only the fields the figures draw. Needs no GPU.
//
//   node website/scripts/sqtt-figures.mjs

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const fixtures = join(root, "device/src/test/fixtures/sqtt/svod");
const out = join(root, "website/src/data/sqtt");
const bench = join(process.env.CARGO_TARGET_DIR ?? join(root, "target"), "release/examples/tk_bench");

const run = (cmd, args) => execFileSync(cmd, args, { cwd: root, stdio: ["ignore", "pipe", "inherit"], maxBuffer: 1 << 28 });
const sqtt = (...args) => JSON.parse(run(bench, ["sqtt", ...args, "--json"]).toString());
const round = (x) => Math.round(x * 10) / 10;
const write = (name, data) => writeFileSync(join(out, name), JSON.stringify(data, null, 2) + "\n");

run("cargo", ["build", "--release", "-p", "svod-tk", "--example", "tk_bench"]);
mkdirSync(out, { recursive: true });

// A trip of the split capture's main loop by bucket group, in the report's bucket order. The load and
// LDS groups are the waits `load_exposed` and `lds_exposed` count (verdict.rs); a wait on anything
// else has no group in the figure and stops the script.
const split = sqtt("--load", join(fixtures, "rdna4_nearest_split"));
const loop = split.main_loop;
const vmem = new Set(["load", "sample", "bvh"]);
const lds = new Set(["ds", "lgkm"]);
const group = (bucket) => {
  if (bucket === "barrier") return "barrier";
  if (bucket.issue) return bucket.issue === "matrix" ? "matrix" : "issue";
  if (bucket.stall) return "stall";
  if (bucket.wait && vmem.has(bucket.wait.counter)) return "load";
  if (bucket.wait && lds.has(bucket.wait.counter)) return "lds";
  if (bucket.idle) return "idle";
  throw new Error(`unknown bucket ${JSON.stringify(bucket)}`);
};
const order = ["matrix", "issue", "stall", "load", "lds", "barrier", "idle"];
const sums = Object.fromEntries(order.map((k) => [k, [0, 0]]));
for (const [bucket, [raw, exposed]] of loop.partition.buckets) {
  const sum = sums[group(bucket)];
  sum[0] += raw;
  sum[1] += exposed;
}
const trips = loop.trips.total;
write("nearest-split-trip.json", {
  fixture: "rdna4_nearest_split",
  shape: split.labels.shape,
  trips,
  groups: order.map((key) => ({ key, raw: round(sums[key][0] / trips), exposed: round(sums[key][1] / trips) })),
});

// The barrier before and after the split, a trip, and the statements it became.
const diff = sqtt("--compare", join(fixtures, "rdna4_nearest_joined"), join(fixtures, "rdna4_nearest_split"));
if (diff.unit !== "trip") throw new Error(`the comparison is per ${diff.unit}, not per trip`);
const barrier = diff.buckets.find((b) => b.bucket === "barrier");
const pair = ([a, b]) => [round(a), round(b)];
write("nearest-barrier-compare.json", {
  fixtures: ["rdna4_nearest_joined", "rdna4_nearest_split"],
  kernel_ns: [diff.a.kernel_ns, diff.b.kernel_ns],
  barrier: { raw: pair(barrier.raw), exposed: pair(barrier.exposed) },
  statements: diff.statements
    .filter((s) => /^(barrier|arrive|await)\b/.test(s.statement))
    .map((s) => ({ statement: s.statement, raw: pair(s.raw), exposed: pair(s.exposed) })),
});
