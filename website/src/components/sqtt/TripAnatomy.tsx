import type { ReactNode } from "react";
import trip from "@site/src/data/sqtt/nearest-split-trip.json";
import { cycles, Figure, labelWidth, styles } from "./Figure";

type Group = "matrix" | "issue" | "stall" | "load" | "lds" | "barrier" | "idle";
const groups = trip.groups as { key: Group; raw: number; exposed: number }[];

export type TripAnatomyLabels = Record<Group, string> & {
  raw: string;
  exposed: string;
  bucket: string;
  total: string;
};

const DEFAULTS: TripAnatomyLabels = {
  matrix: "matrix issue",
  issue: "other issue (VALU, SALU, LDS, …)",
  stall: "stall",
  load: "wait on global loads",
  lds: "wait on LDS",
  barrier: "barrier",
  idle: "idle",
  raw: "raw",
  exposed: "exposed",
  bucket: "bucket",
  total: "total",
};

export const BUCKET_COLORS: Record<Group, string> = {
  matrix: "var(--sqtt-matrix)",
  issue: "var(--sqtt-issue)",
  stall: "var(--sqtt-stall)",
  load: "var(--sqtt-load)",
  lds: "var(--sqtt-lds)",
  barrier: "var(--sqtt-barrier)",
  idle: "var(--sqtt-idle)",
};

const [W, RIGHT, ROW, GAP] = [420, 372, 22, 10];

export default function TripAnatomy({
  labels,
  caption = (d) =>
    `One main-loop trip of the rdna4_nearest_split fixture (${d.shape}): a traced wave's cycles by bucket, averaged over its ${d.trips} trips.`,
}: {
  labels?: Partial<TripAnatomyLabels>;
  /** Given the fixture's shape and trip count, so that a translation types no number. */
  caption?: (d: { shape: string; trips: number }) => ReactNode;
}) {
  const l = { ...DEFAULTS, ...labels };
  const text = caption({ shape: trip.shape, trips: trip.trips });
  const sum = (side: "raw" | "exposed") => groups.reduce((s, g) => s + g[side], 0);
  const rows = (["raw", "exposed"] as const).map((side) => ({ side, total: sum(side) }));
  const left = labelWidth([l.raw, l.exposed], 12);
  const scale = (RIGHT - left) / rows[0].total;
  const height = rows.length * (ROW + GAP);
  return (
    <Figure caption={text}>
      <svg className={styles.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={typeof text === "string" ? text : undefined}>
        {rows.map(({ side, total }, i) => {
          const y = i * (ROW + GAP);
          let at = left;
          return (
            <g key={side}>
              <text x={left - 6} y={y + ROW / 2 + 4} fontSize={12} textAnchor="end">
                {l[side]}
              </text>
              {groups.map((g) => {
                const w = g[side] * scale;
                const rect = (
                  <rect key={g.key} x={at} y={y} width={Math.max(w - 2, 0)} height={ROW} rx={2} fill={BUCKET_COLORS[g.key]}>
                    <title>{`${l[g.key]}: ${cycles(g[side])}`}</title>
                  </rect>
                );
                at += w;
                return rect;
              })}
              <text x={at + 4} y={y + ROW / 2 + 4} fontSize={11} className={styles.muted}>
                {total.toFixed(0)}
              </text>
            </g>
          );
        })}
      </svg>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>{l.bucket}</th>
            <th>{l.raw}</th>
            <th>{l.exposed}</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g.key}>
              <td>
                <span className={styles.swatch} style={{ background: BUCKET_COLORS[g.key], marginRight: "0.4rem" }} />
                {l[g.key]}
              </td>
              <td className={styles.num}>{cycles(g.raw)}</td>
              <td className={styles.num}>{cycles(g.exposed)}</td>
            </tr>
          ))}
          <tr>
            <th>{l.total}</th>
            <td className={styles.num}>{cycles(rows[0].total)}</td>
            <td className={styles.num}>{cycles(rows[1].total)}</td>
          </tr>
        </tbody>
      </table>
    </Figure>
  );
}
