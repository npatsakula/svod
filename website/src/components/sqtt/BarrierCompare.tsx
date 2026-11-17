import type { ReactNode } from "react";
import diff from "@site/src/data/sqtt/nearest-barrier-compare.json";
import { cycles, Figure, labelWidth, Legend, styles } from "./Figure";

export type BarrierCompareLabels = {
  joined: string;
  split: string;
  raw: string;
  exposed: string;
  statement: string;
};

const DEFAULTS: BarrierCompareLabels = {
  joined: "joined",
  split: "split",
  raw: "raw",
  exposed: "exposed",
  statement: "statement",
};

const SIDES = [
  { key: "joined", color: "var(--sqtt-before)" },
  { key: "split", color: "var(--sqtt-barrier)" },
] as const;
const [W, RIGHT, BAR, GAP, GROUP] = [420, 372, 14, 3, 12];

export default function BarrierCompare({
  labels,
  caption = (d) =>
    `The barrier bucket of the main loop, cycles a trip, from tk_bench sqtt --compare on the two fixtures; untraced time ${d.us[0]} → ${d.us[1]} µs.`,
}: {
  labels?: Partial<BarrierCompareLabels>;
  /** Given each side's untraced time in µs, so that a translation types no number. */
  caption?: (d: { us: [string, string] }) => ReactNode;
}) {
  const l = { ...DEFAULTS, ...labels };
  const us = diff.kernel_ns.map((ns) => (ns / 1000).toFixed(0)) as [string, string];
  const text = caption({ us });
  const kinds = (["raw", "exposed"] as const).map((k) => ({ k, values: diff.barrier[k] }));
  const left = labelWidth([l.raw, l.exposed], 12);
  const max = Math.max(...diff.barrier.raw, ...diff.barrier.exposed);
  const scale = (RIGHT - left) / max;
  const group = 2 * BAR + GAP + GROUP;
  return (
    <Figure caption={text}>
      <svg className={styles.svg} viewBox={`0 0 ${W} ${kinds.length * group}`} role="img" aria-label={typeof text === "string" ? text : undefined}>
        {kinds.map(({ k, values }, g) => {
          const top = g * group;
          return (
            <g key={k}>
              <text x={left - 6} y={top + BAR + 5} fontSize={12} textAnchor="end">
                {l[k]}
              </text>
              {SIDES.map(({ key, color }, s) => {
                const [y, w] = [top + s * (BAR + GAP), values[s] * scale];
                return (
                  <g key={key}>
                    <rect x={left} y={y} width={w} height={BAR} rx={2} fill={color}>
                      <title>{`${l[key]}, ${l[k]}: ${cycles(values[s])}`}</title>
                    </rect>
                    <text x={left + w + 4} y={y + BAR - 3} fontSize={11}>
                      {cycles(values[s])}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <Legend items={SIDES.map(({ key, color }, s) => ({ color, label: `${l[key]} (${diff.fixtures[s]})` }))} />
      <table className={styles.table}>
        <thead>
          <tr>
            <th rowSpan={2}>{l.statement}</th>
            {SIDES.map(({ key }) => (
              <th key={key} colSpan={2}>
                {l[key]}
              </th>
            ))}
          </tr>
          <tr>
            {SIDES.flatMap(({ key }) => [
              <th key={`${key}-raw`}>{l.raw}</th>,
              <th key={`${key}-exp`}>{l.exposed}</th>,
            ])}
          </tr>
        </thead>
        <tbody>
          {diff.statements.map((st) => (
            <tr key={st.statement}>
              <td>
                <code>{st.statement}</code>
              </td>
              {[0, 1].flatMap((s) => [
                <td key={`r${s}`} className={styles.num}>
                  {cycles(st.raw[s])}
                </td>,
                <td key={`e${s}`} className={styles.num}>
                  {cycles(st.exposed[s])}
                </td>,
              ])}
            </tr>
          ))}
        </tbody>
      </table>
    </Figure>
  );
}
