import { useId, type ReactNode } from "react";
import { Figure, Hatch, Legend, labelWidth, styles } from "./Figure";

export type RawVsExposedLabels = {
  matrix: string;
  valu: string;
  load: string;
  barrier: string;
  hidden: string;
  exposed: string;
  raw: string;
  exposedSpan: string;
  time: string;
};

const DEFAULTS: RawVsExposedLabels = {
  matrix: "matrix",
  valu: "VALU",
  load: "load wait",
  barrier: "barrier",
  hidden: "raw, hidden by other pipes' work",
  exposed: "exposed: nothing else running",
  raw: "raw",
  exposedSpan: "exposed",
  time: "time on one SIMD →",
};

// A schematic in time units 0..100: busy spans per lane, and which of a wait's spans other pipes hid.
type Span = [number, number, boolean?];
type Lane = "matrix" | "valu" | "load" | "barrier";
const LANES: { key: Lane; color: string; spans: Span[] }[] = [
  { key: "matrix", color: "var(--sqtt-matrix)", spans: [[0, 26], [34, 60], [90, 100]] },
  { key: "valu", color: "var(--sqtt-issue)", spans: [[26, 34], [60, 64]] },
  { key: "load", color: "var(--sqtt-load)", spans: [[16, 64, true], [64, 74]] },
  { key: "barrier", color: "var(--sqtt-barrier)", spans: [[74, 90]] },
];
const WAIT: [number, number] = [16, 74];
const HIDDEN_END = 64;
// Where other pipes' work starts or stops while the wait is hidden: the overlap's edges.
const EDGES = [...new Set(LANES.filter(({ key }) => key !== "load").flatMap(({ spans }) => spans.flatMap(([a, b]) => [a, b])))]
  .filter((t) => t > WAIT[0] && t <= HIDDEN_END)
  .sort((a, b) => a - b);
const [W, RIGHT, ROW, GAP, BRACKETS] = [420, 412, 18, 8, 34];

type X = (t: number) => number;

function Bracket({ x, from, to, y, label }: { x: X; from: number; to: number; y: number; label: string }) {
  const [a, b] = [x(from), x(to)];
  return (
    <g>
      <path d={`M${a},${y - 4}V${y}H${b}V${y - 4}`} fill="none" stroke="var(--sqtt-muted)" />
      <text x={b + 4} y={y + 4} fontSize={11} className={styles.muted}>
        {label}
      </text>
    </g>
  );
}

export default function RawVsExposed({
  labels,
  caption = "A schematic, not trace data: a load wait counts in full as raw, but only its cycles with no other pipe busy are exposed; a barrier with nothing else running is exposed the same way.",
}: {
  labels?: Partial<RawVsExposedLabels>;
  caption?: ReactNode;
}) {
  const l = { ...DEFAULTS, ...labels };
  const hatch = `sqtt-hatch-${useId().replace(/:/g, "")}`;
  const left = labelWidth(LANES.map(({ key }) => l[key]), 12);
  const x: X = (t) => left + ((RIGHT - left) * t) / 100;
  // Each lane's top; the load wait's brackets take a band of their own under its lane.
  const tops: number[] = [];
  LANES.forEach((_, i) => tops.push(i ? tops[i - 1] + ROW + GAP + (LANES[i - 1].key === "load" ? BRACKETS : 0) : 4));
  const load = tops[LANES.findIndex(({ key }) => key === "load")];
  const axis = tops[tops.length - 1] + ROW + GAP;
  return (
    <Figure caption={caption}>
      <svg className={styles.svg} viewBox={`0 0 ${W} ${axis + 20}`} role="img" aria-label={`${l.raw} / ${l.exposedSpan}`}>
        <defs>
          <Hatch id={hatch} color="var(--sqtt-load)" />
        </defs>
        {EDGES.map((t) => (
          <line
            key={t}
            x1={x(t)}
            x2={x(t)}
            y1={tops[0] - 2}
            y2={load + ROW + 2}
            stroke="var(--sqtt-muted)"
            strokeOpacity={0.6}
            strokeDasharray="2 3"
          />
        ))}
        {LANES.map(({ key, color, spans }, i) => {
          const y = tops[i];
          return (
            <g key={key}>
              <text x={left - 6} y={y + ROW / 2 + 4} fontSize={12} textAnchor="end">
                {l[key]}
              </text>
              <line x1={left} x2={RIGHT} y1={y + ROW / 2} y2={y + ROW / 2} stroke="var(--sqtt-rule)" />
              {spans.map(([a, b, hidden], j) => {
                const box = { x: x(a) + 1, y, width: x(b) - x(a) - 2, height: ROW, rx: 3 };
                return hidden ? (
                  <rect key={j} {...box} x={box.x + 0.5} y={y + 0.5} width={box.width - 1} height={ROW - 1} fill={`url(#${hatch})`} stroke={color} />
                ) : (
                  <rect key={j} {...box} fill={color} />
                );
              })}
            </g>
          );
        })}
        <Bracket x={x} from={WAIT[0]} to={WAIT[1]} y={load + ROW + 12} label={l.raw} />
        <Bracket x={x} from={HIDDEN_END} to={WAIT[1]} y={load + ROW + 27} label={l.exposedSpan} />
        <line x1={left} x2={RIGHT} y1={axis} y2={axis} stroke="var(--sqtt-muted)" />
        <text x={RIGHT} y={axis + 14} fontSize={11} textAnchor="end" className={styles.muted}>
          {l.time}
        </text>
      </svg>
      <Legend
        items={[
          { color: "var(--sqtt-load)", hatch: true, label: l.hidden },
          { color: "var(--sqtt-load)", label: l.exposed },
        ]}
      />
    </Figure>
  );
}
