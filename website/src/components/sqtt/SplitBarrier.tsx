import { useId, type ReactNode } from "react";
import { Figure, Hatch, labelWidth, Legend, styles } from "./Figure";

export type SplitBarrierLabels = {
  joined: string;
  split: string;
  wave: string;
  work: string;
  parked: string;
  next: string;
  barrier: string;
  signal: string;
  wait: string;
  release: string;
  joinedNext: string;
};

const DEFAULTS: SplitBarrierLabels = {
  joined: "joined: one s_barrier",
  split: "split: s_barrier_signal, then s_barrier_wait",
  wave: "wave",
  work: "work",
  parked: "parked at the barrier",
  next: "next trip",
  barrier: "s_barrier",
  signal: "s_barrier_signal, after the LDS commit",
  wait: "s_barrier_wait",
  release: "the last wave signals",
  joinedNext: "joined: next trip starts",
};

// A schematic in time units 0..100: each wave's work up to its last shared read, the slowest wave
// arriving last, and the same tail of work after it on both sides.
const ARRIVE = [52, 66, 58, 76];
const TAIL = 16;
const RELEASE = Math.max(...ARRIVE);
const [W, RIGHT, ROW, GAP, HEAD] = [420, 412, 14, 6, 22];
/** Where the joined version starts its next trip: the slowest wave starts its own no sooner split. */
const JOINED_NEXT = RELEASE + TAIL;

type Span = { from: number; to: number; color: string; next?: boolean };
type Wave = { spans: Span[]; signal?: number; wait?: number };

const work = (from: number, to: number): Span => ({ from, to, color: "var(--sqtt-matrix)" });
const parked = (from: number, to: number): Span => ({ from, to, color: "var(--sqtt-barrier)" });
const next = (from: number): Span => ({ from, to: 100, color: "var(--sqtt-matrix)", next: true });

const JOINED: Wave[] = ARRIVE.map((a) => ({
  spans: [work(0, a), parked(a, RELEASE), work(RELEASE, JOINED_NEXT), next(JOINED_NEXT)],
}));
const SPLIT: Wave[] = ARRIVE.map((a) => {
  const [wait, end] = [a + TAIL, Math.max(a + TAIL, RELEASE)];
  return { spans: [work(0, a), work(a, wait), parked(wait, end), next(end)], signal: a, wait };
});

export default function SplitBarrier({
  labels,
  caption = "A schematic, not trace data: four waves over one trip. Split, a wave that arrives early runs its tail instead of parking, and parks only if the slowest wave has not signalled by the time it waits. A dashed line marks where the joined version starts its next trip; the slowest wave starts its own no sooner.",
}: {
  labels?: Partial<SplitBarrierLabels>;
  caption?: ReactNode;
}) {
  const l = { ...DEFAULTS, ...labels };
  const hatch = `sqtt-next-${useId().replace(/:/g, "")}`;
  const names = ARRIVE.map((_, i) => `${l.wave} ${i}`);
  const left = labelWidth(names, 11);
  const x = (t: number) => left + ((RIGHT - left) * t) / 100;
  const panel = ARRIVE.length * (ROW + GAP) + HEAD;
  const draw = (title: string, waves: Wave[], top: number) => (
    <g>
      <text x={0} y={top + 13} fontSize={12} fontWeight={600}>
        {title}
      </text>
      {waves.map(({ spans, signal, wait }, i) => {
        const y = top + HEAD + i * (ROW + GAP);
        return (
          <g key={i}>
            <text x={left - 6} y={y + ROW / 2 + 4} fontSize={11} textAnchor="end" className={styles.muted}>
              {names[i]}
            </text>
            {spans
              .filter(({ from, to }) => to > from)
              .map(({ from, to, color, next }, j) => {
                const box = { x: x(from) + 0.5, y, width: x(to) - x(from) - 1, height: ROW, rx: 2 };
                return next ? (
                  <rect key={j} {...box} x={box.x + 0.5} y={y + 0.5} width={box.width - 1} height={ROW - 1} fill={`url(#${hatch})`} stroke={color} />
                ) : (
                  <rect key={j} {...box} fill={color} />
                );
              })}
            {signal !== undefined && (
              <path d={`M${x(signal) - 4.5},${y}h9l-4.5,7z`} fill="var(--sqtt-ink)" />
            )}
            {wait !== undefined && (
              <path d={`M${x(wait)},${y - 2}v${ROW + 4}`} stroke="var(--sqtt-ink)" strokeWidth={2.5} />
            )}
          </g>
        );
      })}
    </g>
  );
  const split = panel + 8;
  const rows = split + panel - GAP;
  // Under the split panel, a row each: the release's label, then the joined next trip's.
  const guide = (t: number, row: number, label: string, width: number, dash: string) => (
    <g>
      <path d={`M${x(t)},${split + HEAD - 4}V${rows + 14 * row + 2}`} stroke="var(--sqtt-ink)" strokeWidth={width} strokeDasharray={dash} />
      <text x={x(t) - 4} y={rows + 14 * row} fontSize={11} textAnchor="end">
        {label}
      </text>
    </g>
  );
  const bottom = rows + 28;
  return (
    <Figure caption={caption}>
      <svg className={styles.svg} viewBox={`0 0 ${W} ${bottom + 4}`} role="img" aria-label={`${l.joined}; ${l.split}`}>
        <defs>
          <Hatch id={hatch} color="var(--sqtt-matrix)" />
        </defs>
        {draw(l.joined, JOINED, 0)}
        <path
          d={`M${x(RELEASE)},${HEAD - 4}V${panel - GAP + 2}`}
          stroke="var(--sqtt-ink)"
          strokeWidth={2.5}
          strokeDasharray="4 3"
        />
        <text x={x(RELEASE) + 4} y={HEAD - 6} fontSize={11}>
          {l.barrier}
        </text>
        {draw(l.split, SPLIT, split)}
        {guide(RELEASE, 1, l.release, 1, "2 3")}
        {guide(JOINED_NEXT, 2, l.joinedNext, 1.75, "5 3")}
      </svg>
      <Legend
        items={[
          { color: "var(--sqtt-matrix)", label: l.work },
          { color: "var(--sqtt-barrier)", label: l.parked },
          { color: "var(--sqtt-matrix)", hatch: true, label: l.next },
          { color: "var(--sqtt-ink)", mark: "triangle", label: l.signal },
          { color: "var(--sqtt-ink)", mark: "tick", label: l.wait },
        ]}
      />
    </Figure>
  );
}
