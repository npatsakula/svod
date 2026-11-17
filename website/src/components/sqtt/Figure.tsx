import type { ReactNode } from "react";
import styles from "./styles.module.css";

/** A legend entry: a filled square, or one of the schematic's markers. */
export type Swatch = { color: string; label: ReactNode; opacity?: number; mark?: "triangle" | "tick"; hatch?: boolean };

const MARKS = { triangle: "M1,3H11L6,11Z", tick: "M6,0V12" };

/** The legend's swatch for a span drawn with {@link Hatch}. */
const hatched = (color: string) => `repeating-linear-gradient(135deg, ${color} 0 2px, transparent 2px 4.5px)`;

/** Diagonal stripes in `color` over a transparent ground, for a fill that reads in both themes. */
export function Hatch({ id, color }: { id: string; color: string }) {
  return (
    <pattern id={id} width={5} height={5} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width={2} height={5} fill={color} />
    </pattern>
  );
}

export function Figure({ caption, children }: { caption: ReactNode; children: ReactNode }) {
  return (
    <figure className={styles.figure}>
      {children}
      <figcaption className={styles.caption}>{caption}</figcaption>
    </figure>
  );
}

export function Legend({ items }: { items: Swatch[] }) {
  return (
    <ul className={styles.legend}>
      {items.map(({ color, label, opacity = 1, mark, hatch }, i) => (
        <li key={i}>
          {mark ? (
            <svg className={styles.swatch} viewBox="0 0 12 12" aria-hidden>
              <path d={MARKS[mark]} fill={color} stroke={color} strokeWidth={2.5} />
            </svg>
          ) : (
            <span
              className={styles.swatch}
              style={hatch ? { background: hatched(color), boxShadow: `inset 0 0 0 1px ${color}` } : { background: color, opacity }}
            />
          )}
          {label}
        </li>
      ))}
    </ul>
  );
}

/** An em for a CJK glyph, a generous average otherwise. */
const ems = (text: string) => [...text].reduce((w, c) => w + (c.codePointAt(0)! >= 0x2e80 ? 1 : 0.6), 0);

/** Room for the longest of `labels` set at `size` in an SVG. */
export const labelWidth = (labels: string[], size: number) => Math.ceil(Math.max(...labels.map(ems)) * size) + 8;

/** A trip's cycles as the report prints them. */
export const cycles = (x: number) => x.toFixed(1);

export { MARKS, styles };
