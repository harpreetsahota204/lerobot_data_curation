import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { signalColor, theme, verdictColor } from "./theme";
import { HistogramBar, Row } from "./types";

const axisStyle = { fontSize: 10, fill: theme.textDim };
const axisLabelStyle = { fill: theme.textDim, fontSize: 11 };
const tooltipStyle: React.CSSProperties = {
  background: theme.headerBg,
  border: `1px solid ${theme.cardBorder}`,
  borderRadius: 6,
  fontSize: 12,
  color: theme.text,
  padding: "6px 10px",
};

/** Compact tick text for large magnitudes. */
function fmtTick(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${(v / 1e6).toPrecision(3)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toPrecision(3)}k`;
  return Number(v).toPrecision(3).replace(/\.?0+$/, "");
}

/**
 * Bins several series on one shared grid (so series overlay directly).
 * Bins are half-open except the last. A plain array is a single series named "".
 */
export function histogram(input: number[] | Record<string, number[]>, bins = 20): HistogramBar[] {
  const series: Record<string, number[]> = Array.isArray(input) ? { "": input } : input;
  const pooled = Object.values(series).flat();
  if (pooled.length === 0) return [];
  let lo = Math.min(...pooled);
  let hi = Math.max(...pooled);
  if (hi === lo) {
    lo -= 0.5;
    hi += 0.5;
  }
  const width = (hi - lo) / bins;
  const out: HistogramBar[] = Array.from({ length: bins }, (_, i) => ({
    x: lo + width * (i + 0.5),
    x0: lo + width * i,
    x1: lo + width * (i + 1),
    counts: Object.fromEntries(Object.keys(series).map((k) => [k, 0])),
  }));
  for (const [name, values] of Object.entries(series)) {
    for (const v of values) {
      const i = Math.min(bins - 1, Math.floor((v - lo) / width));
      out[i].counts[name] += 1;
    }
  }
  return out;
}

export function inBin(value: number, bar: HistogramBar, isLast: boolean): boolean {
  return value >= bar.x0 && (value < bar.x1 || (isLast && value <= bar.x1));
}

function HistTooltip({ active, payload, signals }: any) {
  if (!active || !payload?.length) return null;
  const bar: HistogramBar = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div>
        {fmtTick(bar.x0)} to {fmtTick(bar.x1)}
      </div>
      {signals.map((s: string) => (
        <div key={s} style={{ color: s ? signalColor(signals, s) : theme.text }}>
          {s ? `${s}: ` : ""}
          {bar.counts[s] ?? 0} episode(s)
        </div>
      ))}
    </div>
  );
}

function lineBin(bars: HistogramBar[], threshold: number): number | null {
  if (bars.length === 0) return null;
  if (threshold < bars[0].x0 || threshold > bars[bars.length - 1].x1) return null;
  const hit = bars.find((b) => threshold >= b.x0 && threshold < b.x1) ?? bars[bars.length - 1];
  return hit.x;
}

/** Score histogram with dashed warn and fail lines (thresholds are z-scores). */
export function ScoreHistogram(props: {
  bars: HistogramBar[];
  warn: number;
  fail: number;
  xLabel: string;
  height: number;
  onBarClick?: (bar: HistogramBar, isLast: boolean) => void;
}) {
  const { bars, warn, fail, xLabel, height, onBarClick } = props;
  const warnX = lineBin(bars, warn);
  const failX = lineBin(bars, fail);
  const data = bars.map((b) => ({ ...b, count: b.counts[""] ?? 0 }));
  const colorFor = (bar: HistogramBar) =>
    bar.x0 >= fail ? theme.fail : bar.x0 >= warn ? theme.warn : theme.bar;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 18, right: 8, bottom: 14, left: 4 }} barGap={0}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="x"
          tick={axisStyle}
          tickFormatter={(v: number) => Number(v).toFixed(1)}
          interval="preserveStartEnd"
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{ value: xLabel, position: "bottom", offset: 0, ...axisLabelStyle }}
        />
        <YAxis
          tick={axisStyle}
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={46}
          label={{ value: "# of episodes", angle: -90, position: "insideLeft", style: { textAnchor: "middle" }, ...axisLabelStyle }}
        />
        <Tooltip content={<HistTooltip signals={[""]} />} cursor={{ fill: "#ffffff10" }} />
        <Bar
          dataKey="count"
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
          onClick={(entry: any) => {
            const bar = (entry?.payload ?? entry) as HistogramBar;
            onBarClick?.(bar, bars.length > 0 && bar.x1 === bars[bars.length - 1].x1);
          }}
          style={{ cursor: onBarClick ? "pointer" : "default" }}
        >
          {data.map((bar) => (
            <Cell key={bar.x0} fill={colorFor(bar)} />
          ))}
        </Bar>
        {warnX !== null && (
          <ReferenceLine x={warnX} stroke={theme.warn} strokeDasharray="5 4" label={{ value: "warn", position: "top", fill: theme.warn, fontSize: 10 }} />
        )}
        {failX !== null && (
          <ReferenceLine x={failX} stroke={theme.fail} strokeDasharray="5 4" label={{ value: "fail", position: "top", fill: theme.fail, fontSize: 10 }} />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Multi-series metric histogram with a dashed warn line per series. */
export function MetricHistogram(props: {
  bars: HistogramBar[];
  /** all series names on the tab, so colors stay stable while filtering */
  allSignals: string[];
  signals: string[];
  xLabel: string;
  height: number;
  warnThresholds?: Record<string, number>;
  onBarClick?: (bar: HistogramBar, signal: string, isLast: boolean) => void;
}) {
  const { bars, allSignals, signals, xLabel, height, warnThresholds, onBarClick } = props;
  const multi = allSignals.length > 1 || (allSignals.length === 1 && allSignals[0] !== "");
  const warnLines = signals
    .map((s) => ({ signal: s, x: warnThresholds?.[s] !== undefined ? lineBin(bars, warnThresholds[s]) : null }))
    .filter((l): l is { signal: string; x: number } => l.x !== null);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={bars} margin={{ top: 18, right: 8, bottom: 14, left: 4 }} barGap={0}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="x"
          tick={axisStyle}
          tickFormatter={fmtTick}
          interval="preserveStartEnd"
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{ value: xLabel, position: "bottom", offset: 0, ...axisLabelStyle }}
        />
        <YAxis
          tick={axisStyle}
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={46}
          label={{ value: "# of episodes", angle: -90, position: "insideLeft", style: { textAnchor: "middle" }, ...axisLabelStyle }}
        />
        <Tooltip content={<HistTooltip signals={signals} />} cursor={{ fill: "#ffffff10" }} />
        {signals.map((s) => (
          <Bar
            key={s}
            name={s}
            dataKey={(bar: HistogramBar) => bar.counts[s] ?? 0}
            fill={multi && s ? signalColor(allSignals, s) : theme.bar}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
            onClick={(entry: any) => {
              const bar = (entry?.payload ?? entry) as HistogramBar;
              onBarClick?.(bar, s, bars.length > 0 && bar.x1 === bars[bars.length - 1].x1);
            }}
            style={{ cursor: onBarClick ? "pointer" : "default" }}
          />
        ))}
        {warnLines.map(({ signal, x }) => (
          <ReferenceLine
            key={signal}
            x={x}
            stroke={multi && signal ? signalColor(allSignals, signal) : theme.warn}
            strokeDasharray="5 4"
            label={multi ? undefined : { value: "warn", position: "top", fill: theme.warn, fontSize: 10 }}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function VerdictBar(props: {
  counts: Record<string, number>;
  onBarClick?: (verdict: string) => void;
  height?: number;
}) {
  const data = Object.entries(props.counts)
    .filter(([verdict, count]) => verdict !== "unknown" || count > 0)
    .map(([verdict, count]) => ({ verdict, count }));

  return (
    <ResponsiveContainer width="100%" height={props.height ?? 220}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 14, left: 4 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="verdict" tick={axisStyle} tickLine={false} axisLine={{ stroke: theme.cardBorder }} label={{ value: "verdict", position: "bottom", offset: 0, ...axisLabelStyle }} />
        <YAxis
          tick={axisStyle}
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={46}
          label={{ value: "# of episodes", angle: -90, position: "insideLeft", style: { textAnchor: "middle" }, ...axisLabelStyle }}
        />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff10" }} formatter={(value: number) => [value, "episodes"]} />
        <Bar
          dataKey="count"
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
          onClick={(entry: any) => props.onBarClick?.(entry.verdict)}
          style={{ cursor: props.onBarClick ? "pointer" : "default" }}
        >
          {data.map((entry) => (
            <Cell key={entry.verdict} fill={verdictColor[entry.verdict] ?? theme.bar} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Episodes per category (task), largest first, with the tail collapsed. */
export function CountBars(props: {
  items: { name: string; count: number }[];
  onBarClick?: (name: string) => void;
  height?: number;
  highlight?: (name: string) => boolean;
}) {
  const data = props.items.slice(0, 12);
  return (
    <ResponsiveContainer width="100%" height={props.height ?? 240}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 30, left: 4 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ ...axisStyle, fontSize: 9 }}
          tickFormatter={(v: string) => (v.length > 12 ? `${v.slice(0, 11)}…` : v)}
          interval={0}
          angle={-30}
          textAnchor="end"
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
        />
        <YAxis tick={axisStyle} allowDecimals={false} tickLine={false} axisLine={false} width={40} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#ffffff10" }} formatter={(v: number) => [v, "episodes"]} />
        <Bar
          dataKey="count"
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
          onClick={(entry: any) => props.onBarClick?.(entry.name)}
          style={{ cursor: props.onBarClick ? "pointer" : "default" }}
        >
          {data.map((d) => (
            <Cell key={d.name} fill={props.highlight?.(d.name) ? theme.warn : theme.bar} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ScatterTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row: Row = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div style={{ fontWeight: 600 }}>{row.episode}</div>
      <div style={{ color: theme.textDim }}>
        iforest {row.values.iforest_score?.toFixed(3)} · novelty {row.values.novelty_knn?.toFixed(3)}
      </div>
      {row.values.is_outlier ? <div style={{ color: theme.fail }}>flagged outlier</div> : null}
    </div>
  );
}

export function OutlierScatter(props: { rows: Row[]; onOpen: (id: string) => void; height?: number }) {
  const points = props.rows.filter(
    (r) => r.values.iforest_score != null && r.values.novelty_knn != null
  ).map((r) => ({ ...r, iforest: r.values.iforest_score as number, knn: r.values.novelty_knn as number }));

  return (
    <ResponsiveContainer width="100%" height={props.height ?? 260}>
      <ScatterChart margin={{ top: 12, right: 16, bottom: 14, left: 0 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" />
        <XAxis
          dataKey="iforest"
          type="number"
          name="iforest_score"
          domain={["auto", "auto"]}
          tick={axisStyle}
          tickFormatter={fmtTick}
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{ value: "iforest score (higher = more anomalous)", position: "bottom", offset: 0, ...axisLabelStyle }}
        />
        <YAxis
          dataKey="knn"
          type="number"
          name="novelty"
          domain={["auto", "auto"]}
          tick={axisStyle}
          tickFormatter={fmtTick}
          tickLine={false}
          axisLine={false}
          label={{ value: "novelty (kNN distance)", angle: -90, position: "insideLeft", ...axisLabelStyle }}
        />
        <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: "3 3" }} />
        <Scatter data={points} isAnimationActive={false} onClick={(p: any) => p?.id && props.onOpen(p.id)} style={{ cursor: "pointer" }}>
          {points.map((r) => (
            <Cell key={r.id} fill={r.values.is_outlier ? theme.fail : theme.bar} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
