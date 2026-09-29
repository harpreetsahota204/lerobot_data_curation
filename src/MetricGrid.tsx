import React from "react";
import { histogram, inBin, MetricHistogram } from "./charts";
import { fmt, signalColor, theme } from "./theme";
import { Card, DataTable } from "./ui";
import { HistogramBar, label, PanelData, Row } from "./types";

type ShowEpisodes = (ids: string[], description: string) => void;

export function ExpandToggle(props: { expanded: boolean; onClick: () => void }) {
  return (
    <button
      onClick={props.onClick}
      title={props.expanded ? "Back to grid" : "Expand this chart"}
      style={{
        background: "none",
        border: `1px solid ${theme.cardBorder}`,
        borderRadius: 4,
        color: theme.textDim,
        width: 22,
        height: 22,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        padding: 0,
      }}
      onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = theme.text)}
      onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = theme.textDim)}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
        {props.expanded ? (
          <>
            <polyline points="4 14 10 14 10 20" />
            <polyline points="20 10 14 10 14 4" />
          </>
        ) : (
          <>
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
          </>
        )}
      </svg>
    </button>
  );
}

/** Shared legend: click a signal to isolate it across every chart and the table. */
export function SignalChips(props: {
  signals: string[];
  active: string | null;
  onPick: (signal: string | null) => void;
}) {
  const { signals, active, onPick } = props;
  if (signals.length < 2) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
      {signals.map((signal) => {
        const isActive = active === signal;
        const dimmed = active !== null && !isActive;
        const color = signalColor(signals, signal);
        return (
          <button
            key={signal}
            onClick={() => onPick(isActive ? null : signal)}
            title={isActive ? "Show all signals" : `Show only ${signal}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              background: isActive ? `${color}22` : "transparent",
              border: `1px solid ${isActive ? color : theme.cardBorder}`,
              borderRadius: 12,
              padding: "2px 9px",
              fontSize: 11,
              color: dimmed ? theme.textDim : theme.text,
              opacity: dimmed ? 0.55 : 1,
              cursor: "pointer",
            }}
          >
            <span style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
            {signal}
          </button>
        );
      })}
    </div>
  );
}

const FAMILY_TITLES: Record<string, string> = {
  motion: "Motion smoothness",
  time: "Time efficiency",
  tracking: "Tracking and contact",
  gripper: "Gripper",
  consistency: "Consistency",
};

/** Values of one metric across rows, per series (signal), for one row set. */
function seriesFor(rows: Row[], metric: string, perSignal: boolean, only: string | null): Record<string, number[]> {
  if (perSignal) {
    const out: Record<string, number[]> = {};
    for (const r of rows) {
      for (const [signal, v] of Object.entries(r.by_signal?.[metric] ?? {})) {
        if (v == null || (only && signal !== only)) continue;
        (out[signal] ??= []).push(v);
      }
    }
    return out;
  }
  const values = rows.map((r) => r.values[metric]).filter((v): v is number => v != null);
  return values.length ? { "": values } : {};
}

function cellValue(row: Row, metric: string, perSignal: boolean, only: string | null): number | null {
  if (perSignal) {
    const by = row.by_signal?.[metric];
    if (!by) return null;
    if (only) return by[only] ?? null;
  }
  return row.values[metric] ?? null;
}

/**
 * A family tab body: signal chips, a grid of histograms (one per metric) and a
 * worst-first table. Used by the Motion & Action tab and any other tab that is
 * "charts plus a ranking".
 */
export default function MetricGrid(props: {
  data: PanelData;
  rows: Row[];
  families: string[];
  onSelect: (id: string) => void;
  onShow: ShowEpisodes;
  selectedId: string | null;
  explainer: string;
  rankingInfo: string;
}) {
  const { data, rows, families, onSelect, onShow, selectedId } = props;
  const [only, setOnly] = React.useState<string | null>(null);
  const [expanded, setExpanded] = React.useState<string | null>(null);

  const metricNames = React.useMemo(
    () =>
      families.flatMap((f) =>
        Object.entries(data.metrics)
          .filter(([name, m]) => m.family === f && (data.computed.includes(name) || (!m.opt_in && m.scored)))
          .map(([name]) => name)
      ),
    [data, families]
  );
  const scoredNames = metricNames.filter((m) => data.metrics[m].scored);

  const chipSignals = React.useMemo(
    () =>
      Array.from(
        new Set(rows.flatMap((r) => metricNames.flatMap((m) => Object.keys(r.by_signal?.[m] ?? {}))))
      ).sort(),
    [rows, metricNames]
  );

  const tabScore = (r: Row) => {
    const zs = scoredNames.map((m) => r.z[m]).filter((z): z is number => z != null);
    return zs.length ? Math.max(...zs) : null;
  };
  const tabFlags = (r: Row) => scoredNames.filter((m) => (r.z[m] ?? -Infinity) >= data.warn_z).length;
  const sorted = [...rows].sort((a, b) => (tabScore(b) ?? -Infinity) - (tabScore(a) ?? -Infinity));

  const shown = expanded ? [expanded] : metricNames;
  const tableMetrics = metricNames.filter((m) => rows.some((r) => r.values[m] != null));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {chipSignals.length > 1 && (
        <div style={{ fontSize: 11, color: theme.textDim }}>
          {only ? `showing only ${only} · click its chip again to show all` : "click a signal chip to isolate it"}
        </div>
      )}
      <SignalChips signals={chipSignals} active={only} onPick={setOnly} />
      <div style={{ fontSize: 11, color: theme.textDim }}>
        click any bar to filter the samples panel · dashed lines = warn thresholds
      </div>

      {families.map((family) => {
        const inFamily = shown.filter((m) => data.metrics[m].family === family);
        if (inFamily.length === 0) return null;
        return (
          <div key={family} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: theme.textDim, textTransform: "uppercase", letterSpacing: 0.5 }}>
              {FAMILY_TITLES[family] ?? family}
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: expanded ? "1fr" : "repeat(auto-fit, minmax(280px, 1fr))",
                gap: 12,
              }}
            >
              {inFamily.map((metric) => {
                const meta = data.metrics[metric];
                const series = seriesFor(rows, metric, meta.per_signal, only);
                const signals = Object.keys(series);
                const bars = histogram(series, 20);
                const thresholds = data.warn_thresholds[metric];
                const direction =
                  meta.kind === "signed_z"
                    ? "closer to 0 is better"
                    : meta.higher_is_worse
                    ? "lower is better"
                    : "higher is better";
                const single = signals.length === 1 ? thresholds?.[signals[0]] : undefined;
                const subtitle = [
                  direction,
                  meta.scored ? null : "not scored",
                  single !== undefined ? `warn ${meta.higher_is_worse ? "≥" : "≤"} ${single.toPrecision(3)}` : null,
                ]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <Card
                    key={metric}
                    title={label(metric).title}
                    subtitle={signals.length > 0 ? subtitle : undefined}
                    info={`${meta.description} Bars count episodes in the current view${
                      meta.per_signal ? ", one colored series per arm" : ""
                    }; dashed lines mark warn thresholds when every episode shares one normalization group. Click a bar to filter the samples panel.`}
                    action={<ExpandToggle expanded={expanded === metric} onClick={() => setExpanded(expanded === metric ? null : metric)} />}
                  >
                    {signals.length === 0 ? (
                      <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: theme.textDim, fontSize: 12 }}>
                        Not computed in the last run (deselected or no data)
                      </div>
                    ) : (
                      <MetricHistogram
                        bars={bars}
                        allSignals={meta.per_signal ? chipSignals : [""]}
                        signals={signals}
                        xLabel={label(metric).short}
                        height={expanded ? 420 : 190}
                        warnThresholds={thresholds}
                        onBarClick={(bar: HistogramBar, signal: string, isLast: boolean) => {
                          const hits = rows.filter((r) => {
                            const v = meta.per_signal ? r.by_signal?.[metric]?.[signal] : r.values[metric];
                            return v != null && inBin(v, bar, isLast);
                          });
                          onShow(hits.map((r) => r.id), `${label(metric).short}${signal ? ` on ${signal}` : ""} in [${bar.x0.toPrecision(3)}, ${bar.x1.toPrecision(3)}]`);
                        }}
                      />
                    )}
                  </Card>
                );
              })}
            </div>
          </div>
        );
      })}

      <Card
        title="Worst-first ranking"
        subtitle={only ? `values for ${only} · click a row to inspect it` : "Click a row to inspect the episode"}
        info={`${props.rankingInfo} Score is the highest robust z-score among this tab's scored metrics. Flags counts metrics at warn or worse (z ≥ ${data.warn_z}).`}
      >
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "score", label: "Score", align: "right" },
            { key: "flags", label: "Flags", align: "right" },
            ...tableMetrics.map((m) => ({ key: m, label: label(m).short, align: "right" as const })),
          ]}
          rowKeys={sorted.map((r) => r.id)}
          rows={sorted.map((r) => {
            const score = tabScore(r);
            return {
              episode: <span style={{ fontWeight: r.id === selectedId ? 700 : 400 }}>{r.episode}</span>,
              score: (
                <span style={{ color: score === null ? theme.text : score >= data.fail_z ? theme.fail : score >= data.warn_z ? theme.warn : theme.text, fontWeight: 600 }}>
                  {fmt(score)}
                </span>
              ),
              flags: tabFlags(r),
              ...Object.fromEntries(
                tableMetrics.map((m) => [m, fmt(cellValue(r, m, data.metrics[m].per_signal, only))])
              ),
            };
          })}
          onRowClick={onSelect}
        />
      </Card>
    </div>
  );
}
