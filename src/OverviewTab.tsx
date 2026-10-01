import React from "react";
import { CountBars, histogram, inBin, OutlierScatter, ScoreHistogram, VerdictBar } from "./charts";
import { fmt, theme, verdictColor } from "./theme";
import { ExpandToggle } from "./MetricGrid";
import { Card, Chip, DataTable } from "./ui";
import { HistogramBar, PanelData, Row } from "./types";

type ShowEpisodes = (ids: string[], description: string) => void;

const EXPLAINERS = {
  histogram:
    "Each bar counts episodes by their profile score: the highest robust z-score across the " +
    "metric groups (worst-of), so one failing group is never diluted by the others. A z-score " +
    "says how many robust standard deviations worse than typical the episode is, measured " +
    "against its own task group when that group has enough episodes, otherwise against the " +
    "whole view. Dashed lines mark warn and fail. Click a bar to filter the samples panel to " +
    "those episodes.",
  verdicts:
    "Profile verdicts: fail at score >= 3, warn at >= 2, otherwise pass. Under the VLA profile a " +
    "weak task instruction also forces at least warn. Integrity and Language are separate checks " +
    "that never enter the score. Click a bar to filter the samples panel.",
  tasks:
    "Episodes per task, largest first. Amber bars are tasks with too few episodes to normalize on their own. Click a bar to filter to that task.",
  sources: "Episodes, mean score and flagged count per source. Click a row to filter to that source.",
  meanScore:
    "The average of the selected profile's score over this source's episodes in the current view. The score " +
    "is each episode's worst metric-group z-score (higher is worse), so a source with a few very bad episodes " +
    "can show a high mean. Read it with the Warn or fail column. Episodes without a score are left out. " +
    "Scores are measured against each episode's own task, or against the whole view for small tasks, so " +
    "comparing sources also compares what tasks and robots they contain.",
  score:
    "The selected profile's score: for each metric group (motion, time, tracking, gripper, consistency, and camera when it was computed) take " +
    "the worst weighted robust z-score of its metrics, then take the highest group value. It is a worst-of, " +
    "never an average, so one failing group is not diluted by the others. A z-score is how many robust " +
    "standard deviations worse than typical an episode is, measured against its own task when that task has " +
    "enough episodes, otherwise against the whole view (marked with *). Amber is 2 or more (warn), red is 3 or " +
    "more (fail). Integrity, Language and outlier scores never enter it.",
  outliers:
    "Each point is an episode, placed by two outlier detectors fit on its metric z-scores within its group. " +
    "X: isolation-forest score (higher = more anomalous). Y: mean distance to its nearest neighbors. Red points " +
    "crossed the warn threshold. Outlier scores never enter the profile score: unusual episodes can be " +
    "exceptionally clean rather than bad.",
  ranking:
    "Episodes ranked worst-first by the selected profile's score. Driver names the metric group " +
    "behind the score. Flags counts groups at warn or worse. Integrity and Language are separate " +
    "pass/warn/fail checks that never enter the score. Click a row to inspect the episode.",
};

function scoreColor(score: number | null, warn: number, fail: number): string {
  if (score === null) return theme.text;
  if (score >= fail) return theme.fail;
  if (score >= warn) return theme.warn;
  return theme.text;
}

function count(rows: Row[], pick: (r: Row) => string): Record<string, number> {
  const out: Record<string, number> = { pass: 0, warn: 0, fail: 0, unknown: 0 };
  for (const r of rows) out[pick(r)] = (out[pick(r)] ?? 0) + 1;
  return out;
}

export default function OverviewTab(props: {
  data: PanelData;
  rows: Row[];
  profile: string;
  onSelect: (id: string) => void;
  onOpen: (id: string) => void;
  onShow: ShowEpisodes;
  selectedId: string | null;
}) {
  const { data, rows, profile, onSelect, onOpen, onShow, selectedId } = props;
  // One chart can take the whole row, like the histograms on the Motion & Action tab.
  const [expanded, setExpanded] = React.useState<string | null>(null);
  const toggle = (id: string) => (
    <ExpandToggle expanded={expanded === id} onClick={() => setExpanded(expanded === id ? null : id)} />
  );
  const show = (id: string) => expanded === null || expanded === id;
  const profileLabel = data.profiles.find((p) => p.id === profile)?.label ?? profile;

  const scored = rows.filter((r) => r.profiles[profile]?.score != null);
  const bars = histogram(scored.map((r) => r.profiles[profile].score as number), 20);

  const verdicts = count(rows, (r) => r.profiles[profile]?.verdict ?? "unknown");
  const integrity = count(rows, (r) => r.integrity);
  const language = count(rows, (r) => r.language);

  const sorted = [...rows].sort(
    (a, b) => (b.profiles[profile]?.score ?? -Infinity) - (a.profiles[profile]?.score ?? -Infinity)
  );

  const taskCounts = new Map<string, number>();
  for (const r of rows) taskCounts.set(r.task, (taskCounts.get(r.task) ?? 0) + 1);
  const taskItems = [...taskCounts.entries()]
    .map(([name, c]) => ({ name: name || "(no task)", count: c }))
    .sort((a, b) => b.count - a.count);

  const bySource = new Map<string, Row[]>();
  for (const r of rows) (bySource.get(r.source) ?? bySource.set(r.source, []).get(r.source)!).push(r);
  const sourceRows = [...bySource.entries()].map(([source, rs]) => {
    const scores = rs.map((r) => r.profiles[profile]?.score).filter((s): s is number => s != null);
    const flagged = rs.filter((r) => ["warn", "fail"].includes(r.profiles[profile]?.verdict ?? "")).length;
    return { source, n: rs.length, mean: scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null, flagged };
  }).sort((a, b) => (b.mean ?? -Infinity) - (a.mean ?? -Infinity));

  const hasOutliers = rows.some((r) => r.values.iforest_score != null);

  const onBar = (bar: HistogramBar, isLast: boolean) => {
    const hits = scored.filter((r) => inBin(r.profiles[profile].score as number, bar, isLast));
    onShow(hits.map((r) => r.id), `score in [${bar.x0.toFixed(2)}, ${bar.x1.toFixed(2)}]`);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ fontSize: 11, color: theme.textDim }}>
        click any bar to filter the samples panel · dashed lines = warn and fail thresholds
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: expanded ? "1fr" : "repeat(auto-fit, minmax(300px, 1fr))",
          gap: 12,
        }}
      >
        {show("score") && (
        <Card
          title="Profile score"
          action={toggle("score")}
          subtitle={`${profileLabel} · higher is worse · warn ≥ ${data.warn_z} · fail ≥ ${data.fail_z}`}
          info={EXPLAINERS.histogram}
        >
          {bars.length === 0 ? (
            <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: theme.textDim, fontSize: 12 }}>
              No scored episodes in this view
            </div>
          ) : (
            <ScoreHistogram bars={bars} warn={data.warn_z} fail={data.fail_z} xLabel="profile score (worst-of z)" height={expanded === "score" ? 440 : 220} onBarClick={onBar} />
          )}
        </Card>
        )}

        {show("verdicts") && (
        <Card title="Verdicts" subtitle="profile · integrity · language · click a bar to filter" info={EXPLAINERS.verdicts} action={toggle("verdicts")}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 4 }}>
            {[
              { title: "profile", counts: verdicts, pick: (r: Row) => r.profiles[profile]?.verdict ?? "unknown" },
              { title: "integrity", counts: integrity, pick: (r: Row) => r.integrity },
              { title: "language", counts: language, pick: (r: Row) => r.language },
            ].map((v) => (
              <div key={v.title}>
                <div style={{ fontSize: 11, color: theme.textDim, textAlign: "center" }}>{v.title}</div>
                <VerdictBar
                  counts={v.counts}
                  height={expanded === "verdicts" ? 400 : 190}
                  onBarClick={(verdict) => onShow(rows.filter((r) => v.pick(r) === verdict).map((r) => r.id), `${v.title} '${verdict}'`)}
                />
              </div>
            ))}
          </div>
        </Card>
        )}

        {show("tasks") && (
        <Card title="Episodes per task" subtitle="coverage · click a bar to filter" info={EXPLAINERS.tasks} action={toggle("tasks")}>
          <CountBars
            items={taskItems}
            height={expanded === "tasks" ? 440 : 240}
            highlight={(name) => (taskCounts.get(name === "(no task)" ? "" : name) ?? 0) < data.under_covered_below}
            onBarClick={(name) => onShow(rows.filter((r) => (r.task || "(no task)") === name).map((r) => r.id), `task '${name.slice(0, 40)}'`)}
          />
        </Card>
        )}

        {hasOutliers && show("outliers") && (
          <Card title="Outliers" subtitle="information only · click a point to inspect it" info={EXPLAINERS.outliers} action={toggle("outliers")}>
            <OutlierScatter rows={rows} onOpen={onSelect} height={expanded === "outliers" ? 440 : 260} />
          </Card>
        )}
      </div>

      <Card title="Per-source summary" subtitle={`${sourceRows.length} source(s) · click a row to filter`} info={EXPLAINERS.sources}>
        <DataTable
          columns={[
            { key: "source", label: "Source" },
            { key: "n", label: "Episodes", align: "right" },
            { key: "mean", label: "Mean score", align: "right", info: EXPLAINERS.meanScore },
            { key: "flagged", label: "Warn or fail", align: "right" },
          ]}
          rowKeys={sourceRows.map((s) => s.source)}
          rows={sourceRows.map((s) => ({ source: s.source, n: s.n, mean: fmt(s.mean), flagged: s.flagged }))}
          onRowClick={(source) => onShow(rows.filter((r) => r.source === source).map((r) => r.id), `source '${source}'`)}
        />
      </Card>

      <Card title="Worst-first ranking" subtitle="Click a row to inspect the episode" info={EXPLAINERS.ranking}>
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "task", label: "Task" },
            { key: "score", label: "Score", align: "right", info: EXPLAINERS.score },
            { key: "flags", label: "Flags", align: "right" },
            { key: "driver", label: "Driver" },
            { key: "integrity", label: "Integrity" },
            { key: "language", label: "Language" },
          ]}
          rowKeys={sorted.map((r) => r.id)}
          rows={sorted.map((r) => {
            const p = r.profiles[profile];
            return {
              episode: <span style={{ fontWeight: r.id === selectedId ? 700 : 400 }}>{r.episode}</span>,
              task: r.task || "–",
              score: (
                <span
                  style={{ color: scoreColor(p?.score ?? null, data.warn_z, data.fail_z), fontWeight: 600 }}
                  title={r.group_basis === "pooled" ? `normalized against the whole view (${r.group_n} episodes)` : `normalized within its task (${r.group_n} episodes)`}
                >
                  {fmt(p?.score ?? null)}
                  {r.group_basis === "pooled" ? " *" : ""}
                </span>
              ),
              flags: p?.n_flags ?? 0,
              driver: p?.driver ?? "–",
              integrity: <Chip label={r.integrity} color={verdictColor[r.integrity] ?? theme.unknown} />,
              language: <Chip label={r.language} color={verdictColor[r.language] ?? theme.unknown} />,
            };
          })}
          onRowClick={onSelect}
        />
        {data.pooled_count > 0 && (
          <div style={{ fontSize: 11, color: theme.textDim, marginTop: 6 }}>
            * normalized against the whole view instead of the episode's own task (task group under {data.min_group} episodes)
          </div>
        )}
      </Card>
    </div>
  );
}
