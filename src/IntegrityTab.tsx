import React from "react";
import { CountBars, VerdictBar } from "./charts";
import { fmt, theme, verdictColor } from "./theme";
import { Card, Chip, DataTable } from "./ui";
import { label, PanelData, Row } from "./types";

type ShowEpisodes = (ids: string[], description: string) => void;

const EXPLAINERS = {
  verdicts:
    "Integrity flags defects that break training regardless of quality: missing or duplicated frames, " +
    "a length that disagrees with the data, a video window that disagrees with the length, NaN or " +
    "infinite values, near-empty episodes. Fail means broken. Warn means a schema differs from the rest " +
    "of the view. These flags never enter a score. Click a bar to filter the samples panel.",
  coverage:
    "Episodes per task, largest first. Amber bars are tasks with too few episodes to learn from or to " +
    "normalize on their own. Reported, never scored. Click a bar to filter to that task.",
  table:
    "Per-episode integrity. The failing-checks column names each check that fired and its value, so you " +
    "know what to inspect. Click a row to inspect the episode.",
};

const ORDER: Record<string, number> = { fail: 0, warn: 1, unknown: 2, pass: 3 };

function failing(row: Row, data: PanelData): string {
  const parts: string[] = [];
  for (const [name, meta] of Object.entries(data.metrics)) {
    if (!meta.check) continue;
    const v = row.values[name];
    if (v != null && v > meta.check[1]) parts.push(`${label(name).short} (${fmt(v, v < 10 ? 2 : 0)})`);
  }
  return parts.join(", ");
}

export default function IntegrityTab(props: {
  data: PanelData;
  rows: Row[];
  onSelect: (id: string) => void;
  onShow: ShowEpisodes;
  selectedId: string | null;
}) {
  const { data, rows, onSelect, onShow, selectedId } = props;

  const counts: Record<string, number> = { pass: 0, warn: 0, fail: 0, unknown: 0 };
  for (const r of rows) counts[r.integrity] = (counts[r.integrity] ?? 0) + 1;

  const taskCounts = new Map<string, number>();
  for (const r of rows) taskCounts.set(r.task, (taskCounts.get(r.task) ?? 0) + 1);
  const items = [...taskCounts.entries()].map(([name, count]) => ({ name: name || "(no task)", count })).sort((a, b) => b.count - a.count);

  const sourceCounts = new Map<string, number>();
  for (const r of rows) sourceCounts.set(r.source, (sourceCounts.get(r.source) ?? 0) + 1);
  const sources = [...sourceCounts.entries()].sort((a, b) => b[1] - a[1]);

  const sorted = [...rows].sort((a, b) => (ORDER[a.integrity] ?? 9) - (ORDER[b.integrity] ?? 9));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
        <Card title="Integrity verdicts" subtitle="fail: a check failed · warn: schema differs · click a bar to filter" info={EXPLAINERS.verdicts}>
          <VerdictBar
            counts={counts}
            onBarClick={(verdict) => onShow(rows.filter((r) => r.integrity === verdict).map((r) => r.id), `integrity '${verdict}'`)}
          />
        </Card>
        <Card
          title="Coverage: episodes per task"
          subtitle={`${data.balance.under_covered?.length ?? 0} under-covered task(s) · click a bar to filter`}
          info={EXPLAINERS.coverage}
        >
          <CountBars
            items={items}
            highlight={(name) => (taskCounts.get(name === "(no task)" ? "" : name) ?? 0) < data.under_covered_below}
            onBarClick={(name) =>
              onShow(rows.filter((r) => (r.task || "(no task)") === name).map((r) => r.id), `task '${name.slice(0, 40)}'`)
            }
          />
        </Card>
      </div>

      <Card title="Episodes per source" subtitle={`${sources.length} source(s)${data.balance.dominant_sources?.length ? " · one source holds more than half" : ""}`}>
        <DataTable
          columns={[
            { key: "source", label: "Source" },
            { key: "n", label: "Episodes", align: "right" },
            { key: "share", label: "Share", align: "right" },
          ]}
          rowKeys={sources.map(([s]) => s)}
          rows={sources.map(([s, n]) => ({ source: s, n, share: `${Math.round((100 * n) / Math.max(1, rows.length))}%` }))}
          onRowClick={(s) => onShow(rows.filter((r) => r.source === s).map((r) => r.id), `source '${s}'`)}
        />
      </Card>

      <Card title="Per-episode integrity" subtitle="Failures first · click a row to inspect the episode" info={EXPLAINERS.table}>
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "verdict", label: "Verdict" },
            { key: "checks", label: "Failing checks" },
            { key: "note", label: "Note" },
          ]}
          rowKeys={sorted.map((r) => r.id)}
          rows={sorted.map((r) => ({
            episode: <span style={{ fontWeight: r.id === selectedId ? 700 : 400 }}>{r.episode}</span>,
            verdict: <Chip label={r.integrity} color={verdictColor[r.integrity] ?? theme.unknown} />,
            checks: failing(r, data) || "–",
            note: r.notes.schema_mismatch || r.notes.video_window_mismatch || "–",
          }))}
          onRowClick={onSelect}
        />
      </Card>
    </div>
  );
}
