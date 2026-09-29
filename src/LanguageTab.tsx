import React from "react";
import { VerdictBar } from "./charts";
import { ExpandToggle } from "./MetricGrid";
import { theme, verdictColor } from "./theme";
import { Card, Chip, DataTable } from "./ui";
import { PanelData, Row } from "./types";

type ShowEpisodes = (ids: string[], description: string) => void;

const EXPLAINERS = {
  verdicts:
    "Static checks on the task string, no model needed. Warn means the string is missing, is a placeholder " +
    "such as 'task desc' or 'Hold', has fewer than 3 words, or (for English text) names no action. A " +
    "vision-language model will not learn what the demonstration shows from such an instruction. These " +
    "flags never enter a score, but the VLA profile raises an episode to at least warn. Click a bar to filter.",
  table:
    "Episodes with a weak task string first. The reason column says which check fired. Click a row to " +
    "inspect the episode.",
};

export default function LanguageTab(props: {
  data: PanelData;
  rows: Row[];
  onSelect: (id: string) => void;
  onShow: ShowEpisodes;
  selectedId: string | null;
}) {
  const { rows, onSelect, onShow, selectedId } = props;
  const [expanded, setExpanded] = React.useState(false);
  const counts: Record<string, number> = { pass: 0, warn: 0, unknown: 0 };
  for (const r of rows) counts[r.language] = (counts[r.language] ?? 0) + 1;
  const sorted = [...rows].sort((a, b) => Number(b.language === "warn") - Number(a.language === "warn"));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Card>
        <div style={{ fontSize: 12, color: theme.textDim, lineHeight: 1.5 }}>
          Static task-string checks only. The instruction-versus-video check with a vision-language model,
          proposed task rewrites and Accept / Reject arrive with phase 3, which is on hold.
        </div>
      </Card>
      <div style={{ display: "grid", gridTemplateColumns: expanded ? "1fr" : "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
        <Card
          title="Language verdicts"
          subtitle="warn: weak or missing task string · click a bar to filter"
          info={EXPLAINERS.verdicts}
          action={<ExpandToggle expanded={expanded} onClick={() => setExpanded(!expanded)} />}
        >
          <VerdictBar
            counts={counts}
            height={expanded ? 440 : 220}
            onBarClick={(verdict) => onShow(rows.filter((r) => r.language === verdict).map((r) => r.id), `language '${verdict}'`)}
          />
        </Card>
      </div>
      <Card title="Task strings" subtitle="Weak strings first · click a row to inspect the episode" info={EXPLAINERS.table}>
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "task", label: "Task" },
            { key: "verdict", label: "Verdict" },
            { key: "reason", label: "Reason" },
          ]}
          rowKeys={sorted.map((r) => r.id)}
          rows={sorted.map((r) => ({
            episode: <span style={{ fontWeight: r.id === selectedId ? 700 : 400 }}>{r.episode}</span>,
            task: r.task || "–",
            verdict: <Chip label={r.language} color={verdictColor[r.language] ?? theme.unknown} />,
            reason: r.notes.task_missing || r.notes.task_generic || "–",
          }))}
          onRowClick={onSelect}
        />
      </Card>
    </div>
  );
}
