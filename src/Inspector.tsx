import React, { useEffect, useMemo, useState } from "react";
import { useOperatorExecutor } from "@fiftyone/operators";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt, signalColor, theme, verdictColor } from "./theme";
import { Button, Card, Chip, DataTable, IconButton } from "./ui";
import { EpisodeDetail, label, PanelData, PLUGIN, Row, Span } from "./types";

const SPAN_COLORS: Record<string, string> = {
  idle: "#9e9e9e",
  pause: theme.warn,
  rough: "#ff8a3d",
  spike: theme.fail,
  recovery: theme.pass,
};

const axis = { fontSize: 10, fill: theme.textDim };
const MAX_DEFAULT_JOINTS = 6;

const EXPLAINERS = {
  metrics:
    "Every metric computed for this episode. Value is the worst-of across arms (or cameras); z says how many " +
    "robust standard deviations worse than typical it is (higher is worse). The breakdown column shows each arm's or camera's " +
    "own value. Not-scored metrics are shown for context.",
  traces:
    "Per-joint action (solid) and state (dashed), each rescaled by the joint's own range so joints with different " +
    "units share one axis. Shaded spans are the events behind the scores: idle stretches, the longest pause, the " +
    "roughest smoothness window, acceleration spikes and regrasp recoveries. Pick joints with the chips.",
  speed:
    "Arm speed over time (range-fractions per second, low-pass filtered). The dashed line is the idle threshold: " +
    "5% of this episode's own moving speed. Time under it at the start, end or middle is what the idle and pause " +
    "metrics measure.",
  gripper:
    "Gripper command rescaled to 0-1. Vertical lines are close (red) and open (green) transitions, found with " +
    "hysteresis. Needs a named gripper dimension and a known open direction.",
  thumbs: "Frames from the first two cameras at the middle of the flagged spans, then near the start.",
};

function Shade({ spans }: { spans: Span[] }) {
  return (
    <>
      {spans.map((s, i) => (
        <ReferenceArea
          key={`${s.kind}-${i}`}
          x1={s.start_s}
          x2={s.end_s}
          fill={SPAN_COLORS[s.kind]}
          fillOpacity={0.18}
          stroke={SPAN_COLORS[s.kind]}
          strokeOpacity={0.4}
        />
      ))}
    </>
  );
}

function ChartFrame(props: { children: React.ReactElement; height: number }) {
  return (
    <ResponsiveContainer width="100%" height={props.height}>
      {props.children}
    </ResponsiveContainer>
  );
}

export default function Inspector(props: {
  data: PanelData;
  row: Row;
  profile: string;
  onOpenViewer: (id: string) => void;
  onClose: () => void;
}) {
  const { data, row, profile, onOpenViewer, onClose } = props;
  const op = useOperatorExecutor(`${PLUGIN}/lr_get_episode_detail`);
  const computeOp = useOperatorExecutor(`${PLUGIN}/lr_prompt_compute`);
  const [joints, setJoints] = useState<number[]>([]);

  useEffect(() => {
    op.execute({ sample_id: row.id });
    setJoints([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [row.id]);

  const detail =
    op.result && (op.result as unknown as EpisodeDetail).sample_id === row.id
      ? (op.result as unknown as EpisodeDetail)
      : null;
  const error = op.result && (op.result as unknown as EpisodeDetail).error;

  const shownJoints = useMemo(() => {
    if (!detail) return [];
    if (joints.length) return joints;
    const dim = detail.action?.[0]?.length ?? detail.state?.[0]?.length ?? 0;
    return Array.from({ length: Math.min(dim, MAX_DEFAULT_JOINTS) }, (_, i) => i);
  }, [detail, joints]);

  const p = row.profiles[profile];

  const metricRows = Object.entries(data.metrics)
    .filter(([name]) => row.values[name] != null || row.raw[name] != null)
    .map(([name, meta]) => {
      const by = row.by_signal?.[name] ?? {};
      const breakdown = Object.entries(by)
        .map(([s, v]) => `${s} ${fmt(v)}`)
        .join(" · ");
      return {
        name,
        family: meta.family,
        value: row.values[name],
        z: row.z[name],
        breakdown,
        note: row.notes[name],
        scored: meta.scored,
      };
    });

  const traceData = useMemo(() => {
    if (!detail) return [];
    return detail.t.map((t, i) => {
      const point: Record<string, number> = { t };
      shownJoints.forEach((j) => {
        if (detail.action) point[`a${j}`] = detail.action[i]?.[j];
        if (detail.state) point[`s${j}`] = detail.state[i]?.[j];
      });
      return point;
    });
  }, [detail, shownJoints]);

  const dim = detail?.action?.[0]?.length ?? detail?.state?.[0]?.length ?? 0;
  const jointIds = Array.from({ length: dim }, (_, i) => i);

  return (
    <Card
      title={`Inspector · ${row.episode}`}
      subtitle={row.task || "(no task)"}
      action={
        <span style={{ display: "inline-flex", gap: 2 }}>
          <IconButton icon="openInNew" title="Open in viewer" onClick={() => onOpenViewer(row.id)} />
          <IconButton icon="close" title="Close inspector" onClick={onClose} />
        </span>
      }
    >
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 10, fontSize: 12 }}>
        <span>
          score{" "}
          <b style={{ color: p?.verdict === "fail" ? theme.fail : p?.verdict === "warn" ? theme.warn : theme.text }}>
            {fmt(p?.score ?? null)}
          </b>
        </span>
        <span>driver <b>{p?.driver ?? "–"}</b></span>
        <Chip label={`profile ${p?.verdict ?? "unknown"}`} color={verdictColor[p?.verdict ?? "unknown"]} />
        <Chip label={`integrity ${row.integrity}`} color={verdictColor[row.integrity] ?? theme.unknown} />
        <Chip label={`language ${row.language}`} color={verdictColor[row.language] ?? theme.unknown} />
        <span style={{ color: theme.textDim }}>
          {row.robot} ·{" "}
          {row.group_basis === "pooled"
            ? `normalized against the whole view (${row.group_n} episodes)`
            : `normalized within its task (${row.group_n} episodes)`}
        </span>
      </div>

      {error ? (
        <div style={{ color: theme.fail, fontSize: 12, padding: 8 }}>Could not read this episode: {String(error)}</div>
      ) : !detail ? (
        <div style={{ color: theme.textDim, fontSize: 12, padding: 16 }}>Loading episode…</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {detail.missing?.length > 0 && (
            <div
              style={{
                display: "flex",
                gap: 12,
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 12px",
                borderRadius: 6,
                border: `1px solid ${theme.cardBorder}`,
                fontSize: 12,
                color: theme.textDim,
              }}
            >
              <span>
                Some charts are missing because the last Compute quality run did not have these picks.{" "}
                {detail.missing.join(" ")} Pick them on the form's Data tab and run it again.
              </span>
              <Button label="Open Compute quality" onClick={() => computeOp.execute({})} />
            </div>
          )}
          {detail.spans.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {detail.spans.map((s, i) => (
                <Chip key={i} label={`${s.label} ${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)}s`} color={SPAN_COLORS[s.kind]} />
              ))}
            </div>
          )}

          <Card title="Metrics" info={EXPLAINERS.metrics}>
            <DataTable
              columns={[
                { key: "metric", label: "Metric" },
                { key: "value", label: "Value", align: "right" },
                { key: "z", label: "z", align: "right" },
                { key: "breakdown", label: "Per arm or camera" },
                { key: "note", label: "Note" },
              ]}
              rowKeys={metricRows.map((m) => m.name)}
              rows={metricRows.map((m) => ({
                metric: (
                  <span title={data.metrics[m.name].description} style={{ color: m.scored ? theme.text : theme.textDim }}>
                    {label(m.name).title}
                    {m.scored ? "" : " (not scored)"}
                  </span>
                ),
                value: fmt(m.value),
                z: (
                  <span style={{ color: m.z != null && m.z >= data.fail_z ? theme.fail : m.z != null && m.z >= data.warn_z ? theme.warn : theme.text }}>
                    {fmt(m.z)}
                  </span>
                ),
                breakdown: m.breakdown || "–",
                note: m.note || "–",
              }))}
              onRowClick={() => undefined}
            />
          </Card>

          {(detail.action || detail.state) && (
            <Card title="Joint traces" subtitle="action solid · state dashed · shaded = flagged spans" info={EXPLAINERS.traces}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 6 }}>
                {jointIds.map((j) => {
                  const on = shownJoints.includes(j);
                  const color = signalColor(jointIds.map(String), String(j));
                  return (
                    <button
                      key={j}
                      onClick={() =>
                        setJoints(on ? shownJoints.filter((x) => x !== j) : [...shownJoints, j].sort((a, b) => a - b))
                      }
                      style={{
                        background: on ? `${color}22` : "transparent",
                        border: `1px solid ${on ? color : theme.cardBorder}`,
                        borderRadius: 10,
                        color: on ? theme.text : theme.textDim,
                        fontSize: 10,
                        padding: "1px 7px",
                        cursor: "pointer",
                      }}
                    >
                      {detail.joint_names[j] ?? j}
                    </button>
                  );
                })}
              </div>
              <ChartFrame height={240}>
                <LineChart data={traceData} margin={{ top: 8, right: 8, bottom: 14, left: 4 }}>
                  <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tick={axis} tickFormatter={(v: number) => `${v.toFixed(0)}s`} tickLine={false} axisLine={{ stroke: theme.cardBorder }} />
                  <YAxis tick={axis} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ background: theme.headerBg, border: `1px solid ${theme.cardBorder}`, fontSize: 11 }} labelFormatter={(v: number) => `${Number(v).toFixed(2)} s`} formatter={(v: number) => Number(v).toFixed(3)} />
                  <Shade spans={detail.spans} />
                  {shownJoints.map((j) => {
                    const color = signalColor(jointIds.map(String), String(j));
                    return (
                      <React.Fragment key={j}>
                        {detail.action && <Line dataKey={`a${j}`} name={`${detail.joint_names[j] ?? j} action`} stroke={color} dot={false} strokeWidth={1.4} isAnimationActive={false} />}
                        {detail.state && <Line dataKey={`s${j}`} name={`${detail.joint_names[j] ?? j} state`} stroke={color} dot={false} strokeWidth={1.2} strokeDasharray="4 3" isAnimationActive={false} />}
                      </React.Fragment>
                    );
                  })}
                </LineChart>
              </ChartFrame>
            </Card>
          )}

          {detail.speed && (
            <Card title="Speed profile" subtitle="dashed line = idle threshold" info={EXPLAINERS.speed}>
              <ChartFrame height={170}>
                <LineChart data={detail.speed.t.map((t, i) => ({ t, v: detail.speed!.v[i] }))} margin={{ top: 8, right: 8, bottom: 14, left: 4 }}>
                  <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tick={axis} tickFormatter={(v: number) => `${v.toFixed(0)}s`} tickLine={false} axisLine={{ stroke: theme.cardBorder }} />
                  <YAxis tick={axis} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ background: theme.headerBg, border: `1px solid ${theme.cardBorder}`, fontSize: 11 }} labelFormatter={(v: number) => `${Number(v).toFixed(2)} s`} formatter={(v: number) => Number(v).toFixed(3)} />
                  <Shade spans={detail.spans.filter((s) => s.kind === "idle" || s.kind === "pause")} />
                  {detail.idle_threshold != null && <ReferenceLine y={detail.idle_threshold} stroke={theme.warn} strokeDasharray="5 4" />}
                  <Line dataKey="v" name="speed" stroke={theme.bar} dot={false} strokeWidth={1.4} isAnimationActive={false} />
                </LineChart>
              </ChartFrame>
            </Card>
          )}

          {Object.keys(detail.grippers).length > 0 && (
            <Card title="Gripper timeline" subtitle="red = close · green = open" info={EXPLAINERS.gripper}>
              {Object.entries(detail.grippers).map(([side, g]) => (
                <div key={side}>
                  <div style={{ fontSize: 11, color: theme.textDim }}>{side}</div>
                  <ChartFrame height={110}>
                    <LineChart data={g.t.map((t, i) => ({ t, v: g.v[i] }))} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
                      <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tick={axis} tickFormatter={(v: number) => `${v.toFixed(0)}s`} tickLine={false} axisLine={{ stroke: theme.cardBorder }} />
                      <YAxis tick={axis} tickLine={false} axisLine={false} width={30} domain={[0, 1]} />
                      <Shade spans={detail.spans.filter((s) => s.kind === "recovery")} />
                      {g.events.map((e, i) => (
                        <ReferenceLine key={i} x={e.t} stroke={e.kind === "close" ? theme.fail : theme.pass} strokeDasharray="3 3" />
                      ))}
                      <Line dataKey="v" stroke={theme.bar} dot={false} strokeWidth={1.4} isAnimationActive={false} />
                    </LineChart>
                  </ChartFrame>
                </div>
              ))}
            </Card>
          )}

          {detail.thumbnails.length > 0 && (
            <Card title="Camera frames" info={EXPLAINERS.thumbs}>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {detail.thumbnails.map((th, i) => (
                  <figure key={i} style={{ margin: 0 }}>
                    <img
                      alt={`${th.camera} at ${th.t}s`}
                      src={`data:image/jpeg;base64,${th.jpeg}`}
                      style={{ display: "block", width: 192, borderRadius: 4, border: `1px solid ${theme.cardBorder}` }}
                    />
                    <figcaption style={{ fontSize: 10, color: theme.textDim, marginTop: 2 }}>
                      {th.camera} · {th.t.toFixed(1)} s
                    </figcaption>
                  </figure>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </Card>
  );
}
