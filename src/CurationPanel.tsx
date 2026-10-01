import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { useOperatorExecutor } from "@fiftyone/operators";
import IntegrityTab from "./IntegrityTab";
import Inspector from "./Inspector";
import LanguageTab from "./LanguageTab";
import MetricGrid from "./MetricGrid";
import OverviewTab from "./OverviewTab";
import { theme } from "./theme";
import { Banner, Button, Card, Tabs } from "./ui";
import { PanelData, PLUGIN } from "./types";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "motion", label: "Motion & Action" },
  { id: "integrity", label: "Integrity & Coverage" },
  { id: "vision", label: "Vision" },
  { id: "language", label: "Language" },
];

const selectStyle: React.CSSProperties = {
  background: theme.card,
  color: theme.text,
  border: `1px solid ${theme.cardBorder}`,
  borderRadius: 6,
  padding: "5px 8px",
  fontSize: 12,
  maxWidth: 260,
};

export default function CurationPanel() {
  const dataOp = useOperatorExecutor(`${PLUGIN}/lr_get_panel_data`);
  const openOp = useOperatorExecutor(`${PLUGIN}/lr_open_episode`);
  const tagOp = useOperatorExecutor(`${PLUGIN}/lr_tag_episodes`);
  const showOp = useOperatorExecutor(`${PLUGIN}/lr_show_episodes`);
  const promptOp = useOperatorExecutor(`${PLUGIN}/lr_prompt_compute`);
  const visionOp = useOperatorExecutor(`${PLUGIN}/lr_prompt_vision`);

  const view = useRecoilValue(fos.view);
  const selected = useRecoilValue(fos.selectedSamples);

  const [data, setData] = useState<PanelData | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [profile, setProfile] = useState<string | null>(null);
  const [task, setTask] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const profilePicked = useRef(false);
  const inspectorRef = useRef<HTMLDivElement>(null);

  // One backend call per refresh; re-fetch whenever the view changes.
  const viewKey = useMemo(() => JSON.stringify(view ?? []), [view]);
  useEffect(() => {
    dataOp.execute({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey]);

  useEffect(() => {
    if (!dataOp.result) return;
    const next = dataOp.result as unknown as PanelData;
    setData(next);
    if (next.scored && !profilePicked.current) setProfile(next.default_profile);
  }, [dataOp.result]);

  const activeProfile = profile ?? data?.default_profile ?? "policy";
  const hasCamera = useMemo(
    () => (data?.computed ?? []).some((m) => data?.metrics[m]?.family === "camera"),
    [data]
  );

  const rows = useMemo(
    () => (data?.rows ?? []).filter((r) => !task || r.task === task),
    [data, task]
  );

  const selectedRow = useMemo(
    () => (selectedId ? (data?.rows ?? []).find((r) => r.id === selectedId) ?? null : null),
    [data, selectedId]
  );

  useEffect(() => {
    if (selectedId && inspectorRef.current) {
      inspectorRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedId]);

  const openEpisode = useCallback(
    (sampleId: string) => openOp.execute({ sample_id: sampleId, profile: activeProfile }),
    [openOp, activeProfile]
  );
  const showEpisodes = useCallback(
    (ids: string[], description: string) => showOp.execute({ sample_ids: ids, description }),
    [showOp]
  );

  const tagScope = selected.size > 0 ? `${selected.size} selected` : `all ${rows.length} in view`;
  const tag = useCallback(
    (tagName: string) => tagOp.execute({ tag: tagName, sample_ids: Array.from(selected) }),
    [tagOp, selected]
  );

  if (!data) {
    return (
      <Center>
        <div style={{ color: theme.textDim, fontSize: 13 }}>Loading episode scores…</div>
      </Center>
    );
  }

  if (!data.scored) {
    return (
      <Center>
        <div style={{ textAlign: "center", maxWidth: 420 }}>
          <div style={{ fontSize: 16, fontWeight: 600, color: theme.text, marginBottom: 8 }}>
            No episode scores yet
          </div>
          <div style={{ fontSize: 13, color: theme.textDim, marginBottom: 16 }}>
            Run <b>LeRobot curation: compute quality</b> to score the current view for motion
            smoothness, time efficiency, integrity and language.
          </div>
          <Button label="Compute quality" primary onClick={() => promptOp.execute({})} />
        </div>
      </Center>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        color: theme.text,
        fontFamily: "inherit",
      }}
    >
      <div style={{ padding: "8px 16px 0" }}>
        {data.config_version_mismatch && (
          <div style={{ marginBottom: 8 }}>
            <Banner>
              This view mixes scores computed under different formula versions (config_version).
              Rankings below blend incomparable runs. Re-run <b>compute quality</b> on the whole
              view to make them comparable again.
            </Banner>
          </div>
        )}
        {data.mixed_runs && (
          <div style={{ marginBottom: 8 }}>
            <Banner>
              This view mixes episodes scored in different runs, so their scores were measured against different
              populations and warn thresholds are hidden. Re-run <b>compute quality</b> on the whole view to make
              them comparable.
            </Banner>
          </div>
        )}
        {data.pooled_count > 0 && (
          <div style={{ marginBottom: 8 }}>
            <Banner>
              {data.pooled_count} of {data.rows.length} episode(s) belong to a task with fewer than{" "}
              {data.min_group} episodes, so they are normalized against the whole view. Scores then
              compare episodes across tasks and robots: treat this ranking as low-confidence.
            </Banner>
          </div>
        )}
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
          <label style={{ fontSize: 11, color: theme.textDim }}>
            Profile{" "}
            <select
              style={selectStyle}
              value={activeProfile}
              onChange={(e) => {
                profilePicked.current = true;
                setProfile(e.target.value);
              }}
            >
              {data.profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <label style={{ fontSize: 11, color: theme.textDim }}>
            Task{" "}
            <select style={selectStyle} value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="">All tasks ({data.rows.length})</option>
              {data.tasks.map((t) => (
                <option key={t.task} value={t.task}>
                  {(t.task || "(no task)").slice(0, 60)} ({t.n})
                </option>
              ))}
            </select>
          </label>
        </div>
        <Tabs tabs={TABS} active={activeTab} onChange={setActiveTab} />
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {activeTab === "overview" && (
          <OverviewTab
            data={data}
            rows={rows}
            profile={activeProfile}
            onSelect={setSelectedId}
            onOpen={openEpisode}
            onShow={showEpisodes}
            selectedId={selectedId}
          />
        )}
        {activeTab === "motion" && (
          <MetricGrid
            data={data}
            rows={rows}
            families={["motion", "time", "tracking", "gripper", "consistency"]}
            onSelect={setSelectedId}
            onShow={showEpisodes}
            selectedId={selectedId}
            explainer=""
            rankingInfo="Episodes ranked worst-first on this tab's metrics. Motion, time, tracking, gripper and consistency metrics each vote as a group."
          />
        )}
        {activeTab === "integrity" && (
          <IntegrityTab data={data} rows={rows} onSelect={setSelectedId} onShow={showEpisodes} selectedId={selectedId} />
        )}
        {activeTab === "vision" &&
          (hasCamera ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                <span style={{ fontSize: 11, color: theme.textDim, flex: 1 }}>
                  Camera metrics from the last run. Each camera is compared with the same camera in other episodes.
                </span>
                <Button label="Recompute camera metrics" onClick={() => visionOp.execute({})} />
              </div>
              <MetricGrid
                data={data}
                rows={rows}
                families={["camera"]}
                onSelect={setSelectedId}
                onShow={showEpisodes}
                selectedId={selectedId}
                explainer=""
                rankingInfo="Episodes ranked worst-first on the camera metrics. Each is taken from the worst camera of the episode, and all five vote as one group."
                signalNoun="camera"
              />
            </div>
          ) : (
            <Card title="Vision">
              <div style={{ padding: "32px 8px", textAlign: "center", color: theme.textDim, fontSize: 13 }}>
                <div style={{ marginBottom: 14, maxWidth: 520, marginLeft: "auto", marginRight: "auto", lineHeight: 1.5 }}>
                  Camera metrics score blur, exposure, clipped pixels, frozen feeds and video-action lag. They decode
                  video, so they take longer than the other metrics. The button opens the compute form on its Camera tab,
                  and a run re-ranks every episode with the camera group included.
                </div>
                <Button label="Compute camera metrics" primary onClick={() => visionOp.execute({})} />
              </div>
            </Card>
          ))}
        {activeTab === "language" && (
          <LanguageTab data={data} rows={rows} onSelect={setSelectedId} onShow={showEpisodes} selectedId={selectedId} />
        )}

        {selectedRow && (
          <div ref={inspectorRef} style={{ marginTop: 12 }}>
            <Inspector
              data={data}
              row={selectedRow}
              profile={activeProfile}
              onOpenViewer={openEpisode}
              onClose={() => setSelectedId(null)}
            />
          </div>
        )}
      </div>

      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          padding: "10px 16px",
          borderTop: `1px solid ${theme.cardBorder}`,
          flexWrap: "wrap",
        }}
      >
        <Button label={`Tag ${tagScope}: review`} onClick={() => tag("review")} />
        <Button label={`Tag ${tagScope}: exclude-candidate`} onClick={() => tag("exclude-candidate")} />
        <Button label={`Tag ${tagScope}: relabel`} onClick={() => tag("relabel")} />
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: theme.textDim }}>
          {rows.length} scored episode(s) shown
        </span>
        <Button label="Refresh" onClick={() => dataOp.execute({})} />
      </div>
    </div>
  );
}

function Center(props: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        minHeight: 240,
      }}
    >
      {props.children}
    </div>
  );
}
