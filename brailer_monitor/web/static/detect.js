const els = {
  cvatZip: document.getElementById("cvatZip"),
  cvatVideo: document.getElementById("cvatVideo"),
  importBtn: document.getElementById("importBtn"),
  importStatus: document.getElementById("importStatus"),
  importLabels: document.getElementById("importLabels"),
  epochs: document.getElementById("epochs"),
  batch: document.getElementById("batch"),
  trainImgsz: document.getElementById("trainImgsz"),
  trainBtn: document.getElementById("trainBtn"),
  resetTrainBtn: document.getElementById("resetTrainBtn"),
  resetAndTrainBtn: document.getElementById("resetAndTrainBtn"),
  stopTrainBtn: document.getElementById("stopTrainBtn"),
  trainStatus: document.getElementById("trainStatus"),
  trainProgressWrap: document.getElementById("trainProgressWrap"),
  trainProgressBar: document.getElementById("trainProgressBar"),
  trainProgressText: document.getElementById("trainProgressText"),
  modelCount: document.getElementById("modelCount"),
  modelList: document.getElementById("modelList"),
  refreshModelsBtn: document.getElementById("refreshModelsBtn"),
  modelFramePanel: document.getElementById("modelFramePanel"),
  modelFrameTitle: document.getElementById("modelFrameTitle"),
  modelFrameStatus: document.getElementById("modelFrameStatus"),
  modelFrameResults: document.getElementById("modelFrameResults"),
  modelFrameMore: document.getElementById("modelFrameMore"),
  detectVideo: document.getElementById("detectVideo"),
  detectUploadSection: document.getElementById("detectUploadSection"),
  detectLakeSection: document.getElementById("detectLakeSection"),
  lakeMedia: document.getElementById("lakeMedia"),
  lakeYearFolder: document.getElementById("lakeYearFolder"),
  lakeVessel: document.getElementById("lakeVessel"),
  lakeStream: document.getElementById("lakeStream"),
  lakeMinuteOffsets: document.getElementById("lakeMinuteOffsets"),
  lakeSecondSuffixes: document.getElementById("lakeSecondSuffixes"),
  lakeUrlPreview: document.getElementById("lakeUrlPreview"),
  lakeStartMonth: document.getElementById("lakeStartMonth"),
  lakeStartDay: document.getElementById("lakeStartDay"),
  lakeStartHour: document.getElementById("lakeStartHour"),
  lakeEndMonth: document.getElementById("lakeEndMonth"),
  lakeEndDay: document.getElementById("lakeEndDay"),
  lakeEndHour: document.getElementById("lakeEndHour"),
  lakeDiscoverBtn: document.getElementById("lakeDiscoverBtn"),
  lakeDiscoverStatus: document.getElementById("lakeDiscoverStatus"),
  detectStreamSection: document.getElementById("detectStreamSection"),
  streamUrl: document.getElementById("streamUrl"),
  streamPreview: document.getElementById("streamPreview"),
  streamDetectFrameOverlay: document.getElementById("streamDetectFrameOverlay"),
  streamDetectOverlay: document.getElementById("streamDetectOverlay"),
  streamDetectOverlayStatus: document.getElementById("streamDetectOverlayStatus"),
  streamPreviewStatus: document.getElementById("streamPreviewStatus"),
  frameStride: document.getElementById("frameStride"),
  confidence: document.getElementById("confidence"),
  detectImgsz: document.getElementById("detectImgsz"),
  useSam: document.getElementById("useSam"),
  skipDarkVideo: document.getElementById("skipDarkVideo"),
  detectBtn: document.getElementById("detectBtn"),
  stopDetectBtn: document.getElementById("stopDetectBtn"),
  resetTimelineBtn: document.getElementById("resetTimelineBtn"),
  saveResultName: document.getElementById("saveResultName"),
  saveResultBtn: document.getElementById("saveResultBtn"),
  savedResultSelect: document.getElementById("savedResultSelect"),
  loadResultBtn: document.getElementById("loadResultBtn"),
  savedResultStatus: document.getElementById("savedResultStatus"),
  postMergeSegments: document.getElementById("postMergeSegments"),
  postRemovePositionOutliers: document.getElementById("postRemovePositionOutliers"),
  postRemoveSizeOutliers: document.getElementById("postRemoveSizeOutliers"),
  postRemoveTallThinBoxes: document.getElementById("postRemoveTallThinBoxes"),
  postRemoveStaticShortTracks: document.getElementById("postRemoveStaticShortTracks"),
  postRemoveTemporalIsolated: document.getElementById("postRemoveTemporalIsolated"),
  postRemoveColorOutliers: document.getElementById("postRemoveColorOutliers"),
  compactTimelineBtn: document.getElementById("compactTimelineBtn"),
  compactTimelineStatus: document.getElementById("compactTimelineStatus"),
  generateReportBtn: document.getElementById("generateReportBtn"),
  refreshReportsBtn: document.getElementById("refreshReportsBtn"),
  reportStatus: document.getElementById("reportStatus"),
  reportLinks: document.getElementById("reportLinks"),
  reportList: document.getElementById("reportList"),
  detectStatus: document.getElementById("detectStatus"),
  detectProgressWrap: document.getElementById("detectProgressWrap"),
  detectProgressBar: document.getElementById("detectProgressBar"),
  detectProgressText: document.getElementById("detectProgressText"),
  frameResults: document.getElementById("frameResults"),
  frameCount: document.getElementById("frameCount"),
  timelineAxis: document.getElementById("timelineAxis"),
  timelineViewport: document.getElementById("timelineViewport"),
  timelineTrack: document.getElementById("timelineTrack"),
  timelineTicks: document.getElementById("timelineTicks"),
  timelineZoomIn: document.getElementById("timelineZoomIn"),
  timelineZoomOut: document.getElementById("timelineZoomOut"),
  timelineZoomReset: document.getElementById("timelineZoomReset"),
  timelineZoomLabel: document.getElementById("timelineZoomLabel"),
  timelineRangeStart: document.getElementById("timelineRangeStart"),
  timelineRangeEnd: document.getElementById("timelineRangeEnd"),
  timelineRail: document.getElementById("timelineRail"),
  detectFrameMore: document.getElementById("detectFrameMore"),
  importFramePanel: document.getElementById("importFramePanel"),
  importFrameToggle: document.getElementById("importFrameToggle"),
  importFrameBody: document.getElementById("importFrameBody"),
  importFrameCount: document.getElementById("importFrameCount"),
  importFrameSplit: document.getElementById("importFrameSplit"),
  importFrameResults: document.getElementById("importFrameResults"),
  importFrameMore: document.getElementById("importFrameMore"),
  frameLightbox: document.getElementById("frameLightbox"),
  frameLightboxGallery: document.getElementById("frameLightboxGallery"),
  frameLightboxViewer: document.getElementById("frameLightboxViewer"),
  frameLightboxViewport: document.getElementById("frameLightboxViewport"),
  frameLightboxImg: document.getElementById("frameLightboxImg"),
  frameLightboxMeta: document.getElementById("frameLightboxMeta"),
  frameLightboxClose: document.getElementById("frameLightboxClose"),
  frameLightboxBack: document.getElementById("frameLightboxBack"),
  frameLightboxZoomIn: document.getElementById("frameLightboxZoomIn"),
  frameLightboxZoomOut: document.getElementById("frameLightboxZoomOut"),
  frameLightboxZoomReset: document.getElementById("frameLightboxZoomReset"),
  frameLightboxZoomLabel: document.getElementById("frameLightboxZoomLabel"),
  frameLightboxPrev: document.getElementById("frameLightboxPrev"),
  frameLightboxNext: document.getElementById("frameLightboxNext"),
  frameLightboxCounter: document.getElementById("frameLightboxCounter"),
};

const POLL_MS = 500;
const IMPORT_FRAME_LIMIT = 48;
const MODEL_FRAME_PAGE_SIZE = 12;
const DETECT_FRAME_LIMIT = 200;
const TIMELINE_ZOOM_MIN = 1;
const TIMELINE_ZOOM_MAX = 48;
const TIMELINE_ZOOM_STEP = 1.35;
const FRAME_VIEWER_ZOOM_MIN = 1;
const FRAME_VIEWER_ZOOM_MAX = 8;
const FRAME_VIEWER_ZOOM_STEP = 1.2;
const LAKE_DISCOVER_TIMEOUT_MS = 45000;

let lightboxSegment = null;
let lightboxFrames = [];
let lightboxFrameIndex = 0;
let lightboxViewerZoom = 1;
let lightboxPanX = 0;
let lightboxPanY = 0;
let lightboxDragPointer = null;
let timelineClickTimer = null;

let pollTimer = null;
let handledDetectJobId = null;
let activeDetectJobId = null;
let importFrameOffset = 0;
let importFrameTotal = 0;
let detectFrameOffset = 0;
let detectFrameTotal = 0;
let lastTimelineEventCount = 0;
let lastTimelineRefreshAt = 0;
let timelineRefreshPending = false;
let resetTimelineActive = false;
let timelineRange = null;
let timelineAxisSegments = [];
let timelineZoom = 1;
let detectSourceMode = "upload";
let lakeVideosReady = 0;
let lakeSpec = null;
let detectFrameJobId = null;
let lastPipelineState = null;
let currentLoadedResultId = null;
let importFramesLoaded = false;
let importPanelExpanded = false;
let labelSummaryKey = "";
let detectSession = 0;
let detectSessionActive = false;
let detectStopRequested = false;
let modelListKey = "";
let selectedModelIds = new Set();
let modelSelectionTouched = false;
let modelFrameFrames = [];
let modelFrameVisible = 0;
let streamHls = null;
let streamPreviewUrl = "";
let streamOverlayKey = "";
let streamOverlayHideTimer = null;
let streamRestartTimer = null;
let streamRestartCount = 0;

function sizedPreviewUrl(url, width) {
  if (!url || !width) return url || "";
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}width=${width}`;
}

function formatModelDate(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function formatModelSize(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}MB`;
  return `${(n / 1000).toFixed(0)}KB`;
}

function normalizeModelName(name) {
  return String(name || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
}

function isDefaultExcludedDetectionModel(model) {
  return normalizeModelName(model?.name) === "two_03032220";
}

function hasBrailerClass(model) {
  return (model?.class_names || []).some((name) => String(name || "").toLowerCase().includes("brailer"));
}

function defaultDetectionModelIds(models, explicitIds) {
  const list = Array.isArray(models) ? models : [];
  const availableIds = new Set(list.map((model) => model.id));
  const fromServer = Array.isArray(explicitIds)
    ? explicitIds.filter((id) => availableIds.has(id))
    : [];
  if (fromServer.length) return fromServer;
  return list
    .filter((model) => hasBrailerClass(model) && !isDefaultExcludedDetectionModel(model))
    .map((model) => model.id)
    .filter(Boolean);
}

function selectedDetectionModelIds() {
  const available = new Set((lastPipelineState?.models || []).map((model) => model.id));
  return [...selectedModelIds].filter((id) => !available.size || available.has(id));
}

function renderModels(models, activeId, defaultModelIds) {
  if (!els.modelList) return;
  const list = Array.isArray(models) ? models : [];
  if (els.modelCount) els.modelCount.textContent = String(list.length);

  const availableIds = new Set(list.map((model) => model.id));
  selectedModelIds = new Set([...selectedModelIds].filter((id) => availableIds.has(id)));
  if (!selectedModelIds.size && !modelSelectionTouched && list.length) {
    const defaults = defaultDetectionModelIds(list, defaultModelIds ?? lastPipelineState?.default_detection_model_ids);
    if (defaults.length) {
      defaults.forEach((id) => selectedModelIds.add(id));
    } else {
      const fallbackId = availableIds.has(activeId) ? activeId : list[0].id;
      if (fallbackId) selectedModelIds.add(fallbackId);
    }
  }

  const selectedKey = [...selectedModelIds].sort().join(",");
  const key = JSON.stringify(list.map((m) => [m.id, m.name, m.id === activeId, selectedModelIds.has(m.id), selectedKey]));
  if (key === modelListKey) return;
  modelListKey = key;

  if (!list.length) {
    els.modelList.innerHTML =
      '<div class="model-empty">아직 저장된 모델이 없습니다. 학습을 완료하면 여기에 표시됩니다.</div>';
    return;
  }

  els.modelList.innerHTML = list
    .map((m) => {
      const isActive = m.id === activeId;
      const selected = selectedModelIds.has(m.id);
      const classes = (m.class_names || []).join(", ") || "-";
      const size = formatModelSize(m.size_bytes);
      const metaParts = [
        formatModelDate(m.created_at),
        `epoch ${m.epochs || 0}`,
        `클래스: ${classes}`,
        `train ${m.train_images || 0}/val ${m.val_images || 0}`,
      ];
      if (size) metaParts.push(size);
      return `
        <div class="model-item${isActive ? " active" : ""}" data-model-id="${m.id}">
          <div class="model-item-main">
            <div class="model-item-name">
              ${isActive ? '<span class="model-active-badge">사용 중</span>' : ""}
              ${selected ? '<span class="model-detect-badge">탐지</span>' : ""}
              <span class="model-name-text">${m.name}</span>
            </div>
            <div class="model-item-meta">${metaParts.join(" · ")}</div>
          </div>
          <div class="model-item-actions">
            <label class="model-select">
              <input type="checkbox" data-model-action="toggle-detect" ${selected ? "checked" : ""} />
              탐지
            </label>
            <button type="button" class="secondary small" data-model-action="activate" ${isActive ? "disabled" : ""}>${isActive ? "선택됨" : "사용"}</button>
            <button type="button" class="secondary small" data-model-action="frames">프레임</button>
            <button type="button" class="secondary small" data-model-action="rename">이름변경</button>
            <button type="button" class="secondary small danger" data-model-action="delete">삭제</button>
          </div>
        </div>`;
    })
    .join("");
}

async function loadModels() {
  try {
    const data = await api("/api/pipeline/models");
    renderModels(data.models, data.active_id, data.default_detection_model_ids);
    updateDetectButtonState();
  } catch (err) {
    console.error("loadModels failed", err);
  }
}

function appendFrameCards(container, frames, { previewWidth = 0 } = {}) {
  for (const frame of frames) {
    const card = document.createElement("div");
    card.className = "frame-card";
    card.title = "더블클릭하여 크게 보기";
    const objects = (frame.objects || [])
      .map((obj) => `${obj.class_name} (${obj.shape})`)
      .join("<br>");
    const title = `frame #${frame.frame_index} · ${frame.split}`;
    const previewSrc = sizedPreviewUrl(frame.preview_url, previewWidth);
    card.innerHTML = `
      <img src="${previewSrc}" alt="" loading="lazy" decoding="async" />
      <div class="meta">
        <div>
          <strong>frame #${frame.frame_index}</strong>
          <span class="split-tag">${frame.split}</span>
        </div>
        <div class="objects">${objects || "(객체 없음)"}</div>
      </div>`;
    card.dataset.previewSrc = frame.preview_url;
    card.dataset.frameTitle = title;
    card.dataset.frameObjects = objects;
    container.appendChild(card);
  }
}

function renderMoreModelFrames() {
  if (!els.modelFrameResults) return;
  const next = modelFrameFrames.slice(modelFrameVisible, modelFrameVisible + MODEL_FRAME_PAGE_SIZE);
  appendFrameCards(els.modelFrameResults, next, { previewWidth: 360 });
  modelFrameVisible += next.length;
  if (els.modelFrameMore) {
    els.modelFrameMore.classList.toggle("hidden", modelFrameVisible >= modelFrameFrames.length);
  }
  if (els.modelFrameStatus) {
    els.modelFrameStatus.textContent = `${modelFrameVisible}/${modelFrameFrames.length}프레임 · 모델 저장 시점`;
  }
}

async function showModelFrames(modelId, modelName = "") {
  if (!els.modelFramePanel || !els.modelFrameResults) return;
  els.modelFramePanel.classList.remove("hidden");
  els.modelFrameResults.innerHTML = "";
  modelFrameFrames = [];
  modelFrameVisible = 0;
  els.modelFrameMore?.classList.add("hidden");
  if (els.modelFrameTitle) els.modelFrameTitle.textContent = `${modelName || modelId} 학습 프레임`;
  if (els.modelFrameStatus) els.modelFrameStatus.textContent = "불러오는 중...";
  try {
    const result = await api(`/api/pipeline/models/${modelId}/frames?limit=48`);
    const frames = result.frames || [];
    if (frames.length) {
      modelFrameFrames = frames;
      renderMoreModelFrames();
    } else {
      els.modelFrameResults.innerHTML =
        '<div class="model-empty">이 모델에는 저장된 학습 프레임이 없습니다.</div>';
      if (els.modelFrameStatus) els.modelFrameStatus.textContent = "저장된 프레임 없음";
    }
  } catch (err) {
    if (els.modelFrameStatus) els.modelFrameStatus.textContent = err.message;
  }
}

async function handleModelAction(event) {
  const btn = event.target.closest("[data-model-action]");
  if (!btn) return;
  const item = btn.closest("[data-model-id]");
  const modelId = item?.dataset.modelId;
  if (!modelId) return;
  const action = btn.dataset.modelAction;

  try {
    if (action === "toggle-detect") {
      modelSelectionTouched = true;
      if (btn.checked) selectedModelIds.add(modelId);
      else selectedModelIds.delete(modelId);
      modelListKey = "";
      renderModels(
        lastPipelineState?.models || [],
        lastPipelineState?.active_model_id,
        lastPipelineState?.default_detection_model_ids,
      );
      updateDetectButtonState();
    } else if (action === "activate") {
      btn.disabled = true;
      await api(`/api/pipeline/models/${modelId}/activate`, { method: "POST" });
      modelSelectionTouched = true;
      selectedModelIds.add(modelId);
      modelListKey = "";
      const { state } = await api("/api/pipeline/state");
      lastPipelineState = state;
      renderPipelineState(state);
      renderModels(state.models, state.active_model_id, state.default_detection_model_ids);
      setStatus(els.trainStatus, "선택한 모델을 탐지에 사용합니다.", "ok");
      const modelName = item.querySelector(".model-name-text")?.textContent || "";
      await showModelFrames(modelId, modelName);
    } else if (action === "frames") {
      const modelName = item.querySelector(".model-name-text")?.textContent || "";
      await showModelFrames(modelId, modelName);
    } else if (action === "rename") {
      const current = item.querySelector(".model-name-text")?.textContent || "";
      const name = prompt("모델 이름을 입력하세요", current);
      if (name == null || !name.trim()) return;
      await api(`/api/pipeline/models/${modelId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      modelListKey = "";
      await loadModels();
    } else if (action === "delete") {
      if (!confirm("이 모델을 삭제할까요? 되돌릴 수 없습니다.")) return;
      await api(`/api/pipeline/models/${modelId}`, { method: "DELETE" });
      selectedModelIds.delete(modelId);
      modelListKey = "";
      const { state } = await api("/api/pipeline/state");
      lastPipelineState = state;
      renderPipelineState(state);
      renderModels(state.models, state.active_model_id, state.default_detection_model_ids);
    }
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
    modelListKey = "";
    await loadModels();
  }
}

function isDetectBusy(state = lastPipelineState) {
  return (
    detectSessionActive ||
    !!activeDetectJobId ||
    isDetectActive(state)
  );
}

function endDetectSession() {
  detectSessionActive = false;
}

function isDetectStopping(state = lastPipelineState) {
  return state?.detect_status === "cancelling" || detectStopRequested;
}

function showDetectProgressFromState(state) {
  if (!state || (state.detect_status !== "running" && state.detect_status !== "cancelling")) {
    return false;
  }
  const processed = state.detect_processed_frames || 0;
  const total = state.detect_total_frames || 0;
  const withObjects = state.detect_frames_with_objects || 0;
  const pct =
    state.detect_progress_pct ??
    (total > 0 ? processed / total : processed > 0 ? 0.01 : 0);
  if (processed === 0 && total <= 0) {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
    setStatus(els.detectStatus, detectPreparingMessage(state));
  } else {
    const barText = detectProgressText(processed, total, withObjects, pct, state);
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, pct, barText);
    setStatus(els.detectStatus, detectStatusDetail(processed, total, withObjects, pct, state));
  }
  els.detectBtn.disabled = true;
  return true;
}

function updateStopButtons(state = lastPipelineState) {
  const trainRunning =
    state?.train_status === "running" || state?.train_progress === "cancelling";
  const detectRunning =
    isDetectBusy(state) ||
    state?.detect_status === "running" ||
    state?.detect_status === "cancelling";
  const importDone = state?.import_status === "completed";
  const canResetTrain = importDone && !trainRunning;

  els.stopTrainBtn?.classList.toggle("hidden", !trainRunning);
  els.resetTrainBtn?.classList.toggle("hidden", !canResetTrain);
  els.resetAndTrainBtn?.classList.toggle("hidden", !canResetTrain);
  els.stopDetectBtn?.classList.toggle("hidden", !detectRunning);
  if (els.stopDetectBtn) {
    els.stopDetectBtn.disabled = state?.detect_status === "cancelling" || detectStopRequested;
  }
  if (trainRunning) {
    els.trainBtn.disabled = true;
  }
  if (detectRunning) {
    els.detectBtn.disabled = true;
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  if (!res.ok) {
    if (text) {
      try {
        const payload = JSON.parse(text);
        if (payload?.detail) {
          throw new Error(
            typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail),
          );
        }
      } catch (parseErr) {
        if (!(parseErr instanceof SyntaxError)) {
          throw parseErr;
        }
      }
    }
    if (res.status === 404) {
      throw new Error("API를 찾을 수 없습니다. 웹 서버를 재시작한 뒤 다시 시도하세요.");
    }
    throw new Error(text || res.statusText || `HTTP ${res.status}`);
  }
  if (!text) return {};
  return JSON.parse(text);
}

function getDetectSourceMode() {
  const selected = document.querySelector('input[name="detectSource"]:checked');
  if (selected?.value === "lake") return "lake";
  if (selected?.value === "stream") return "stream";
  return "upload";
}

function setDetectSourceMode(mode) {
  detectSourceMode = mode === "lake" || mode === "stream" ? mode : "upload";
  els.detectUploadSection?.classList.toggle("hidden", detectSourceMode !== "upload");
  els.detectLakeSection?.classList.toggle("hidden", detectSourceMode !== "lake");
  els.detectStreamSection?.classList.toggle("hidden", detectSourceMode !== "stream");
  if (els.detectBtn) {
    els.detectBtn.textContent = detectSourceMode === "stream" ? "스트림 탐지 시작" : "탐지 시작";
  }
  if (els.stopDetectBtn) {
    els.stopDetectBtn.textContent = detectSourceMode === "stream" ? "스트림 탐지 종료" : "탐지 중지";
  }
  if (detectSourceMode === "stream") {
    updateStreamPreview();
  } else {
    clearStreamDetectionOverlay();
  }
  updateDetectButtonState();
}

function streamUrlWithCacheBust(url) {
  if (!url) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}_reload=${Date.now()}`;
}

function destroyStreamHls() {
  if (streamHls) {
    streamHls.destroy();
    streamHls = null;
  }
}

function scheduleStreamPreviewRestart(reason = "stream reset") {
  if (!els.streamPreview || !els.streamUrl) return;
  if (streamRestartTimer) return;
  streamRestartTimer = setTimeout(() => {
    streamRestartTimer = null;
    restartStreamPreview(reason);
  }, 650);
}

function restartStreamPreview(reason = "stream reset") {
  if (!els.streamPreview || !els.streamUrl) return;
  const url = els.streamUrl.value.trim();
  if (!url) return;
  streamRestartCount += 1;
  streamPreviewUrl = "";
  if (els.streamPreviewStatus) {
    els.streamPreviewStatus.textContent = `스트림 재연결 중... (${reason})`;
  }
  updateStreamPreview({ force: true, cacheBust: true });
}

function updateStreamPreview({ force = false, cacheBust = false } = {}) {
  if (!els.streamPreview || !els.streamUrl) return;
  const url = els.streamUrl.value.trim();
  if (!url || (!force && url === streamPreviewUrl)) return;
  streamPreviewUrl = url;
  destroyStreamHls();
  els.streamPreview.controls = false;
  els.streamPreview.autoplay = true;
  els.streamPreview.muted = true;
  els.streamPreview.playsInline = true;
  els.streamPreview.removeAttribute("src");
  els.streamPreview.load();
  const sourceUrl = cacheBust ? streamUrlWithCacheBust(url) : url;
  if (els.streamPreview.canPlayType("application/vnd.apple.mpegurl")) {
    els.streamPreview.src = sourceUrl;
    els.streamPreview.load();
    els.streamPreview.play?.().catch(() => {});
    if (els.streamPreviewStatus) els.streamPreviewStatus.textContent = "스트림 미리보기 준비됨";
  } else if (window.Hls?.isSupported()) {
    streamHls = new window.Hls({
      lowLatencyMode: true,
      liveSyncDurationCount: 2,
      liveMaxLatencyDurationCount: 6,
      manifestLoadingMaxRetry: 999,
      levelLoadingMaxRetry: 999,
      fragLoadingMaxRetry: 999,
      manifestLoadingRetryDelay: 500,
      levelLoadingRetryDelay: 500,
      fragLoadingRetryDelay: 500,
      manifestLoadingMaxRetryTimeout: 5000,
      levelLoadingMaxRetryTimeout: 5000,
      fragLoadingMaxRetryTimeout: 5000,
    });
    streamHls.loadSource(sourceUrl);
    streamHls.attachMedia(els.streamPreview);
    streamHls.on(window.Hls.Events.MANIFEST_PARSED, () => {
      els.streamPreview.play?.().catch(() => {});
    });
    streamHls.on(window.Hls.Events.ERROR, (_event, data) => {
      if (!data?.fatal) return;
      if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
        if (els.streamPreviewStatus) els.streamPreviewStatus.textContent = "스트림 미디어 복구 중...";
        streamHls?.recoverMediaError();
        return;
      }
      if (data.type === window.Hls.ErrorTypes.NETWORK_ERROR) {
        if (els.streamPreviewStatus) els.streamPreviewStatus.textContent = "스트림 네트워크 복구 중...";
        streamHls?.startLoad();
        scheduleStreamPreviewRestart("network");
        return;
      }
      scheduleStreamPreviewRestart("fatal");
    });
    if (els.streamPreviewStatus) els.streamPreviewStatus.textContent = "스트림 미리보기 준비됨";
  } else if (els.streamPreviewStatus) {
    els.streamPreviewStatus.textContent = "이 브라우저에서 HLS 미리보기를 바로 재생할 수 없습니다. ";
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = "스트림 열기";
    els.streamPreviewStatus.appendChild(link);
  }
  clearStreamDetectionOverlay();
}

function clearStreamDetectionOverlay() {
  streamOverlayKey = "";
  if (streamOverlayHideTimer) {
    clearTimeout(streamOverlayHideTimer);
    streamOverlayHideTimer = null;
  }
  const canvas = els.streamDetectOverlay;
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.add("hidden");
  }
  if (els.streamDetectFrameOverlay) {
    els.streamDetectFrameOverlay.removeAttribute("src");
    els.streamDetectFrameOverlay.classList.add("hidden");
  }
  els.streamDetectOverlayStatus?.classList.add("hidden");
}

function streamOverlayContentRect(sourceWidth, sourceHeight, boxWidth, boxHeight) {
  if (!sourceWidth || !sourceHeight || !boxWidth || !boxHeight) {
    return { left: 0, top: 0, width: boxWidth, height: boxHeight };
  }
  const sourceAspect = sourceWidth / sourceHeight;
  const boxAspect = boxWidth / boxHeight;
  if (sourceAspect > boxAspect) {
    const height = boxWidth / sourceAspect;
    return { left: 0, top: (boxHeight - height) / 2, width: boxWidth, height };
  }
  const width = boxHeight * sourceAspect;
  return { left: (boxWidth - width) / 2, top: 0, width, height: boxHeight };
}

function drawStreamDetectionOverlay(state) {
  const canvas = els.streamDetectOverlay;
  const video = els.streamPreview;
  if (!canvas || !video) return;

  const active = state?.detect_status === "running" || state?.detect_status === "cancelling";
  const streamJob =
    detectSourceMode === "stream" ||
    String(state?.detect_video_name || "").startsWith("live_stream_");
  const detections = state?.detect_overlay_detections || [];
  if (!active || !streamJob || !detections.length) {
    clearStreamDetectionOverlay();
    return;
  }

  const key = [
    state.detect_overlay_job_id || state.detect_job_id || "",
    state.detect_overlay_frame_index ?? "",
    state.detect_overlay_updated_at || "",
  ].join(":");
  const wasNewFrame = key !== streamOverlayKey;
  const frameOverlay = els.streamDetectFrameOverlay;
  if (
    !wasNewFrame &&
    canvas.classList.contains("hidden") &&
    (!frameOverlay || frameOverlay.classList.contains("hidden"))
  ) {
    return;
  }
  streamOverlayKey = key;

  const previewPath = state.detect_overlay_preview_path;
  const previewJobId = state.detect_overlay_job_id || state.detect_job_id;
  if (previewPath && previewJobId && frameOverlay) {
    const src = `${previewUrl(previewJobId, previewPath)}?t=${encodeURIComponent(
      state.detect_overlay_updated_at || key,
    )}`;
    if (frameOverlay.getAttribute("src") !== src) {
      frameOverlay.classList.add("hidden");
      frameOverlay.onload = () => {
        if (frameOverlay.getAttribute("src") !== src) return;
        frameOverlay.classList.remove("hidden");
        if (els.streamDetectOverlayStatus) {
          els.streamDetectOverlayStatus.textContent = `탐지 프레임 · 객체 ${detections.length}개`;
          els.streamDetectOverlayStatus.classList.remove("hidden");
        }
      };
      frameOverlay.onerror = () => {
        if (frameOverlay.getAttribute("src") !== src) return;
        frameOverlay.classList.add("hidden");
        frameOverlay.removeAttribute("src");
        els.streamDetectOverlayStatus?.classList.add("hidden");
      };
      frameOverlay.src = src;
    } else if (frameOverlay.complete && frameOverlay.naturalWidth > 0) {
      frameOverlay.classList.remove("hidden");
    }
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    canvas.classList.add("hidden");
    if (frameOverlay.complete && frameOverlay.naturalWidth > 0 && els.streamDetectOverlayStatus) {
      els.streamDetectOverlayStatus.textContent = `탐지 프레임 · 객체 ${detections.length}개`;
      els.streamDetectOverlayStatus.classList.remove("hidden");
    }
    if (wasNewFrame) {
      if (streamOverlayHideTimer) clearTimeout(streamOverlayHideTimer);
      streamOverlayHideTimer = setTimeout(() => {
        if (streamOverlayKey === key) {
          frameOverlay.classList.add("hidden");
          frameOverlay.removeAttribute("src");
          els.streamDetectOverlayStatus?.classList.add("hidden");
        }
      }, 2500);
    }
    return;
  }

  const rect = video.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const canvasWidth = Math.round(width * dpr);
  const canvasHeight = Math.round(height * dpr);
  if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
  }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const sourceWidth = Number(state.detect_overlay_width || video.videoWidth || width);
  const sourceHeight = Number(state.detect_overlay_height || video.videoHeight || height);
  const content = streamOverlayContentRect(sourceWidth, sourceHeight, width, height);
  const scaleX = content.width / Math.max(sourceWidth, 1);
  const scaleY = content.height / Math.max(sourceHeight, 1);
  const drawPolygon = (polygon, fillStyle, strokeStyle, alpha = 0.22, lineWidth = 3) => {
    if (!Array.isArray(polygon) || !polygon.length) return;
    ctx.beginPath();
    polygon.forEach((point, pointIndex) => {
      const x = content.left + Number(point[0] || 0) * scaleX;
      const y = content.top + Number(point[1] || 0) * scaleY;
      if (pointIndex === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.globalAlpha = alpha;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.stroke();
  };

  detections.forEach((det, index) => {
    const color = index % 2 ? "#fbbf24" : "#22c55e";
    const bbox = Array.isArray(det.bbox_xyxy) ? det.bbox_xyxy.map(Number) : [];
    const modelLabel = det.model_name || (det.ensemble_model_names || []).join(", ");
    const label = `${det.class_name || "object"} ${Math.round(Number(det.confidence || 0) * 100)}%${modelLabel ? ` · ${modelLabel}` : ""}`;
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    const yoloPolygon = det.yolo_polygon_xy || (det.segmentation_source === "yolo" ? det.polygon_xy : null);
    const samPolygon = det.segmentation_source === "yolo" ? null : det.polygon_xy;
    drawPolygon(yoloPolygon, "rgba(6, 182, 212, 0.35)", "#06b6d4", 0.32, 2);
    drawPolygon(samPolygon, "rgba(250, 204, 21, 0.34)", "#facc15", 0.28, 3);

    if (bbox.length === 4 && bbox.every(Number.isFinite)) {
      const x = content.left + bbox[0] * scaleX;
      const y = content.top + bbox[1] * scaleY;
      const w = Math.max(1, (bbox[2] - bbox[0]) * scaleX);
      const h = Math.max(1, (bbox[3] - bbox[1]) * scaleY);
      ctx.strokeRect(x, y, w, h);
      ctx.font = "700 13px system-ui, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelY = Math.max(content.top + 4, y - 22);
      ctx.fillStyle = "rgba(5, 7, 10, 0.84)";
      ctx.fillRect(x, labelY, textWidth + 12, 19);
      ctx.fillStyle = color;
      ctx.fillText(label, x + 6, labelY + 14);
    }
  });

  canvas.classList.remove("hidden");
  if (els.streamDetectOverlayStatus) {
    els.streamDetectOverlayStatus.textContent = `객체 ${detections.length}개 탐지`;
    els.streamDetectOverlayStatus.classList.remove("hidden");
  }

  if (wasNewFrame) {
    if (streamOverlayHideTimer) clearTimeout(streamOverlayHideTimer);
    streamOverlayHideTimer = setTimeout(() => {
      if (streamOverlayKey === key) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        canvas.classList.add("hidden");
        els.streamDetectOverlayStatus?.classList.add("hidden");
      }
    }, 3000);
  }
}

function parseLakeNumberList(value, { min = 0, max = 59, label = "값" } = {}) {
  const parts = String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!parts.length) throw new Error(`${label}을 입력하세요.`);
  const seen = new Set();
  const values = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) throw new Error(`${label}은 숫자와 쉼표만 사용할 수 있습니다.`);
    const number = Number(part);
    if (number < min || number > max) throw new Error(`${label}은 ${min}-${max} 사이여야 합니다.`);
    if (!seen.has(number)) {
      seen.add(number);
      values.push(number);
    }
  }
  return values;
}

function formatLakeList(values) {
  return values.map((value) => String(value).padStart(2, "0")).join(",");
}

function readLakeSelection() {
  const minuteOffsets = parseLakeNumberList(els.lakeMinuteOffsets?.value, {
    max: 4,
    label: "시작 분 후보",
  });
  const secondSuffixes = parseLakeNumberList(els.lakeSecondSuffixes?.value, {
    label: "초 suffix",
  }).map((value) => String(value).padStart(2, "0"));
  return {
    media: els.lakeMedia?.value || null,
    year_folder: els.lakeYearFolder?.value || null,
    vessel: els.lakeVessel?.value || null,
    stream: els.lakeStream?.value || null,
    minute_offsets: minuteOffsets,
    second_suffixes: secondSuffixes,
  };
}

function readLakeRange() {
  return {
    ...readLakeSelection(),
    start_month: Number(els.lakeStartMonth.value),
    start_day: Number(els.lakeStartDay.value),
    start_hour: Number(els.lakeStartHour.value),
    end_month: Number(els.lakeEndMonth.value),
    end_day: Number(els.lakeEndDay.value),
    end_hour: Number(els.lakeEndHour.value),
  };
}

function hasUsableModel(state = lastPipelineState) {
  return (
    state?.train_status === "completed" ||
    !!state?.active_model_id ||
    (Array.isArray(state?.models) && state.models.length > 0)
  );
}

function updateDetectButtonState() {
  const modelReady = hasUsableModel();
  const detectActive = isDetectBusy();
  if (!modelReady) {
    els.detectBtn.disabled = true;
    if (lastPipelineState?.import_status === "completed" && !detectActive && els.detectStatus) {
      const hint =
        lastPipelineState?.train_status === "running"
          ? "학습 완료 후 탐지할 수 있습니다."
          : "학습된 모델이 없습니다. 학습을 먼저 완료하거나 모델을 선택하세요.";
      setStatus(els.detectStatus, hint);
    }
    return;
  }
  if (!selectedDetectionModelIds().length && !detectActive) {
    els.detectBtn.disabled = true;
    if (els.detectStatus) {
      setStatus(els.detectStatus, "탐지에 사용할 모델을 하나 이상 선택하세요.");
    }
    return;
  }
  if (detectSourceMode === "lake") {
    els.detectBtn.disabled = lakeVideosReady <= 0;
    return;
  }
  if (detectSourceMode === "stream") {
    els.detectBtn.disabled = !(els.streamUrl?.value || "").trim();
    return;
  }
  els.detectBtn.disabled = !els.detectVideo.files?.length;
}

function populateLakeSelect(selectEl, component) {
  if (!selectEl || !component) return;
  const options = component.options || [];
  const def = component.default || options[0];
  selectEl.innerHTML = options
    .map(
      (value) =>
        `<option value="${value}"${value === def ? " selected" : ""}>${value}</option>`,
    )
    .join("");
}

function updateLakeUrlPreview() {
  if (!els.lakeUrlPreview || !lakeSpec) return;
  let sel;
  try {
    sel = readLakeSelection();
  } catch (err) {
    els.lakeUrlPreview.textContent = err.message;
    return;
  }
  const minute = String(sel.minute_offsets?.[0] ?? lakeSpec.minute_offsets?.[0] ?? 0).padStart(2, "0");
  const suffix = sel.second_suffixes?.[0] || lakeSpec.components?.suffix?.default || "16";
  const filename = `${sel.vessel}_${sel.stream}_YYMMDD_HH${minute}${suffix}.mp4`;
  els.lakeUrlPreview.textContent =
    `${lakeSpec.base_host}${sel.media}/${sel.year_folder}/MM/DD/HH/${filename}` +
    " · 선택한 시작 분에서 5분 간격";
}

async function loadLakeConfig() {
  try {
    const spec = await api("/api/pipeline/lake-videos/config");
    lakeSpec = spec;
    const components = spec.components || {};
    populateLakeSelect(els.lakeMedia, components.media);
    populateLakeSelect(els.lakeYearFolder, components.year_folder);
    populateLakeSelect(els.lakeVessel, components.vessel);
    populateLakeSelect(els.lakeStream, components.stream);
    if (els.lakeMinuteOffsets) {
      els.lakeMinuteOffsets.value = formatLakeList(spec.minute_offsets || [0]);
    }
    if (els.lakeSecondSuffixes) {
      const defaultSuffix = components.suffix?.default || "16";
      els.lakeSecondSuffixes.value = String(defaultSuffix).padStart(2, "0");
    }
    updateLakeUrlPreview();
  } catch (err) {
    console.error("loadLakeConfig failed", err);
  }
}

async function discoverLakeVideos() {
  let range;
  try {
    range = readLakeRange();
  } catch (err) {
    lakeVideosReady = 0;
    setStatus(els.lakeDiscoverStatus, err.message, "error");
    updateDetectButtonState();
    return null;
  }
  setStatus(els.lakeDiscoverStatus, "영상 목록 확인 중... 후보 URL을 병렬 확인합니다.");
  els.lakeDiscoverBtn.disabled = true;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), LAKE_DISCOVER_TIMEOUT_MS);
  try {
    const result = await api("/api/pipeline/lake-videos/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...range, check_exists: true }),
      signal: controller.signal,
    });
    lakeVideosReady = result.found_count || 0;
    const preview = (result.videos || [])
      .slice(0, 3)
      .map((video) => video.filename)
      .join(", ");
    const suffix = result.found_count > 3 ? " ..." : "";
    setStatus(
      els.lakeDiscoverStatus,
      `후보 ${result.candidate_count}개 중 ${result.found_count}개 확인됨${preview ? ` · ${preview}${suffix}` : ""}`,
      lakeVideosReady > 0 ? "ok" : "error",
    );
    updateDetectButtonState();
    return result;
  } catch (err) {
    lakeVideosReady = 0;
    const message =
      err.name === "AbortError"
        ? "영상 확인 시간이 초과되었습니다. 시작 분 후보나 시간 범위를 줄여 다시 시도하세요."
        : err.message;
    setStatus(els.lakeDiscoverStatus, message, "error");
    updateDetectButtonState();
    return null;
  } finally {
    window.clearTimeout(timeoutId);
    els.lakeDiscoverBtn.disabled = false;
  }
}

function beginDetectSession(batchTotal, queuePending = Math.max(0, batchTotal - 1)) {
  clearStreamDetectionOverlay();
  currentLoadedResultId = null;
  detectSession += 1;
  detectSessionActive = true;
  handledDetectJobId = null;
  detectStopRequested = false;
  activeDetectJobId = null;
  lastPipelineState = {
    ...(lastPipelineState || {}),
    detect_status: "running",
    detect_batch_total: batchTotal,
    detect_batch_done: 0,
    detect_batch_index: 0,
    detect_queue_pending: queuePending,
    detect_video_name: null,
    detect_processed_frames: 0,
    detect_total_frames: 0,
    detect_frames_with_objects: 0,
    detect_progress_pct: 0,
    detect_overlay_job_id: null,
    detect_overlay_preview_path: null,
    detect_overlay_frame_index: null,
    detect_overlay_timestamp_sec: null,
    detect_overlay_width: 0,
    detect_overlay_height: 0,
    detect_overlay_detections: null,
    detect_overlay_updated_at: null,
    detect_error: null,
  };
  setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
  setStatus(els.detectStatus, "탐지 준비 중...");
  updateStopButtons(lastPipelineState);
  startPolling();
}

async function finishDetectStart(result, batchTotal) {
  const running = result.jobs?.find((job) => job.job_id) || result.job;
  activeDetectJobId = running?.job_id || result.job_id || null;
  try {
    const { state } = await api("/api/pipeline/state");
    lastPipelineState = {
      ...state,
      detect_status: state.detect_status === "idle" ? "running" : state.detect_status,
      detect_job_id: activeDetectJobId || state.detect_job_id,
      detect_video_name: state.detect_video_name || running?.video_name || null,
      detect_batch_total: result.batch_total ?? result.batch_size ?? batchTotal,
      detect_batch_done: result.batch_done ?? state.detect_batch_done ?? 0,
      detect_batch_index: state.detect_batch_index ?? 0,
      detect_queue_pending: result.queue_pending ?? state.detect_queue_pending ?? 0,
    };
  } catch {
    lastPipelineState = {
      ...(lastPipelineState || {}),
      detect_status: "running",
      detect_job_id: activeDetectJobId || lastPipelineState?.detect_job_id,
      detect_video_name: running?.video_name || lastPipelineState?.detect_video_name || null,
      detect_batch_total: result.batch_total ?? result.batch_size ?? batchTotal,
      detect_batch_done: result.batch_done ?? 0,
      detect_batch_index: lastPipelineState?.detect_batch_index ?? 0,
      detect_queue_pending: result.queue_pending ?? 0,
    };
  }
  // Backend has accepted the request and is now running/queued; the transient
  // "session" flag is no longer needed (prevents it from blocking terminal UI).
  if (activeDetectJobId || (result.queue_pending || 0) > 0 || result.batch_total > 0) {
    endDetectSession();
  }
  const s = lastPipelineState;
  if (activeDetectJobId) {
    showDetectProgress(
      s.detect_processed_frames || 0,
      s.detect_total_frames || 0,
      s.detect_frames_with_objects || 0,
      s.detect_progress_pct || 0,
      s,
    );
  } else if ((result.queue_pending || 0) > 0) {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
    setStatus(els.detectStatus, `대기열 ${result.queue_pending}개 · 첫 비디오 시작 중...`);
  } else {
    showDetectProgressFromState(s);
  }
  updateStopButtons(s);
  startPolling();
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

function setProgress(wrap, bar, textEl, pct, text) {
  if (pct == null) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  bar.style.width = `${Math.round(pct * 100)}%`;
  textEl.textContent = text;
}

function detectBatchSuffix(state) {
  const parts = [];
  if (state?.detect_batch_total > 1) {
    const current = state.detect_batch_index || Math.min((state.detect_batch_done || 0) + 1, state.detect_batch_total);
    parts.push(`비디오 ${current}/${state.detect_batch_total}`);
  }
  if (state?.detect_video_name) {
    parts.push(state.detect_video_name);
  }
  if (state?.detect_queue_pending > 0) {
    parts.push(`대기 ${state.detect_queue_pending}`);
  }
  return parts.length ? ` · ${parts.join(" · ")}` : "";
}

function detectProgressText(processed, total, withObjects, pct, state) {
  if (total <= 0) {
    return `스트림 탐지 중 · ${processed}프레임 처리 · 객체 ${withObjects}개${detectBatchSuffix(state)}`;
  }
  const totalLabel = total > 0 ? total : "?";
  return `탐지 중 ${processed}/${totalLabel} 프레임 · 객체 ${withObjects}개 (${Math.round(pct * 100)}%)${detectBatchSuffix(state)}`;
}

function detectPreparingMessage(state) {
  if (state?.detect_status === "cancelling") {
    return "탐지 중지 중...";
  }
  if (state?.detect_error && state.detect_status === "running") {
    return `${state.detect_error}${detectBatchSuffix(state)}`;
  }
  return `영상 준비 중${detectBatchSuffix(state)}`;
}

function detectStatusDetail(processed, total, withObjects, pct, state) {
  if (state?.detect_status === "cancelling") {
    return "탐지 중지 중...";
  }
  if (processed === 0 && total <= 0) {
    return detectPreparingMessage(state);
  }
  if (total <= 0) {
    return `스트림 탐지 진행 중 · ${processed}프레임 처리 · 객체 ${withObjects}개${detectBatchSuffix(state)}`;
  }
  if (processed === 0 && total > 0) {
    if (state?.detect_error && state.detect_status === "running") {
      return `${state.detect_error}${detectBatchSuffix(state)}`;
    }
    return `모델 로딩 중 · ${total}프레임 예정${detectBatchSuffix(state)}`;
  }
  return `탐지 진행 중 · 객체 ${withObjects}개 (${Math.round(pct * 100)}%)${detectBatchSuffix(state)}`;
}

function shapeSummary(obj) {
  const parts = [];
  if (obj.polygon_count > 0) parts.push(`polygon ${obj.polygon_count}`);
  if (obj.box_count > 0) parts.push(`box ${obj.box_count}`);
  return parts.length ? parts.join(", ") : "annotation 없음";
}

function renderLabelSummary(summary) {
  const wrap = els.importLabels;
  if (!summary?.objects?.length) {
    wrap.classList.add("hidden");
    wrap.innerHTML = "";
    labelSummaryKey = "";
    return;
  }

  const key = JSON.stringify(summary.objects.map((obj) => [obj.name, obj.annotation_count]));
  const isLoading = wrap.textContent.includes("annotation 읽는 중");
  if (key === labelSummaryKey && !isLoading) return;
  labelSummaryKey = key;

  const items = summary.objects
    .map((obj) => {
      const color = obj.color || "#3b82f6";
      const typeLabel = obj.cvat_type ? ` · CVAT ${obj.cvat_type}` : "";
      return `
        <li class="label-item">
          <span class="label-swatch" style="background:${color}"></span>
          <div class="label-name">${obj.name}</div>
          <div class="label-meta">
            annotation ${obj.annotation_count}개 · ${obj.frame_count}프레임 · ${shapeSummary(obj)}${typeLabel}
          </div>
        </li>`;
    })
    .join("");

  wrap.classList.remove("hidden");
  wrap.innerHTML = `
    <h3>정의된 객체 (${summary.objects.length})</h3>
    <ul class="label-list">${items}</ul>
    <div class="label-summary-foot">
      총 ${summary.total_annotations} annotation · ${summary.annotated_frames}프레임 · YOLO ${summary.task_type}
    </div>`;
}

function appendImportFrameCards(frames) {
  for (const frame of frames) {
    const card = document.createElement("div");
    card.className = "frame-card";
    card.title = "더블클릭하여 크게 보기";
    const objects = (frame.objects || [])
      .map((obj) => `${obj.class_name} (${obj.shape})`)
      .join("<br>");
    const title = `frame #${frame.frame_index} · ${frame.split}`;
    card.innerHTML = `
      <img src="${frame.preview_url}" alt="" loading="lazy" />
      <div class="meta">
        <div>
          <strong>frame #${frame.frame_index}</strong>
          <span class="split-tag">${frame.split}</span>
        </div>
        <div class="objects">${objects || "(객체 없음)"}</div>
      </div>`;
    card.dataset.previewSrc = frame.preview_url;
    card.dataset.frameTitle = title;
    card.dataset.frameObjects = objects;
    els.importFrameResults.appendChild(card);
  }
}

function handleFrameCardDblClick(event) {
  clearTimeout(timelineClickTimer);
  const segmentRow = event.target.closest(".timeline-segment-row, .timeline-segment-marker");
  if (segmentRow?.dataset.segmentId) {
    openSegmentLightboxById(segmentRow.dataset.segmentId);
    return;
  }
  const card = event.target.closest(".frame-card");
  if (!card?.dataset.previewSrc) return;
  openFrameLightbox(card.dataset.previewSrc, card.dataset.frameTitle, card.dataset.frameObjects);
}

function showImportFramePanel(meta) {
  const total = (meta?.train_images || 0) + (meta?.val_images || 0);
  if (total === 0) {
    els.importFramePanel.classList.add("hidden");
    return;
  }
  els.importFramePanel.classList.remove("hidden");
  els.importFrameCount.textContent = total;
}

function setImportPanelExpanded(expanded) {
  importPanelExpanded = expanded;
  els.importFramePanel.classList.toggle("collapsed", !expanded);
  els.importFrameBody.classList.toggle("hidden", !expanded);
  els.importFrameToggle.setAttribute("aria-expanded", String(expanded));
  if (expanded && !importFramesLoaded) {
    loadImportFrames({ reset: true });
  }
}

async function loadImportFrames({ reset = false } = {}) {
  if (reset) {
    importFrameOffset = 0;
    els.importFrameResults.innerHTML = "";
    importFramesLoaded = false;
  }
  const split = els.importFrameSplit.value;
  try {
    const data = await api(
      `/api/pipeline/dataset/frames?split=${encodeURIComponent(split)}&offset=${importFrameOffset}&limit=${IMPORT_FRAME_LIMIT}`,
    );
    importFrameTotal = data.total || 0;
    els.importFrameCount.textContent = importFrameTotal;
    if (importFrameTotal === 0) {
      els.importFramePanel.classList.add("hidden");
      return;
    }
    els.importFramePanel.classList.remove("hidden");
    appendImportFrameCards(data.frames || []);
    importFrameOffset += (data.frames || []).length;
    importFramesLoaded = importFrameOffset > 0;
    els.importFrameMore.classList.toggle("hidden", importFrameOffset >= importFrameTotal);
  } catch (err) {
    if (reset) {
      els.importFrameResults.innerHTML = "";
    }
    console.error("loadImportFrames failed", err);
  }
}

async function previewCvatFile(file) {
  if (!file) {
    renderLabelSummary(null);
    return;
  }
  const form = new FormData();
  form.append("file", file);
  els.importLabels.classList.remove("hidden");
  els.importLabels.innerHTML = '<div class="label-summary-foot">annotation 읽는 중...</div>';
  try {
    const summary = await api("/api/pipeline/preview-cvat", { method: "POST", body: form });
    renderLabelSummary(summary);
  } catch (err) {
    renderLabelSummary(null);
    setStatus(els.importStatus, err.message, "error");
  }
}

function showDetectProgress(processed, total, withObjects, pct, state = lastPipelineState) {
  if (processed === 0 && total <= 0) {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
    setStatus(els.detectStatus, detectPreparingMessage(state));
  } else {
    setProgress(
      els.detectProgressWrap,
      els.detectProgressBar,
      els.detectProgressText,
      pct,
      detectProgressText(processed, total, withObjects, pct, state),
    );
    setStatus(els.detectStatus, detectStatusDetail(processed, total, withObjects, pct, state));
  }
  els.detectBtn.disabled = true;
}

function showDetectComplete(job, state = lastPipelineState) {
  const batch =
    state?.detect_batch_total > 1
      ? ` (${state.detect_batch_done}/${state.detect_batch_total} 비디오)`
      : "";
  const timelineCount = state?.detect_timeline_events ?? job.frames_with_detections;
  setProgress(
    els.detectProgressWrap,
    els.detectProgressBar,
    els.detectProgressText,
    1,
    `완료: ${job.processed_frames}프레임 · 객체 ${job.frames_with_detections}프레임${batch}`,
  );
  const failNote = state?.detect_error ? ` · ${state.detect_error}` : "";
  setStatus(
    els.detectStatus,
    `탐지 완료: 누적 ${timelineCount}건 · 마지막 ${job.video_name}${failNote}`,
    "ok",
  );
  els.detectBtn.disabled = false;
}

function isTrainActive(state) {
  return state?.train_status === "running";
}

function isDetectActive(state) {
  if (!state) return false;
  return (
    state.detect_status === "running" ||
    state.detect_status === "cancelling" ||
    (state.detect_queue_pending || 0) > 0 ||
    ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
      state.detect_status !== "error" &&
      state.detect_status !== "cancelled")
  );
}

function isDetectBatchContinuing(state, currentJobId = null) {
  if (!state || state.detect_status === "error" || state.detect_status === "cancelled") {
    return false;
  }
  return (
    (state.detect_queue_pending || 0) > 0 ||
    (state.detect_status === "running" &&
      (!currentJobId || !state.detect_job_id || state.detect_job_id !== currentJobId)) ||
    ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
      state.detect_status !== "completed")
  );
}

// Whether a detection job is genuinely in flight (ignores the transient
// "session requested" flag so terminal states can always clear the UI).
function isDetectLive(state = lastPipelineState) {
  return (
    !!activeDetectJobId ||
    state?.detect_status === "running" ||
    state?.detect_status === "cancelling" ||
    (state?.detect_queue_pending || 0) > 0
  );
}

function trainStatusText(state) {
  const epoch = state.train_epoch || 0;
  const total = state.train_epochs || Number(els.epochs.value) || 1;
  const pct = Math.round((state.train_progress_pct || 0) * 100);
  const progress = state.train_progress || "";
  const batch = state.train_batch || 0;
  const batches = state.train_batches || 0;
  const batchText = batches ? ` · batch ${batch}/${batches}` : "";

  if (progress === "cancelling") {
    return {
      bar: `학습 중지 중 · epoch ${epoch}/${total}${batchText}`,
      status: "학습 중지 요청됨 · 현재 배치가 끝나면 멈춥니다...",
    };
  }
  if (progress === "starting") {
    return { bar: "학습 준비 중...", status: "모델·데이터 로딩 중..." };
  }
  if (progress === "training" && epoch === 0) {
    return { bar: "학습 시작 중...", status: "첫 epoch 준비 중..." };
  }
  if (epoch > 0) {
    return {
      bar: `epoch ${epoch}/${total}${batchText} (${pct}%)`,
      status: `학습 중 · epoch ${epoch}/${total}${batchText} · ${pct}%`,
    };
  }
  return { bar: progress || "학습 중...", status: progress || "학습 중..." };
}

function renderPipelineState(state) {
  const meta = state.dataset_meta;
  currentLoadedResultId = state.loaded_saved_result_id || null;
  if (els.resetTimelineBtn) {
    els.resetTimelineBtn.disabled = resetTimelineActive;
    if (resetTimelineActive) {
      els.resetTimelineBtn.setAttribute("aria-disabled", "true");
    } else {
      els.resetTimelineBtn.removeAttribute("aria-disabled");
    }
  }
  if (els.savedResultSelect && currentLoadedResultId) {
    els.savedResultSelect.value = currentLoadedResultId;
  }

  if (state.models !== undefined) {
    renderModels(state.models, state.active_model_id, state.default_detection_model_ids);
  }

  if (state.import_status === "completed" && meta) {
    setStatus(
      els.importStatus,
      `Import 완료 · train ${meta.train_images} / val ${meta.val_images} · ${meta.task_type}`,
      "ok",
    );
    showImportFramePanel(meta);
    els.trainBtn.disabled = false;
  } else if (state.import_status === "error") {
    setStatus(els.importStatus, state.import_error, "error");
  } else if (state.import_status === "running") {
    setStatus(els.importStatus, "Import 진행 중...");
  }

  if (state.train_status === "running") {
    const { bar, status } = trainStatusText(state);
    setProgress(
      els.trainProgressWrap,
      els.trainProgressBar,
      els.trainProgressText,
      state.train_progress_pct ?? 0,
      bar,
    );
    setStatus(els.trainStatus, status);
    els.trainBtn.disabled = true;
  } else if (state.train_status === "completed") {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, 1, "학습 완료");
    const activeLabel = state.active_model_name
      ? `사용 모델: ${state.active_model_name}`
      : `학습 완료: ${state.train_weights}`;
    setStatus(els.trainStatus, activeLabel, "ok");
    els.trainBtn.disabled = false;
    updateDetectButtonState();
  } else if (state.train_status === "cancelled") {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, null);
    setStatus(els.trainStatus, state.train_error || "학습이 중지되었습니다.", "error");
    els.trainBtn.disabled = false;
  } else if (state.train_status === "error") {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, null);
    setStatus(els.trainStatus, state.train_error, "error");
    els.trainBtn.disabled = false;
  } else {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, null);
    if (state.import_status === "completed" && state.train_status === "idle") {
      setStatus(els.trainStatus, "학습된 모델 없음 · 학습을 시작하세요.");
    }
  }

  if (state.detect_status === "running" || state.detect_status === "cancelling") {
    showDetectProgressFromState(state);
  } else if (detectSessionActive && !isDetectLive(state)) {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
    setStatus(els.detectStatus, "비디오 업로드·탐지 준비 중...");
    els.detectBtn.disabled = true;
  } else if (state.detect_status === "completed" && !isDetectLive(state) && !detectSessionActive) {
    const batch =
      state.detect_batch_total > 1
        ? ` (${state.detect_batch_done}/${state.detect_batch_total} 비디오)`
        : "";
    setProgress(
      els.detectProgressWrap,
      els.detectProgressBar,
      els.detectProgressText,
      1,
      `탐지 완료 · ${state.detect_processed_frames || 0}프레임 · 객체 ${state.detect_frames_with_objects || 0}개${batch}`,
    );
    const failNote = state.detect_error ? ` · ${state.detect_error}` : "";
    setStatus(
      els.detectStatus,
      `탐지 완료 · 누적 ${state.detect_timeline_events || 0}건${failNote}`,
      "ok",
    );
    els.detectBtn.disabled = false;
    detectStopRequested = false;
    endDetectSession();
    updateDetectButtonState();
  } else if (state.detect_status === "error" && !detectSessionActive) {
    activeDetectJobId = null;
    endDetectSession();
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, state.detect_error || "탐지 실패", "error");
    els.detectBtn.disabled = false;
    detectStopRequested = false;
    updateDetectButtonState();
  } else if (state.detect_status === "cancelled" && !isDetectLive(state) && !detectSessionActive) {
    activeDetectJobId = null;
    endDetectSession();
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, state.detect_error || "탐지가 중지되었습니다.", "error");
    detectStopRequested = false;
    updateDetectButtonState();
  }

  maybeRefreshTimelineFromState(state);
  drawStreamDetectionOverlay(state);
  updateStopButtons(state);
}

function formatConfidence(confidence) {
  if (confidence == null || Number.isNaN(confidence)) return "-";
  return `${(confidence * 100).toFixed(0)}%`;
}

function bboxAreaPx(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const w = Math.max(0, Number(bbox[2]) - Number(bbox[0]));
  const h = Math.max(0, Number(bbox[3]) - Number(bbox[1]));
  const area = Math.round(w * h);
  return area > 0 ? area : null;
}

function resolveAreaPx(item) {
  const direct = Number(item?.max_area_px ?? item?.area_px);
  if (Number.isFinite(direct) && direct > 0) return direct;
  return bboxAreaPx(item?.bbox_xyxy);
}

function formatAreaPx(areaPx) {
  const px = Number(areaPx);
  if (!Number.isFinite(px) || px <= 0) return "-";
  if (px >= 1_000_000) return `${(px / 1_000_000).toFixed(2)}M px`;
  if (px >= 10_000) return `${(px / 1000).toFixed(1)}k px`;
  return `${px.toLocaleString()} px`;
}

function formatSegmentStats(segmentOrFrame) {
  const conf = formatConfidence(segmentOrFrame.max_confidence ?? segmentOrFrame.confidence);
  const area = formatAreaPx(resolveAreaPx(segmentOrFrame));
  const maskArea = Number(segmentOrFrame.mask_area_px);
  const maskWidth = Number(segmentOrFrame.mask_width_px);
  const maskHeight = Number(segmentOrFrame.mask_height_px);
  const polygonPoints = Number(segmentOrFrame.polygon_point_count);
  const parts = [`신뢰도 ${conf}`, `면적 ${area}`];
  if (Number.isFinite(maskArea) && maskArea > 0) {
    parts.push(`mask ${formatAreaPx(maskArea)}`);
  }
  if (Number.isFinite(maskWidth) && maskWidth > 0 && Number.isFinite(maskHeight) && maskHeight > 0) {
    parts.push(`mask ${maskWidth}x${maskHeight}`);
  }
  if (Number.isFinite(polygonPoints) && polygonPoints > 0) {
    parts.push(`polygon ${polygonPoints}점`);
  }
  return parts.join(" · ");
}

function framePreviewSrc(frame) {
  return previewUrl(frame.job_id, frame.preview_path);
}

function clampFrameViewerZoom(value) {
  return Math.min(FRAME_VIEWER_ZOOM_MAX, Math.max(FRAME_VIEWER_ZOOM_MIN, value));
}

function resetLightboxViewerTransform() {
  lightboxViewerZoom = 1;
  lightboxPanX = 0;
  lightboxPanY = 0;
  applyLightboxViewerTransform();
}

function applyLightboxViewerTransform() {
  if (!els.frameLightboxImg) return;
  els.frameLightboxImg.style.transform = `translate(calc(-50% + ${lightboxPanX}px), calc(-50% + ${lightboxPanY}px)) scale(${lightboxViewerZoom})`;
  if (els.frameLightboxZoomLabel) {
    els.frameLightboxZoomLabel.textContent = `${Math.round(lightboxViewerZoom * 100)}%`;
  }
}

function setLightboxViewerZoom(nextZoom) {
  lightboxViewerZoom = clampFrameViewerZoom(nextZoom);
  applyLightboxViewerTransform();
}

function showLightboxMode(mode) {
  const isGallery = mode === "gallery";
  els.frameLightboxGallery?.classList.toggle("hidden", !isGallery);
  els.frameLightboxViewer?.classList.toggle("hidden", isGallery);
  els.frameLightboxBack?.classList.toggle(
    "hidden",
    isGallery || !lightboxFrames || lightboxFrames.length <= 1,
  );
}

function renderLightboxSegmentMeta(segment, frame) {
  if (!els.frameLightboxMeta) return;
  const range =
    segment?.time_label ||
    `${segment?.start_absolute_time_label || ""} – ${segment?.end_absolute_time_label || ""}`;
  if (frame && segment) {
    const time = frame.absolute_time_label || fmtTime(frame.timestamp_sec);
    els.frameLightboxMeta.innerHTML = `
      <div><strong>${segment.class_name}</strong> · ${time}</div>
      <div class="objects">${segment.video_name} · ${formatSegmentStats(frame)}</div>`;
    return;
  }
  if (segment) {
    els.frameLightboxMeta.innerHTML = `
      <div><strong>${segment.class_name}</strong> · ${range}</div>
      <div class="objects">${segment.video_name} · ${lightboxFrames.length}프레임 · ${formatSegmentStats(segment)}</div>`;
    return;
  }
  els.frameLightboxMeta.innerHTML = "";
}

function showLightboxFrame(index) {
  if (!lightboxFrames.length || !els.frameLightboxImg) return;
  lightboxFrameIndex = Math.max(0, Math.min(index, lightboxFrames.length - 1));
  const frame = lightboxFrames[lightboxFrameIndex];
  const src = framePreviewSrc(frame);
  if (!src) return;

  showLightboxMode("viewer");
  resetLightboxViewerTransform();
  els.frameLightboxImg.onload = () => resetLightboxViewerTransform();
  els.frameLightboxImg.src = src;
  renderLightboxSegmentMeta(lightboxSegment, frame);

  if (els.frameLightboxCounter) {
    els.frameLightboxCounter.textContent = `${lightboxFrameIndex + 1} / ${lightboxFrames.length}`;
  }
  if (els.frameLightboxPrev) {
    els.frameLightboxPrev.disabled = lightboxFrameIndex <= 0;
  }
  if (els.frameLightboxNext) {
    els.frameLightboxNext.disabled = lightboxFrameIndex >= lightboxFrames.length - 1;
  }
}

function openLightboxGallery(segment, frames) {
  if (!els.frameLightbox || !frames?.length) return;
  lightboxSegment = segment;
  lightboxFrames = frames;
  const gallery = els.frameLightboxGallery;
  gallery.innerHTML = frames
    .map((frame, index) => {
      const src = framePreviewSrc(frame);
      const time = frame.absolute_time_label || fmtTime(frame.timestamp_sec);
      const stats = formatSegmentStats(frame);
      return `
        <div class="lightbox-gallery-item" data-frame-index="${index}" title="클릭하여 확대">
          <img src="${src}" alt="" loading="lazy" />
          <div class="caption">${time} · ${stats}</div>
        </div>`;
    })
    .join("");
  showLightboxMode("gallery");
  renderLightboxSegmentMeta(segment);
  els.frameLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function openFrameLightbox(imgSrc, titleHtml, objectsHtml) {
  if (!imgSrc || !els.frameLightbox) return;
  lightboxSegment = null;
  lightboxFrames = [{ preview_path: null, job_id: null, _src: imgSrc, _title: titleHtml, _objects: objectsHtml }];
  lightboxFrameIndex = 0;
  showLightboxMode("viewer");
  resetLightboxViewerTransform();
  els.frameLightboxImg.src = imgSrc;
  els.frameLightboxMeta.innerHTML = `
    <div><strong>${titleHtml}</strong></div>
    <div class="objects">${objectsHtml || "(없음)"}</div>`;
  if (els.frameLightboxCounter) els.frameLightboxCounter.textContent = "1 / 1";
  if (els.frameLightboxPrev) els.frameLightboxPrev.disabled = true;
  if (els.frameLightboxNext) els.frameLightboxNext.disabled = true;
  els.frameLightboxBack?.classList.add("hidden");
  els.frameLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function previewUrl(jobId, previewPath) {
  if (!jobId || !previewPath) return "";
  return `/api/pipeline/detect/${jobId}/previews/${previewPath}`;
}

function openSegmentLightbox(segment, frames) {
  openLightboxGallery(segment, frames);
}

async function openSegmentViewerById(segmentId, frameIndex = 0) {
  try {
    const { segment, frames } = await api(
      `/api/pipeline/detect/timeline/segment/${encodeURIComponent(segmentId)}`,
    );
    if (!frames?.length) {
      alert("표시할 프레임이 없습니다.");
      return;
    }
    lightboxSegment = segment;
    lightboxFrames = frames;
    showLightboxFrame(frameIndex);
    els.frameLightbox?.classList.remove("hidden");
    document.body.style.overflow = "hidden";
  } catch (err) {
    console.error("openSegmentViewerById failed", err);
    alert(err.message || "프레임을 불러오지 못했습니다.");
  }
}

async function openSegmentLightboxById(segmentId, segmentSummary) {
  try {
    const { segment, frames } = await api(
      `/api/pipeline/detect/timeline/segment/${encodeURIComponent(segmentId)}`,
    );
    openSegmentLightbox(segment || segmentSummary, frames);
  } catch (err) {
    console.error("openSegmentLightboxById failed", err);
    alert(err.message || "세그먼트를 불러오지 못했습니다.");
  }
}

function closeFrameLightbox() {
  if (!els.frameLightbox) return;
  els.frameLightbox.classList.add("hidden");
  if (els.frameLightboxImg) {
    els.frameLightboxImg.removeAttribute("src");
    els.frameLightboxImg.onload = null;
  }
  if (els.frameLightboxGallery) {
    els.frameLightboxGallery.classList.add("hidden");
    els.frameLightboxGallery.innerHTML = "";
  }
  els.frameLightboxViewer?.classList.add("hidden");
  lightboxSegment = null;
  lightboxFrames = [];
  lightboxFrameIndex = 0;
  lightboxDragPointer = null;
  els.frameLightboxViewport?.classList.remove("is-dragging");
  resetLightboxViewerTransform();
  document.body.style.overflow = "";
}

function handleTimelineFrameClick(event) {
  const row = event.target.closest(".timeline-segment-row");
  const marker = event.target.closest(".timeline-segment-marker");
  const segmentId = row?.dataset.segmentId || marker?.dataset.segmentId;
  if (!segmentId) return;
  clearTimeout(timelineClickTimer);
  timelineClickTimer = setTimeout(() => {
    openSegmentViewerById(segmentId, 0);
  }, 220);
}

function handleLightboxViewportWheel(event) {
  if (els.frameLightboxViewer?.classList.contains("hidden")) return;
  event.preventDefault();
  const delta = event.deltaY < 0 ? FRAME_VIEWER_ZOOM_STEP : 1 / FRAME_VIEWER_ZOOM_STEP;
  setLightboxViewerZoom(lightboxViewerZoom * delta);
}

function handleLightboxViewportPointerDown(event) {
  if (els.frameLightboxViewer?.classList.contains("hidden")) return;
  if (event.button !== 0) return;
  lightboxDragPointer = {
    x: event.clientX,
    y: event.clientY,
    panX: lightboxPanX,
    panY: lightboxPanY,
  };
  els.frameLightboxViewport?.classList.add("is-dragging");
  els.frameLightboxViewport?.setPointerCapture?.(event.pointerId);
}

function handleLightboxViewportPointerMove(event) {
  if (!lightboxDragPointer) return;
  lightboxPanX = lightboxDragPointer.panX + (event.clientX - lightboxDragPointer.x);
  lightboxPanY = lightboxDragPointer.panY + (event.clientY - lightboxDragPointer.y);
  applyLightboxViewerTransform();
}

function handleLightboxViewportPointerUp(event) {
  if (!lightboxDragPointer) return;
  lightboxDragPointer = null;
  els.frameLightboxViewport?.classList.remove("is-dragging");
  els.frameLightboxViewport?.releasePointerCapture?.(event.pointerId);
}

function clampTimelineZoom(value) {
  return Math.min(TIMELINE_ZOOM_MAX, Math.max(TIMELINE_ZOOM_MIN, value));
}

function getTimelineViewportWidth() {
  const width = els.timelineViewport?.clientWidth || 0;
  return Math.max(width, 320);
}

function getTimelineTrackWidth() {
  return Math.round(getTimelineViewportWidth() * timelineZoom);
}

function setTimelineZoom(nextZoom, { keepCenter = true } = {}) {
  const viewport = els.timelineViewport;
  const prevWidth = getTimelineTrackWidth();
  const centerRatio =
    viewport && prevWidth > 0
      ? (viewport.scrollLeft + viewport.clientWidth / 2) / prevWidth
      : 0.5;

  timelineZoom = clampTimelineZoom(nextZoom);
  renderTimelineAxis(timelineRange, timelineAxisSegments);

  if (viewport && keepCenter) {
    const nextWidth = getTimelineTrackWidth();
    viewport.scrollLeft = Math.max(0, centerRatio * nextWidth - viewport.clientWidth / 2);
  }
}

function updateTimelineZoomLabel() {
  if (els.timelineZoomLabel) {
    els.timelineZoomLabel.textContent = `${Math.round(timelineZoom * 100)}%`;
  }
}

function segmentPositionPx(segment, range, trackWidthPx) {
  if (!range?.range_start || !range?.range_end) {
    return { left: trackWidthPx * 0.5, width: 24 };
  }
  const startMs = new Date(segment.start_absolute_time || range.range_start).getTime();
  const endMs = new Date(segment.end_absolute_time || segment.start_absolute_time || range.range_end).getTime();
  const rangeStartMs = new Date(range.range_start).getTime();
  const rangeEndMs = new Date(range.range_end).getTime();
  const total = Math.max(rangeEndMs - rangeStartMs, 1);
  const left = ((startMs - rangeStartMs) / total) * trackWidthPx;
  const span = Math.max(((endMs - startMs) / total) * trackWidthPx, 10 * timelineZoom);
  const minWidth = Math.max(14, 8 * timelineZoom);
  return {
    left: Math.max(0, Math.min(left, trackWidthPx - minWidth)),
    width: Math.max(span, minWidth),
  };
}

function renderTimelineTicks(range, trackWidthPx) {
  if (!els.timelineTicks || !range?.range_start || !range?.range_end) {
    if (els.timelineTicks) els.timelineTicks.innerHTML = "";
    return;
  }
  const startMs = new Date(range.range_start).getTime();
  const endMs = new Date(range.range_end).getTime();
  const total = Math.max(endMs - startMs, 1);
  const hourMs = 3600000;
  const minPxBetweenTicks = 70;
  let stepHours = 1;
  while ((hourMs * stepHours / total) * trackWidthPx < minPxBetweenTicks) {
    stepHours *= 2;
    if (stepHours > 168) break;
  }

  const ticks = [];
  let tickMs = Math.ceil(startMs / hourMs) * hourMs;
  while (tickMs <= endMs) {
    const left = ((tickMs - startMs) / total) * trackWidthPx;
    const label = new Date(tickMs).toLocaleString("ko-KR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    ticks.push(`<div class="timeline-tick" style="left:${left}px"><span>${label}</span></div>`);
    tickMs += hourMs * stepHours;
  }
  els.timelineTicks.style.width = `${trackWidthPx}px`;
  els.timelineTicks.innerHTML = ticks.join("");
}

function renderTimelineAxis(range, segments) {
  if (!els.timelineAxis || !els.timelineRail || !els.timelineTrack) return;
  timelineAxisSegments = segments || timelineAxisSegments;
  if (!range?.range_start || !timelineAxisSegments.length) {
    els.timelineAxis.classList.add("hidden");
    els.timelineRail.innerHTML = "";
    if (els.timelineTicks) els.timelineTicks.innerHTML = "";
    return;
  }

  timelineRange = range;
  const trackWidthPx = getTimelineTrackWidth();
  els.timelineAxis.classList.remove("hidden");
  if (els.timelineRangeStart) els.timelineRangeStart.textContent = range.range_start_label || "";
  if (els.timelineRangeEnd) els.timelineRangeEnd.textContent = range.range_end_label || "";

  els.timelineTrack.style.width = `${trackWidthPx}px`;
  els.timelineRail.style.width = `${trackWidthPx}px`;
  els.timelineRail.style.height = `${Math.round(56 + Math.min(timelineZoom, 8) * 3)}px`;

  renderTimelineTicks(range, trackWidthPx);
  els.timelineRail.innerHTML = timelineAxisSegments
    .map((segment) => {
      const pos = segmentPositionPx(segment, range, trackWidthPx);
      const src = previewUrl(segment.preview_job_id || segment.job_id, segment.preview_path);
      const img = src ? `<img src="${src}" alt="" loading="lazy" />` : "";
      const frames = segment.frame_count > 1 ? `${segment.frame_count}프` : "";
      const showImg = timelineZoom >= 1.5 || pos.width >= 36;
      return `
        <div
          class="timeline-segment-marker"
          style="left:${pos.left}px;width:${pos.width}px;"
          data-segment-id="${segment.segment_id}"
          title="${segment.class_name} · ${segment.time_label || ""} · ${formatSegmentStats(segment)}"
        >
          <span class="marker-bar"></span>
          ${showImg ? img : ""}
          <span class="marker-label">${segment.class_name}${frames ? ` ${frames}` : ""}</span>
        </div>`;
    })
    .join("");
  updateTimelineZoomLabel();
}

function appendTimelineSegments(segments) {
  for (const segment of segments) {
    const row = document.createElement("div");
    row.className = "timeline-segment-row";
    row.title = "클릭: 확대 · 더블클릭: 전체 프레임";
    const src = previewUrl(segment.preview_job_id || segment.job_id, segment.preview_path);
    const img = src ? `<img src="${src}" alt="" loading="lazy" />` : "";
    const frames =
      segment.frame_count > 1 ? `${segment.frame_count}프레임` : "1프레임";
    row.innerHTML = `
      ${img}
      <div class="meta">
        <div class="time"><strong>${segment.class_name}</strong> · ${segment.time_label || "-"}</div>
        <div class="detail">${segment.video_name} · ${frames} · ${formatSegmentStats(segment)}</div>
      </div>`;
    row.dataset.segmentId = segment.segment_id;
    els.frameResults.appendChild(row);
  }
}

async function loadTimeline({ reset = true } = {}) {
  try {
    if (reset) {
      els.frameResults.innerHTML = "";
      detectFrameOffset = 0;
      timelineZoom = 1;
      const { timeline } = await api(
        `/api/pipeline/detect/timeline?offset=0&limit=${DETECT_FRAME_LIMIT}`,
      );
      const pageSegments = timeline.segments || [];
      timelineRange = {
        range_start: timeline.range_start,
        range_end: timeline.range_end,
        range_start_label: timeline.range_start_label,
        range_end_label: timeline.range_end_label,
      };
      timelineAxisSegments = timeline.axis_segments || pageSegments;
      detectFrameTotal = timeline.total || 0;
      lastTimelineEventCount = timeline.event_count ?? detectFrameTotal;
      if (els.frameCount) els.frameCount.textContent = detectFrameTotal;
      renderTimelineAxis(timelineRange, timelineAxisSegments);
      appendTimelineSegments(pageSegments);
      detectFrameOffset = pageSegments.length;
      els.detectFrameMore?.classList.toggle("hidden", detectFrameOffset >= detectFrameTotal);
      if (timeline.videos?.length) {
        handledDetectJobId = timeline.videos[timeline.videos.length - 1].job_id;
      }
      return timeline;
    }

    const offset = detectFrameOffset;
    const { timeline } = await api(
      `/api/pipeline/detect/timeline?offset=${offset}&limit=${DETECT_FRAME_LIMIT}`,
    );
    detectFrameTotal = timeline.total || 0;
    lastTimelineEventCount = timeline.event_count ?? detectFrameTotal;
    if (els.frameCount) els.frameCount.textContent = detectFrameTotal;
    appendTimelineSegments(timeline.segments || []);
    detectFrameOffset += (timeline.segments || []).length;
    els.detectFrameMore?.classList.toggle("hidden", detectFrameOffset >= detectFrameTotal);
    return timeline;
  } catch (err) {
    console.error("loadTimeline failed", err);
    if (reset) {
      els.frameResults.innerHTML = "";
      if (els.frameCount) els.frameCount.textContent = "0";
      els.timelineAxis?.classList.add("hidden");
    }
    return null;
  }
}

function maybeRefreshTimelineFromState(state) {
  const count = Number(
    state?.detect_timeline?.event_count ??
      state?.detect_timeline_events ??
      0,
  );
  if (!Number.isFinite(count) || count <= 0) return;
  if (timelineRefreshPending) return;
  if (count === lastTimelineEventCount && detectFrameTotal > 0) return;

  const now = Date.now();
  if (now - lastTimelineRefreshAt < 800) return;
  lastTimelineRefreshAt = now;
  timelineRefreshPending = true;
  loadTimeline({ reset: true }).finally(() => {
    timelineRefreshPending = false;
  });
}

async function resetTimeline(event) {
  event?.preventDefault();
  if (resetTimelineActive) return;
  const detectionRunning = isDetectLive(lastPipelineState) || isDetectBusy(lastPipelineState);
  const message = detectionRunning
    ? "탐지가 진행 중입니다. 누적된 결과만 초기화하고 탐지는 계속됩니다. 진행할까요?"
    : "누적 탐지 결과를 모두 초기화할까요?";
  if (!confirm(message)) return;

  resetTimelineActive = true;
  if (els.resetTimelineBtn) {
    els.resetTimelineBtn.disabled = true;
    els.resetTimelineBtn.setAttribute("aria-disabled", "true");
  }
  try {
    await api("/api/pipeline/detect/timeline/reset", { method: "POST" });
    detectFrameOffset = 0;
    detectFrameTotal = 0;
    detectFrameJobId = null;
    currentLoadedResultId = null;
    timelineRange = null;
    timelineAxisSegments = [];
    timelineZoom = 1;
    if (els.frameResults) els.frameResults.innerHTML = "";
    if (els.frameCount) els.frameCount.textContent = "0";
    els.timelineAxis?.classList.add("hidden");
    if (els.timelineRail) els.timelineRail.innerHTML = "";
    if (els.timelineTicks) els.timelineTicks.innerHTML = "";
    els.detectFrameMore?.classList.add("hidden");

    if (detectionRunning) {
      // Detection still running: only the accumulated display was cleared.
      showDetectProgressFromState(lastPipelineState);
    } else {
      // Idle: don't strand the "탐지 시작" button — make sure it is usable again.
      handledDetectJobId = null;
      activeDetectJobId = null;
      endDetectSession();
      setStatus(els.detectStatus, "탐지 결과가 초기화되었습니다.", "ok");
      updateDetectButtonState();
    }
  } catch (err) {
    setStatus(els.detectStatus, err.message, "error");
  } finally {
    resetTimelineActive = false;
    if (els.resetTimelineBtn) {
      els.resetTimelineBtn.disabled = false;
      els.resetTimelineBtn.removeAttribute("aria-disabled");
    }
  }
}

async function compactTimeline() {
  if (!els.compactTimelineBtn) return;
  const detectionRunning = isDetectLive(lastPipelineState) || isDetectBusy(lastPipelineState);
  if (detectionRunning && !confirm("탐지가 진행 중입니다. 현재까지 누적된 결과에 후처리를 적용할까요?")) {
    return;
  }

  els.compactTimelineBtn.disabled = true;
  if (els.compactTimelineStatus) els.compactTimelineStatus.textContent = "후처리 중...";
  try {
    const body = {
      max_gap_sec: 8,
      merge_segments: !!els.postMergeSegments?.checked,
      remove_position_outliers: !!els.postRemovePositionOutliers?.checked,
      remove_size_outliers: !!els.postRemoveSizeOutliers?.checked,
      remove_tall_thin_boxes: !!els.postRemoveTallThinBoxes?.checked,
      remove_static_short_tracks: !!els.postRemoveStaticShortTracks?.checked,
      remove_temporal_isolated: !!els.postRemoveTemporalIsolated?.checked,
      remove_color_outliers: !!els.postRemoveColorOutliers?.checked,
    };
    const result = await api("/api/pipeline/detect/timeline/compact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const timeline = result.timeline || {};
    await loadTimeline({ reset: true });
    const { state } = await api("/api/pipeline/state");
    lastPipelineState = state;
    currentLoadedResultId = state.loaded_saved_result_id || currentLoadedResultId;
    renderPipelineState(state);
    const before = timeline.before_segment_count ?? 0;
    const after = timeline.segment_count ?? 0;
    const merged = timeline.merged_segment_count ?? Math.max(0, before - after);
    const removed = timeline.removed_detection_count ?? 0;
    const removedEvents = timeline.removed_event_count ?? 0;
    if (els.compactTimelineStatus) {
      els.compactTimelineStatus.textContent =
        `후처리 완료 · 구간 ${before}개 -> ${after}개 · ${merged}개 병합 · 탐지 ${removed}개 제거 · 프레임 ${removedEvents}개 제거`;
    }
  } catch (err) {
    if (els.compactTimelineStatus) els.compactTimelineStatus.textContent = err.message;
  } finally {
    els.compactTimelineBtn.disabled = false;
  }
}

function savedResultLabel(item) {
  const name = item.name || item.id;
  const savedAt = item.saved_at ? item.saved_at.slice(0, 19).replace("T", " ") : "";
  const segments = item.segment_count ?? 0;
  const videos = item.video_count ?? 0;
  const suffix = savedAt ? ` · ${savedAt}` : "";
  return `${name} · ${segments}구간 · ${videos}영상${suffix}`;
}

async function loadSavedResultsList() {
  if (!els.savedResultSelect) return;
  try {
    const { results } = await api("/api/pipeline/detect/results");
    els.savedResultSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = results?.length ? "저장 결과 선택" : "저장된 결과 없음";
    els.savedResultSelect.appendChild(placeholder);
    for (const item of results || []) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = savedResultLabel(item);
      els.savedResultSelect.appendChild(option);
    }
    if (currentLoadedResultId) {
      els.savedResultSelect.value = currentLoadedResultId;
    }
  } catch (err) {
    if (els.savedResultStatus) els.savedResultStatus.textContent = err.message;
  }
}

async function saveCurrentDetectionResults() {
  if (!els.saveResultBtn) return;
  const name = els.saveResultName?.value.trim() || "";
  if (!name) {
    if (els.savedResultStatus) els.savedResultStatus.textContent = "저장 이름을 입력하세요.";
    els.saveResultName?.focus();
    return;
  }
  els.saveResultBtn.disabled = true;
  if (els.savedResultStatus) els.savedResultStatus.textContent = "저장 중...";
  try {
    const { result } = await api("/api/pipeline/detect/results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (els.savedResultStatus) {
      els.savedResultStatus.textContent = `저장 완료 · ${result.segment_count ?? 0}구간`;
    }
    if (els.saveResultName) els.saveResultName.value = "";
    await loadSavedResultsList();
    if (els.savedResultSelect) els.savedResultSelect.value = result.id;
    currentLoadedResultId = null;
  } catch (err) {
    if (els.savedResultStatus) els.savedResultStatus.textContent = err.message;
  } finally {
    els.saveResultBtn.disabled = false;
  }
}

async function loadSelectedDetectionResults() {
  const resultId = els.savedResultSelect?.value || "";
  if (!resultId) {
    if (els.savedResultStatus) els.savedResultStatus.textContent = "열 저장 결과를 선택하세요.";
    return;
  }
  const selectedText = els.savedResultSelect.options[els.savedResultSelect.selectedIndex]?.textContent || resultId;
  if (!confirm(`현재 탐지 결과를 선택한 저장 결과로 바꿉니다.\n\n${selectedText}\n\n계속할까요?`)) {
    return;
  }
  if (els.loadResultBtn) els.loadResultBtn.disabled = true;
  if (els.savedResultStatus) els.savedResultStatus.textContent = "불러오는 중...";
  try {
    const { result } = await api(`/api/pipeline/detect/results/${encodeURIComponent(resultId)}/load`, {
      method: "POST",
    });
    currentLoadedResultId = result.id || resultId;
    detectFrameOffset = 0;
    detectFrameTotal = 0;
    detectFrameJobId = null;
    activeDetectJobId = null;
    handledDetectJobId = null;
    endDetectSession();
    await loadTimeline({ reset: true });
    const { state } = await api("/api/pipeline/state");
    lastPipelineState = state;
    renderPipelineState(state);
    if (els.savedResultStatus) {
      els.savedResultStatus.textContent = `불러오기 완료 · ${result.name || result.id} · ${result.segment_count ?? 0}구간`;
    }
  } catch (err) {
    if (els.savedResultStatus) els.savedResultStatus.textContent = err.message;
  } finally {
    if (els.loadResultBtn) els.loadResultBtn.disabled = false;
  }
}

function renderReportLinks(files) {
  if (!els.reportLinks) return;
  const labels = { html: "HTML", csv: "CSV", json: "JSON" };
  els.reportLinks.innerHTML = Object.entries(files || {})
    .map(([kind, filename]) => {
      const href = `/api/pipeline/detect/report/${encodeURIComponent(filename)}`;
      const target = kind === "html" ? ' target="_blank" rel="noreferrer"' : "";
      return `<a href="${href}"${target}>${labels[kind] || kind}</a>`;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function reportFileLinks(files, reportId = "") {
  const labels = { html: "HTML", csv: "CSV", json: "JSON" };
  const links = Object.entries(files || {})
    .map(([kind, filename]) => {
      const href = `/api/pipeline/detect/report/${encodeURIComponent(filename)}`;
      const target = kind === "html" ? ' target="_blank" rel="noreferrer"' : "";
      return `<a href="${href}"${target}>${labels[kind] || escapeHtml(kind)}</a>`;
    });
  if (reportId) {
    const deleteHref = `/api/pipeline/detect/reports/${encodeURIComponent(reportId)}/delete/redirect`;
    links.push(`<a class="danger-link" href="${deleteHref}">Delete</a>`);
  }
  return links.join("");
}

function renderReportList(reports) {
  if (!els.reportList) return;
  if (!reports?.length) {
    els.reportList.innerHTML = '<div class="report-empty hint">생성된 리포트가 없습니다.</div>';
    return;
  }
  els.reportList.innerHTML = reports
    .map((report) => {
      const created = report.created_at
        ? report.created_at.slice(0, 19).replace("T", " ")
        : report.id;
      const source = report.source_summary || "-";
      const models = report.model_summary || "-";
      const confidence = report.confidence_summary || "-";
      const segments = report.segment_count ?? "-";
      const frames = report.detection_frame_count ?? "-";
      const videos = report.video_count ?? "-";
      return `
        <div class="report-item">
          <div class="report-item-main">
            <div class="report-item-title">${escapeHtml(created)}</div>
            <div class="report-item-meta">소스 ${escapeHtml(source)} · 모델 ${escapeHtml(models)} · Confidence ${escapeHtml(confidence)}</div>
            <div class="report-item-meta">구간 ${escapeHtml(segments)} · 탐지 프레임 ${escapeHtml(frames)} · 영상 ${escapeHtml(videos)}</div>
          </div>
          <div class="report-item-actions">
            <div class="report-links">${reportFileLinks(report.files, report.id)}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

async function loadReportList() {
  if (!els.reportList) return;
  try {
    const { reports } = await api("/api/pipeline/detect/reports");
    renderReportList(reports || []);
  } catch (err) {
    els.reportList.innerHTML = `<div class="report-empty hint">${escapeHtml(err.message)}</div>`;
  }
}

async function generateDetectionReport() {
  if (!els.generateReportBtn) return;
  els.generateReportBtn.disabled = true;
  if (els.reportStatus) els.reportStatus.textContent = "문서 생성 중...";
  if (els.reportLinks) els.reportLinks.innerHTML = "";
  try {
    const savedResultId = currentLoadedResultId || lastPipelineState?.loaded_saved_result_id || null;
    const body = savedResultId ? { saved_result_id: savedResultId } : {};
    const result = await api("/api/pipeline/detect/report/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderReportLinks(result.files);
    await loadReportList();
    const count = result.report?.segment_count ?? 0;
    const source = result.source?.type === "saved_result" ? " · 열린 저장 결과 기준" : "";
    if (els.reportStatus) els.reportStatus.textContent = `생성 완료 · ${count}개 구간${source}`;
  } catch (err) {
    if (els.reportStatus) els.reportStatus.textContent = err.message;
  } finally {
    els.generateReportBtn.disabled = false;
  }
}

async function finalizeDetectJob(job) {
  activeDetectJobId = null;
  endDetectSession();
  showDetectComplete(job, lastPipelineState);
  await loadTimeline({ reset: true });
  handledDetectJobId = job.job_id;
}

async function updateDetectUI(jobId) {
  if (!jobId) return null;
  try {
    const { job } = await api(`/api/pipeline/detect/${jobId}`);

    if (job.status === "cancelled") {
      // A cancelled job means the whole detection was stopped (the queue is
      // cleared on cancel). Only stay "running" if the backend genuinely still
      // reports an active session — never just because activeDetectJobId is set,
      // otherwise we could never clear it and the button stays disabled forever.
      if (isDetectActive(lastPipelineState)) {
        showDetectProgressFromState(lastPipelineState);
        return "running";
      }
      activeDetectJobId = null;
      detectStopRequested = false;
      endDetectSession();
      setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
      setStatus(els.detectStatus, job.error || "탐지가 중지되었습니다.", "error");
      updateDetectButtonState();
      return "done";
    }

    if (job.status === "running") {
      if (isDetectStopping()) {
        setStatus(els.detectStatus, "탐지 중지 중...");
        return "cancelling";
      }
      showDetectProgress(
        job.processed_frames || 0,
        job.total_frames || 0,
        job.frames_with_detections || 0,
        job.progress || 0,
        lastPipelineState,
      );
      return "running";
    }

    if (job.status === "completed") {
      detectStopRequested = false;
      const { state: fresh } = await api("/api/pipeline/state");
      lastPipelineState = fresh;
      if (isDetectBatchContinuing(fresh, job.job_id)) {
        await loadTimeline({ reset: true });
        const currentVideo = fresh.detect_video_name ? ` · 현재 ${fresh.detect_video_name}` : "";
        setStatus(
          els.detectStatus,
          `비디오 완료: ${job.video_name} · 다음 파일 처리 중...${currentVideo}`,
        );
        activeDetectJobId = fresh.detect_job_id || null;
        return "running";
      }
      await finalizeDetectJob(job);
      renderPipelineState(fresh);
      return "done";
    }

    if (job.status === "error") {
      detectStopRequested = false;
      const { state: fresh } = await api("/api/pipeline/state");
      lastPipelineState = fresh;
      if (isDetectBatchContinuing(fresh, job.job_id)) {
        const currentVideo = fresh.detect_video_name ? ` · 현재 ${fresh.detect_video_name}` : "";
        setStatus(
          els.detectStatus,
          `비디오 실패: ${job.video_name} · 다음 파일 처리 중...${currentVideo}`,
        );
        activeDetectJobId = fresh.detect_job_id || null;
        return "running";
      }
      activeDetectJobId = null;
      endDetectSession();
      setStatus(els.detectStatus, job.error || "탐지 실패", "error");
      setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
      els.detectBtn.disabled = false;
      updateDetectButtonState();
      return "done";
    }
  } catch (err) {
    console.error("updateDetectUI failed", err);
  }
  return null;
}

async function poll() {
  try {
    const { state } = await api("/api/pipeline/state");
    lastPipelineState = state;
    if (state.detect_status === "running" && state.detect_job_id) {
      activeDetectJobId = state.detect_job_id;
    }
    renderPipelineState(state);

    if (isTrainActive(state)) {
      updateStopButtons(state);
      return;
    }

    if (state.detect_status === "cancelling") {
      updateStopButtons(state);
      return;
    }

    if (
      (state.train_status === "cancelled" || state.detect_status === "cancelled") &&
      !isDetectActive(state) &&
      !activeDetectJobId
    ) {
      detectStopRequested = false;
      endDetectSession();
      updateDetectButtonState();
      stopPolling();
      return;
    }

    const jobId =
      activeDetectJobId ||
      (state.detect_status === "running" || state.detect_status === "cancelling"
        ? state.detect_job_id
        : null);
    const batchRunning =
      !isDetectStopping(state) &&
      (state.detect_status === "running" ||
        (state.detect_queue_pending || 0) > 0 ||
        ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
          state.detect_status !== "error" &&
          state.detect_status !== "cancelled"));
    const needsDetectUI =
      isDetectActive(state) ||
      !!activeDetectJobId ||
      (detectSessionActive && !handledDetectJobId) ||
      (state.detect_status === "completed" &&
        state.detect_job_id &&
        handledDetectJobId !== state.detect_job_id &&
        !detectSessionActive);

    if (jobId && needsDetectUI) {
      const outcome = await updateDetectUI(jobId);
      if (outcome === "done" && !batchRunning) {
        stopPolling();
        return;
      }
    } else if (isDetectBusy(state) && state.detect_status === "running") {
      showDetectProgressFromState(state);
    }

    if (!isDetectBusy(state) && !batchRunning) {
      stopPolling();
    }
  } catch (err) {
    console.error("poll failed", err);
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(poll, POLL_MS);
  poll();
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function needsPolling(state) {
  return isTrainActive(state) || isDetectBusy(state);
}

els.cvatZip.addEventListener("change", () => {
  previewCvatFile(els.cvatZip.files[0]);
});

els.importFrameSplit.addEventListener("change", () => {
  importFramesLoaded = false;
  if (importPanelExpanded) {
    loadImportFrames({ reset: true });
  }
});

els.importFrameToggle.addEventListener("click", () => {
  setImportPanelExpanded(!importPanelExpanded);
});

els.importFrameMore.addEventListener("click", () => {
  loadImportFrames();
});

els.detectFrameMore.addEventListener("click", () => {
  loadTimeline({ reset: false });
});

els.saveResultBtn?.addEventListener("click", saveCurrentDetectionResults);
els.loadResultBtn?.addEventListener("click", loadSelectedDetectionResults);
els.compactTimelineBtn?.addEventListener("click", compactTimeline);
els.generateReportBtn?.addEventListener("click", generateDetectionReport);
els.refreshReportsBtn?.addEventListener("click", loadReportList);

els.refreshModelsBtn?.addEventListener("click", () => {
  modelListKey = "";
  loadModels();
});

els.modelList?.addEventListener("click", handleModelAction);
els.modelFrameMore?.addEventListener("click", renderMoreModelFrames);

els.frameResults.addEventListener("click", handleTimelineFrameClick);
els.frameResults.addEventListener("dblclick", handleFrameCardDblClick);
els.timelineViewport?.addEventListener("click", handleTimelineFrameClick);
els.timelineViewport?.addEventListener("dblclick", handleFrameCardDblClick);
els.importFrameResults.addEventListener("dblclick", handleFrameCardDblClick);
els.modelFrameResults?.addEventListener("dblclick", handleFrameCardDblClick);
els.frameLightboxGallery?.addEventListener("click", (event) => {
  const item = event.target.closest(".lightbox-gallery-item");
  if (!item) return;
  showLightboxFrame(Number(item.dataset.frameIndex) || 0);
});

els.timelineZoomIn?.addEventListener("click", () => {
  setTimelineZoom(timelineZoom * TIMELINE_ZOOM_STEP);
});
els.timelineZoomOut?.addEventListener("click", () => {
  setTimelineZoom(timelineZoom / TIMELINE_ZOOM_STEP);
});
els.timelineZoomReset?.addEventListener("click", () => {
  timelineZoom = 1;
  renderTimelineAxis(timelineRange, timelineAxisSegments);
  if (els.timelineViewport) els.timelineViewport.scrollLeft = 0;
});
els.timelineViewport?.addEventListener(
  "wheel",
  (event) => {
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1 / TIMELINE_ZOOM_STEP : TIMELINE_ZOOM_STEP;
    setTimelineZoom(timelineZoom * factor);
  },
  { passive: false },
);
window.addEventListener("resize", () => {
  if (timelineAxisSegments.length) {
    renderTimelineAxis(timelineRange, timelineAxisSegments);
  }
});

els.frameLightboxClose?.addEventListener("click", closeFrameLightbox);
els.frameLightboxBack?.addEventListener("click", () => {
  if (lightboxSegment && lightboxFrames.length) {
    openLightboxGallery(lightboxSegment, lightboxFrames);
  }
});
els.frameLightboxZoomIn?.addEventListener("click", () => {
  setLightboxViewerZoom(lightboxViewerZoom * FRAME_VIEWER_ZOOM_STEP);
});
els.frameLightboxZoomOut?.addEventListener("click", () => {
  setLightboxViewerZoom(lightboxViewerZoom / FRAME_VIEWER_ZOOM_STEP);
});
els.frameLightboxZoomReset?.addEventListener("click", resetLightboxViewerTransform);
els.frameLightboxPrev?.addEventListener("click", () => showLightboxFrame(lightboxFrameIndex - 1));
els.frameLightboxNext?.addEventListener("click", () => showLightboxFrame(lightboxFrameIndex + 1));
els.frameLightboxViewport?.addEventListener("wheel", handleLightboxViewportWheel, { passive: false });
els.frameLightboxViewport?.addEventListener("pointerdown", handleLightboxViewportPointerDown);
els.frameLightboxViewport?.addEventListener("pointermove", handleLightboxViewportPointerMove);
els.frameLightboxViewport?.addEventListener("pointerup", handleLightboxViewportPointerUp);
els.frameLightboxViewport?.addEventListener("pointercancel", handleLightboxViewportPointerUp);
els.frameLightbox?.addEventListener("click", (event) => {
  if (event.target === els.frameLightbox) closeFrameLightbox();
});
document.addEventListener("keydown", (event) => {
  if (els.frameLightbox?.classList.contains("hidden")) return;
  if (event.key === "Escape") {
    closeFrameLightbox();
    return;
  }
  if (els.frameLightboxViewer?.classList.contains("hidden")) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    showLightboxFrame(lightboxFrameIndex - 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    showLightboxFrame(lightboxFrameIndex + 1);
  } else if (event.key === "+" || event.key === "=") {
    setLightboxViewerZoom(lightboxViewerZoom * FRAME_VIEWER_ZOOM_STEP);
  } else if (event.key === "-") {
    setLightboxViewerZoom(lightboxViewerZoom / FRAME_VIEWER_ZOOM_STEP);
  } else if (event.key === "0") {
    resetLightboxViewerTransform();
  }
});

els.importBtn.addEventListener("click", async () => {
  const file = els.cvatZip.files[0];
  if (!file) return alert("CVAT annotation 파일(.zip 또는 .xml)을 선택하세요.");
  const video = els.cvatVideo.files[0];
  const form = new FormData();
  form.append("file", file);
  if (video) form.append("video", video);
  setStatus(els.importStatus, "Import 중...");
  try {
    const result = await api("/api/pipeline/import-cvat", { method: "POST", body: form });
    setStatus(
      els.importStatus,
      `Import 완료 · train ${result.train_images} / val ${result.val_images} · ${result.task_type}`,
      "ok",
    );
    if (result.label_summary) {
      renderLabelSummary(result.label_summary);
    }
    showImportFramePanel(result);
    importFramesLoaded = false;
    if (importPanelExpanded) {
      await loadImportFrames({ reset: true });
    }
    els.trainBtn.disabled = false;
  } catch (err) {
    setStatus(els.importStatus, err.message, "error");
  }
});

els.trainBtn.addEventListener("click", async () => {
  setStatus(els.trainStatus, "학습 시작...");
  els.trainBtn.disabled = true;
  try {
    await api("/api/pipeline/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: Number(els.epochs.value),
        batch: Number(els.batch.value),
        imgsz: Number(els.trainImgsz?.value || 416),
      }),
    });
    updateStopButtons({ ...lastPipelineState, train_status: "running" });
    startPolling();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
    els.trainBtn.disabled = false;
  }
});

function trainRequestBody() {
  return {
    epochs: Number(els.epochs.value),
    batch: Number(els.batch.value),
    imgsz: Number(els.trainImgsz?.value || 416),
  };
}

els.resetTrainBtn?.addEventListener("click", async () => {
  if (
    !confirm(
      "학습된 모델 파일과 YOLO 학습 기록을 삭제할까요?\n다음 학습은 YOLO 기본 가중치(yolo11n)부터 새로 시작합니다.",
    )
  ) {
    return;
  }
  els.resetTrainBtn.disabled = true;
  els.resetAndTrainBtn.disabled = true;
  try {
    const result = await api("/api/pipeline/train/reset", { method: "POST" });
    const count = result.deleted?.length || 0;
    lastPipelineState = {
      ...(lastPipelineState || {}),
      train_status: "idle",
      train_weights: null,
      train_error: null,
      train_progress: null,
      train_epoch: 0,
      train_epochs: 0,
      train_batch: 0,
      train_batches: 0,
      train_progress_pct: 0,
    };
    renderPipelineState(lastPipelineState);
    setStatus(els.trainStatus, `모델 초기화 완료 · 삭제 ${count}개`, "ok");
    updateDetectButtonState();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
  } finally {
    els.resetTrainBtn.disabled = false;
    els.resetAndTrainBtn.disabled = false;
  }
});

els.resetAndTrainBtn?.addEventListener("click", async () => {
  if (
    !confirm(
      "학습된 모델을 삭제하고 YOLO 기본 가중치부터 새로 학습을 시작할까요?",
    )
  ) {
    return;
  }
  els.resetAndTrainBtn.disabled = true;
  els.resetTrainBtn.disabled = true;
  els.trainBtn.disabled = true;
  setStatus(els.trainStatus, "모델 초기화 후 학습 시작...");
  setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, 0, "준비 중...");
  try {
    await api("/api/pipeline/train/reset-and-start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(trainRequestBody()),
    });
    lastPipelineState = {
      ...(lastPipelineState || {}),
      train_status: "running",
      train_weights: null,
      train_error: null,
    };
    updateStopButtons(lastPipelineState);
    updateDetectButtonState();
    startPolling();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
    els.trainBtn.disabled = lastPipelineState?.import_status === "completed";
  } finally {
    els.resetAndTrainBtn.disabled = false;
    els.resetTrainBtn.disabled = false;
  }
});

els.stopTrainBtn?.addEventListener("click", async () => {
  els.stopTrainBtn.disabled = true;
  setStatus(els.trainStatus, "학습 중지 요청...");
  try {
    const result = await api("/api/pipeline/train/stop", { method: "POST" });
    if (result.cancelled) {
      lastPipelineState = {
        ...(lastPipelineState || {}),
        train_status: result.reason === "stale_recovered" ? "cancelled" : "running",
        train_progress: result.reason === "stale_recovered" ? "cancelled" : "cancelling",
        train_error:
          result.reason === "stale_recovered"
            ? "학습 세션이 끊겨 중지됨 (서버 재시작 등)"
            : lastPipelineState?.train_error,
      };
      if (result.reason === "stale_recovered") {
        renderPipelineState(lastPipelineState);
      }
      updateStopButtons(lastPipelineState);
    } else {
      setStatus(els.trainStatus, "진행 중인 학습이 없습니다.", "error");
      updateStopButtons();
    }
    startPolling();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
  } finally {
    els.stopTrainBtn.disabled = false;
  }
});

els.stopDetectBtn?.addEventListener("click", async () => {
  if (detectStopRequested || isDetectStopping()) {
    return;
  }
  detectStopRequested = true;
  els.stopDetectBtn.disabled = true;
  setStatus(els.detectStatus, "탐지 중지 요청...");
  try {
    const result = await api("/api/pipeline/detect/stop", { method: "POST" });
    if (!result.cancelled) {
      detectStopRequested = false;
      if (lastPipelineState?.detect_status === "cancelled") {
        setStatus(els.detectStatus, "이미 중지되었습니다.", "error");
      }
      updateStopButtons();
      return;
    }
    lastPipelineState = {
      ...(lastPipelineState || {}),
      detect_status: "cancelling",
      detect_queue_pending: 0,
      detect_error: "중지 요청됨",
    };
    updateStopButtons(lastPipelineState);
    if (!pollTimer) {
      startPolling();
    }
  } catch (err) {
    detectStopRequested = false;
    setStatus(els.detectStatus, err.message, "error");
    updateStopButtons();
  }
});

els.lakeDiscoverBtn?.addEventListener("click", discoverLakeVideos);

[els.lakeMedia, els.lakeYearFolder, els.lakeVessel, els.lakeStream].forEach((selectEl) => {
  selectEl?.addEventListener("change", () => {
    lakeVideosReady = 0;
    setStatus(els.lakeDiscoverStatus, "");
    updateLakeUrlPreview();
    updateDetectButtonState();
  });
});

[els.lakeMinuteOffsets, els.lakeSecondSuffixes].forEach((inputEl) => {
  inputEl?.addEventListener("input", () => {
    lakeVideosReady = 0;
    setStatus(els.lakeDiscoverStatus, "");
    updateLakeUrlPreview();
    updateDetectButtonState();
  });
});

document.querySelectorAll('input[name="detectSource"]').forEach((input) => {
  input.addEventListener("change", () => {
    lakeVideosReady = 0;
    setDetectSourceMode(getDetectSourceMode());
    if (detectSourceMode === "lake") {
      setStatus(els.lakeDiscoverStatus, "");
    }
  });
});

els.detectVideo?.addEventListener("change", updateDetectButtonState);
els.streamUrl?.addEventListener("change", () => {
  streamPreviewUrl = "";
  updateStreamPreview();
  updateDetectButtonState();
});
els.streamUrl?.addEventListener("input", updateDetectButtonState);
els.streamPreview?.addEventListener("error", () => scheduleStreamPreviewRestart("video error"));
els.streamPreview?.addEventListener("emptied", () => {
  if (detectSourceMode === "stream") scheduleStreamPreviewRestart("stream emptied");
});
els.streamPreview?.addEventListener("stalled", () => {
  if (detectSourceMode === "stream") scheduleStreamPreviewRestart("stalled");
});
els.streamPreview?.addEventListener("waiting", () => {
  if (detectSourceMode !== "stream") return;
  window.setTimeout(() => {
    const video = els.streamPreview;
    if (!video || !video.paused || video.readyState >= 3) return;
    scheduleStreamPreviewRestart("waiting");
  }, 2500);
});
els.streamPreview?.addEventListener("playing", () => {
  streamRestartCount = 0;
  if (streamRestartTimer) {
    clearTimeout(streamRestartTimer);
    streamRestartTimer = null;
  }
});

els.detectBtn.addEventListener("click", async () => {
  detectSourceMode = getDetectSourceMode();
  els.detectBtn.disabled = true;
  const modelIds = selectedDetectionModelIds();
  if (!modelIds.length) {
    alert("탐지에 사용할 모델을 하나 이상 선택하세요.");
    updateDetectButtonState();
    return;
  }

  if (detectSourceMode === "lake") {
    if (lakeVideosReady <= 0) {
      const discovered = await discoverLakeVideos();
      if (!discovered?.found_count) {
        els.detectBtn.disabled = false;
        return;
      }
    }
    const range = readLakeRange();
    const batchTotal = lakeVideosReady;
    beginDetectSession(batchTotal);
    setStatus(els.detectStatus, `영상 다운로드·탐지 준비 중... (${batchTotal}개)`);
    try {
      const result = await api("/api/pipeline/detect/lake", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...range,
          frame_stride: Number(els.frameStride.value),
          confidence: Number(els.confidence.value),
          imgsz: Number(els.detectImgsz?.value || 416),
          use_sam: !!els.useSam?.checked,
          skip_dark_video: !!els.skipDarkVideo?.checked,
          check_exists: true,
          model_ids: modelIds,
        }),
      });
      await finishDetectStart(result, batchTotal);
    } catch (err) {
      activeDetectJobId = null;
      endDetectSession();
      stopPolling();
      setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
      setStatus(els.detectStatus, err.message, "error");
      updateDetectButtonState();
    }
    return;
  }

  if (detectSourceMode === "stream") {
    const streamUrl = (els.streamUrl?.value || "").trim();
    if (!streamUrl) {
      alert("스트림 주소를 입력하세요.");
      els.detectBtn.disabled = false;
      return;
    }
    updateStreamPreview();
    beginDetectSession(1, 0);
    setStatus(els.detectStatus, "실시간 스트림 탐지 시작 중...");
    try {
      const result = await api("/api/pipeline/detect/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stream_url: streamUrl,
          frame_stride: Number(els.frameStride.value),
          confidence: Number(els.confidence.value),
          imgsz: Number(els.detectImgsz?.value || 416),
          use_sam: !!els.useSam?.checked,
          model_ids: modelIds,
        }),
      });
      await finishDetectStart(result, 1);
    } catch (err) {
      activeDetectJobId = null;
      endDetectSession();
      stopPolling();
      setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
      setStatus(els.detectStatus, err.message, "error");
      updateDetectButtonState();
    }
    return;
  }

  const files = els.detectVideo.files;
  if (!files.length) {
    alert("비디오 파일을 하나 이상 선택하세요.");
    els.detectBtn.disabled = false;
    return;
  }
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  form.append("frame_stride", els.frameStride.value);
  form.append("confidence", els.confidence.value);
  form.append("imgsz", els.detectImgsz?.value || "416");
  form.append("use_sam", els.useSam?.checked ? "true" : "false");
  form.append("skip_dark_video", els.skipDarkVideo?.checked ? "true" : "false");
  for (const modelId of modelIds) {
    form.append("model_ids", modelId);
  }
  beginDetectSession(files.length);
  setStatus(els.detectStatus, `비디오 업로드 중... (${files.length}개)`);
  try {
    const result = await api("/api/pipeline/detect", { method: "POST", body: form });
    await finishDetectStart(result, files.length);
  } catch (err) {
    activeDetectJobId = null;
    endDetectSession();
    stopPolling();
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, err.message, "error");
    updateDetectButtonState();
  }
});

api("/api/pipeline/state")
  .then(({ state }) => {
    const meta = state.dataset_meta;
    if (meta?.label_summary) {
      renderLabelSummary(meta.label_summary);
    } else if (meta?.class_names?.length) {
      renderLabelSummary({
        objects: meta.class_names.map((name) => ({
          name,
          annotation_count: 0,
          box_count: 0,
          polygon_count: 0,
          frame_count: 0,
        })),
        total_annotations: meta.total_annotations || 0,
        annotated_frames: 0,
        task_type: meta.task_type,
      });
    }
    if (state.import_status === "completed" && meta) {
      showImportFramePanel(meta);
    }
    renderPipelineState(state);
    lastPipelineState = state;
    renderModels(state.models, state.active_model_id, state.default_detection_model_ids);
    const timelineCount =
      state.detect_timeline?.segment_count ??
      state.detect_timeline_events ??
      0;
    if (timelineCount > 0) {
      loadTimeline({ reset: true });
    } else if (state.detect_status === "completed" && state.detect_job_id) {
      handledDetectJobId = state.detect_job_id;
    }
    if (state.detect_job_id && (state.detect_status === "running" || state.detect_queue_pending > 0)) {
      activeDetectJobId = state.detect_job_id;
      startPolling();
    } else if (needsPolling(state) || activeDetectJobId) {
      startPolling();
    }
  })
  .catch((err) => {
    console.error(err);
    loadModels();
  });

loadSavedResultsList();
loadReportList();
if (els.resetTimelineBtn) {
  els.resetTimelineBtn.disabled = false;
  els.resetTimelineBtn.removeAttribute("aria-disabled");
}
setDetectSourceMode("upload");
loadLakeConfig();
