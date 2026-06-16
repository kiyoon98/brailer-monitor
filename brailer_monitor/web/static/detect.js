const els = {
  cvatZip: document.getElementById("cvatZip"),
  cvatVideo: document.getElementById("cvatVideo"),
  importBtn: document.getElementById("importBtn"),
  importStatus: document.getElementById("importStatus"),
  importLabels: document.getElementById("importLabels"),
  epochs: document.getElementById("epochs"),
  batch: document.getElementById("batch"),
  trainBtn: document.getElementById("trainBtn"),
  resetTrainBtn: document.getElementById("resetTrainBtn"),
  resetAndTrainBtn: document.getElementById("resetAndTrainBtn"),
  stopTrainBtn: document.getElementById("stopTrainBtn"),
  trainStatus: document.getElementById("trainStatus"),
  trainProgressWrap: document.getElementById("trainProgressWrap"),
  trainProgressBar: document.getElementById("trainProgressBar"),
  trainProgressText: document.getElementById("trainProgressText"),
  detectVideo: document.getElementById("detectVideo"),
  detectUploadSection: document.getElementById("detectUploadSection"),
  detectLakeSection: document.getElementById("detectLakeSection"),
  lakeProfile: document.getElementById("lakeProfile"),
  lakeBaseUrl: document.getElementById("lakeBaseUrl"),
  lakeStartMonth: document.getElementById("lakeStartMonth"),
  lakeStartDay: document.getElementById("lakeStartDay"),
  lakeStartHour: document.getElementById("lakeStartHour"),
  lakeEndMonth: document.getElementById("lakeEndMonth"),
  lakeEndDay: document.getElementById("lakeEndDay"),
  lakeEndHour: document.getElementById("lakeEndHour"),
  lakeDiscoverBtn: document.getElementById("lakeDiscoverBtn"),
  lakeDiscoverStatus: document.getElementById("lakeDiscoverStatus"),
  frameStride: document.getElementById("frameStride"),
  confidence: document.getElementById("confidence"),
  detectBtn: document.getElementById("detectBtn"),
  stopDetectBtn: document.getElementById("stopDetectBtn"),
  resetTimelineBtn: document.getElementById("resetTimelineBtn"),
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
const DETECT_FRAME_LIMIT = 200;
const TIMELINE_ZOOM_MIN = 1;
const TIMELINE_ZOOM_MAX = 48;
const TIMELINE_ZOOM_STEP = 1.35;
const FRAME_VIEWER_ZOOM_MIN = 1;
const FRAME_VIEWER_ZOOM_MAX = 8;
const FRAME_VIEWER_ZOOM_STEP = 1.2;

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
let timelineRange = null;
let timelineAxisSegments = [];
let timelineZoom = 1;
let detectSourceMode = "upload";
let lakeVideosReady = 0;
let lakeProfileId = null;
let detectFrameJobId = null;
let lastPipelineState = null;
let importFramesLoaded = false;
let importPanelExpanded = false;
let labelSummaryKey = "";
let detectSession = 0;
let detectSessionActive = false;
let detectStopRequested = false;

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
  const barText = detectProgressText(processed, total, withObjects, pct, state);
  setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, pct, barText);
  setStatus(
    els.detectStatus,
    state.detect_status === "cancelling" ? "탐지 중지 중..." : barText,
  );
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
  return selected?.value === "lake" ? "lake" : "upload";
}

function setDetectSourceMode(mode) {
  detectSourceMode = mode === "lake" ? "lake" : "upload";
  els.detectUploadSection?.classList.toggle("hidden", detectSourceMode === "lake");
  els.detectLakeSection?.classList.toggle("hidden", detectSourceMode !== "lake");
  updateDetectButtonState();
}

function readLakeRange() {
  return {
    profile: lakeProfileId || els.lakeProfile?.value || null,
    start_month: Number(els.lakeStartMonth.value),
    start_day: Number(els.lakeStartDay.value),
    start_hour: Number(els.lakeStartHour.value),
    end_month: Number(els.lakeEndMonth.value),
    end_day: Number(els.lakeEndDay.value),
    end_hour: Number(els.lakeEndHour.value),
  };
}

function updateDetectButtonState() {
  const trainDone = lastPipelineState?.train_status === "completed";
  const detectActive = isDetectBusy();
  if (!trainDone) {
    els.detectBtn.disabled = true;
    if (lastPipelineState?.import_status === "completed" && !detectActive && els.detectStatus) {
      const hint =
        lastPipelineState?.train_status === "running"
          ? "학습 완료 후 탐지할 수 있습니다."
          : "학습된 모델이 없습니다. 학습을 먼저 완료하세요.";
      setStatus(els.detectStatus, hint);
    }
    return;
  }
  if (detectSourceMode === "lake") {
    els.detectBtn.disabled = lakeVideosReady <= 0;
    return;
  }
  els.detectBtn.disabled = !els.detectVideo.files?.length;
}

function formatLakeFilenameHint(config) {
  const minute = String(config.minute_slots?.[0] ?? 0).padStart(2, "0");
  const suffix = config.second_suffixes?.[0] ?? "??";
  return `${config.file_prefix}_YYMMDD_HH${minute}${suffix}.mp4`;
}

async function loadLakeConfig(profile = lakeProfileId) {
  try {
    const query = profile ? `?profile=${encodeURIComponent(profile)}` : "";
    const config = await api(`/api/pipeline/lake-videos/config${query}`);
    lakeProfileId = config.profile || profile || null;
    if (els.lakeProfile && config.profiles?.length) {
      const selected = lakeProfileId || config.profile;
      els.lakeProfile.innerHTML = config.profiles
        .map((item) => {
          const isSelected = item.id === selected || (!selected && item.default);
          return `<option value="${item.id}"${isSelected ? " selected" : ""}>${item.label}</option>`;
        })
        .join("");
      lakeProfileId = els.lakeProfile.value;
    }
    if (els.lakeBaseUrl) {
      els.lakeBaseUrl.textContent = `${config.base_url} · ${formatLakeFilenameHint(config)} · ${config.year}년`;
    }
  } catch (err) {
    console.error("loadLakeConfig failed", err);
  }
}

async function discoverLakeVideos() {
  const range = readLakeRange();
  setStatus(els.lakeDiscoverStatus, "영상 목록 확인 중...");
  els.lakeDiscoverBtn.disabled = true;
  try {
    const result = await api("/api/pipeline/lake-videos/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...range, check_exists: true }),
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
    setStatus(els.lakeDiscoverStatus, err.message, "error");
    updateDetectButtonState();
    return null;
  } finally {
    els.lakeDiscoverBtn.disabled = false;
  }
}

function beginDetectSession(batchTotal, queuePending = Math.max(0, batchTotal - 1)) {
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
    detect_queue_pending: queuePending,
    detect_processed_frames: 0,
    detect_total_frames: 0,
    detect_frames_with_objects: 0,
    detect_progress_pct: 0,
    detect_error: null,
  };
  setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
  setStatus(els.detectStatus, "탐지 준비 중...");
  updateStopButtons(lastPipelineState);
  startPolling();
}

function finishDetectStart(result, batchTotal) {
  const running = result.jobs?.find((job) => job.job_id) || result.job;
  activeDetectJobId = running?.job_id || result.job_id || null;
  lastPipelineState = {
    ...(lastPipelineState || {}),
    detect_status: "running",
    detect_job_id: activeDetectJobId || lastPipelineState?.detect_job_id,
    detect_batch_total: result.batch_total ?? result.batch_size ?? batchTotal,
    detect_batch_done: result.batch_done ?? 0,
    detect_queue_pending: result.queue_pending ?? 0,
  };
  if (activeDetectJobId) {
    showDetectProgress(0, 0, 0, 0, lastPipelineState);
  } else if ((result.queue_pending || 0) > 0) {
    setStatus(els.detectStatus, `대기열 ${result.queue_pending}개 · 첫 비디오 시작 중...`);
  } else {
    showDetectProgressFromState(lastPipelineState);
  }
  updateStopButtons(lastPipelineState);
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

function detectProgressText(processed, total, withObjects, pct, state) {
  const totalLabel = total > 0 ? total : "?";
  let batch = "";
  if (state?.detect_batch_total > 1) {
    const current = Math.min((state.detect_batch_done || 0) + 1, state.detect_batch_total);
    batch = ` · 비디오 ${current}/${state.detect_batch_total}`;
  }
  if (state?.detect_queue_pending > 0) {
    batch += ` · 대기 ${state.detect_queue_pending}`;
  }
  return `탐지 중 ${processed}/${totalLabel} 프레임 · 객체 ${withObjects}개 (${Math.round(pct * 100)}%)${batch}`;
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
  if (key === labelSummaryKey) return;
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
  setProgress(
    els.detectProgressWrap,
    els.detectProgressBar,
    els.detectProgressText,
    pct,
    detectProgressText(processed, total, withObjects, pct, state),
  );
  const detail = detectProgressText(processed, total, withObjects, pct, state);
  setStatus(els.detectStatus, detail);
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
  setStatus(
    els.detectStatus,
    `탐지 완료: 누적 ${timelineCount}건 · 마지막 ${job.video_name}`,
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

function trainStatusText(state) {
  const epoch = state.train_epoch || 0;
  const total = state.train_epochs || Number(els.epochs.value) || 1;
  const pct = Math.round((state.train_progress_pct || 0) * 100);
  const progress = state.train_progress || "";

  if (progress === "cancelling") {
    return {
      bar: `학습 중지 중 · epoch ${epoch}/${total}`,
      status: "학습 중지 요청됨 · 현재 배치가 끝나면 멈춥니다...",
    };
  }
  if (progress === "starting") {
    return { bar: "학습 준비 중...", status: "모델·데이터 로딩 중..." };
  }
  if (progress === "training" && epoch === 0) {
    return { bar: "학습 시작 중...", status: "첫 epoch 진행 중..." };
  }
  if (epoch > 0) {
    return {
      bar: `epoch ${epoch}/${total} (${pct}%)`,
      status: `학습 중 · epoch ${epoch}/${total} · ${pct}%`,
    };
  }
  return { bar: progress || "학습 중...", status: progress || "학습 중..." };
}

function renderPipelineState(state) {
  const meta = state.dataset_meta;

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
    setStatus(els.trainStatus, `학습 완료: ${state.train_weights}`, "ok");
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
  } else if (
    detectSessionActive &&
    !activeDetectJobId &&
    (state.detect_status === "completed" || state.detect_status === "cancelled")
  ) {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, 0, "준비 중...");
    setStatus(els.detectStatus, "비디오 업로드·탐지 준비 중...");
    els.detectBtn.disabled = true;
  } else if (state.detect_status === "completed" && !isDetectBusy(state)) {
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
    setStatus(
      els.detectStatus,
      `탐지 완료 · 누적 ${state.detect_timeline_events || 0}건`,
      "ok",
    );
    els.detectBtn.disabled = false;
    detectStopRequested = false;
    endDetectSession();
    updateDetectButtonState();
  } else if (state.detect_status === "error") {
    activeDetectJobId = null;
    endDetectSession();
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, state.detect_error || "탐지 실패", "error");
    els.detectBtn.disabled = false;
    detectStopRequested = false;
    updateDetectButtonState();
  } else if (state.detect_status === "cancelled" && !isDetectBusy(state)) {
    activeDetectJobId = null;
    endDetectSession();
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, state.detect_error || "탐지가 중지되었습니다.", "error");
    detectStopRequested = false;
    updateDetectButtonState();
  }

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
  return `신뢰도 ${conf} · 면적 ${area}`;
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

async function fetchAllTimelineSegments() {
  const allSegments = [];
  let offset = 0;
  while (true) {
    const { timeline } = await api(
      `/api/pipeline/detect/timeline?offset=${offset}&limit=${DETECT_FRAME_LIMIT}`,
    );
    allSegments.push(...(timeline.segments || []));
    const fetched = timeline.segments?.length || 0;
    if (fetched === 0 || offset + fetched >= (timeline.total || 0)) {
      return { timeline, allSegments };
    }
    offset += fetched;
  }
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
      const { timeline, allSegments } = await fetchAllTimelineSegments();
      timelineRange = {
        range_start: timeline.range_start,
        range_end: timeline.range_end,
        range_start_label: timeline.range_start_label,
        range_end_label: timeline.range_end_label,
      };
      timelineAxisSegments = timeline.axis_segments || allSegments;
      detectFrameTotal = timeline.total || 0;
      if (els.frameCount) els.frameCount.textContent = detectFrameTotal;
      renderTimelineAxis(timelineRange, timelineAxisSegments);
      appendTimelineSegments(allSegments);
      detectFrameOffset = allSegments.length;
      els.detectFrameMore?.classList.add("hidden");
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

async function resetTimeline() {
  if (!confirm("누적 탐지 결과를 모두 초기화할까요?")) return;
  await api("/api/pipeline/detect/timeline/reset", { method: "POST" });
  detectFrameOffset = 0;
  detectFrameTotal = 0;
  detectFrameJobId = null;
  handledDetectJobId = null;
  timelineRange = null;
  timelineAxisSegments = [];
  timelineZoom = 1;
  els.frameResults.innerHTML = "";
  if (els.frameCount) els.frameCount.textContent = "0";
  els.timelineAxis?.classList.add("hidden");
  if (els.timelineRail) els.timelineRail.innerHTML = "";
  if (els.timelineTicks) els.timelineTicks.innerHTML = "";
  els.detectFrameMore?.classList.add("hidden");
  setStatus(els.detectStatus, "탐지 결과가 초기화되었습니다.", "ok");
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
  if (isDetectStopping()) {
    setStatus(els.detectStatus, "탐지 중지 중...");
    return "cancelling";
  }
  try {
    const { job } = await api(`/api/pipeline/detect/${jobId}`);

    if (job.status === "cancelled") {
      if (isDetectActive(lastPipelineState) || activeDetectJobId) {
        showDetectProgressFromState(lastPipelineState);
        return "running";
      }
      activeDetectJobId = null;
      detectStopRequested = false;
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
      if (isDetectStopping()) {
        setStatus(els.detectStatus, "탐지 중지 중...");
        return "cancelling";
      }
      const { state: fresh } = await api("/api/pipeline/state");
      lastPipelineState = fresh;
      const batchStillRunning =
        (fresh.detect_queue_pending || 0) > 0 ||
        (fresh.detect_status === "running" && fresh.detect_job_id !== job.job_id) ||
        ((fresh.detect_batch_total || 0) > (fresh.detect_batch_done || 0) &&
          fresh.detect_status !== "error" &&
          fresh.detect_status !== "completed");
      if (batchStillRunning) {
        await loadTimeline({ reset: true });
        setStatus(
          els.detectStatus,
          `비디오 완료: ${job.video_name} · 다음 파일 처리 중...`,
        );
        activeDetectJobId = fresh.detect_job_id || null;
        return "running";
      }
      await finalizeDetectJob(job);
      renderPipelineState(fresh);
      return "done";
    }

    if (job.status === "error") {
      activeDetectJobId = null;
      setStatus(els.detectStatus, job.error || "탐지 실패", "error");
      setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
      els.detectBtn.disabled = false;
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
          state.detect_status !== "error"));
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

els.resetTimelineBtn?.addEventListener("click", resetTimeline);

els.frameResults.addEventListener("click", handleTimelineFrameClick);
els.frameResults.addEventListener("dblclick", handleFrameCardDblClick);
els.timelineViewport?.addEventListener("click", handleTimelineFrameClick);
els.timelineViewport?.addEventListener("dblclick", handleFrameCardDblClick);
els.importFrameResults.addEventListener("dblclick", handleFrameCardDblClick);
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

els.lakeProfile?.addEventListener("change", async () => {
  lakeProfileId = els.lakeProfile.value;
  lakeVideosReady = 0;
  setStatus(els.lakeDiscoverStatus, "");
  await loadLakeConfig(lakeProfileId);
  updateDetectButtonState();
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

els.detectBtn.addEventListener("click", async () => {
  detectSourceMode = getDetectSourceMode();
  els.detectBtn.disabled = true;

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
    try {
      const result = await api("/api/pipeline/detect/lake", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...range,
          frame_stride: Number(els.frameStride.value),
          confidence: Number(els.confidence.value),
          check_exists: true,
        }),
      });
      finishDetectStart(result, batchTotal);
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
  beginDetectSession(files.length);
  setStatus(els.detectStatus, `비디오 업로드 중... (${files.length}개)`);
  try {
    const result = await api("/api/pipeline/detect", { method: "POST", body: form });
    finishDetectStart(result, files.length);
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
  .catch(console.error);

setDetectSourceMode("upload");
loadLakeConfig();
