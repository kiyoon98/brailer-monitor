const els = {
  cvatZip: document.getElementById("cvatZip"),
  cvatVideo: document.getElementById("cvatVideo"),
  importBtn: document.getElementById("importBtn"),
  importStatus: document.getElementById("importStatus"),
  importLabels: document.getElementById("importLabels"),
  epochs: document.getElementById("epochs"),
  batch: document.getElementById("batch"),
  trainBtn: document.getElementById("trainBtn"),
  stopTrainBtn: document.getElementById("stopTrainBtn"),
  trainStatus: document.getElementById("trainStatus"),
  trainProgressWrap: document.getElementById("trainProgressWrap"),
  trainProgressBar: document.getElementById("trainProgressBar"),
  trainProgressText: document.getElementById("trainProgressText"),
  detectVideo: document.getElementById("detectVideo"),
  detectUploadSection: document.getElementById("detectUploadSection"),
  detectLakeSection: document.getElementById("detectLakeSection"),
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
  frameLightboxImg: document.getElementById("frameLightboxImg"),
  frameLightboxMeta: document.getElementById("frameLightboxMeta"),
  frameLightboxClose: document.getElementById("frameLightboxClose"),
};

const POLL_MS = 500;
const IMPORT_FRAME_LIMIT = 48;
const DETECT_FRAME_LIMIT = 200;
const TIMELINE_ZOOM_MIN = 1;
const TIMELINE_ZOOM_MAX = 48;
const TIMELINE_ZOOM_STEP = 1.35;

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
let detectFrameJobId = null;
let lastPipelineState = null;
let importFramesLoaded = false;
let importPanelExpanded = false;
let labelSummaryKey = "";
let detectSession = 0;
let detectStopRequested = false;

function isDetectStopping(state = lastPipelineState) {
  return state?.detect_status === "cancelling" || state?.detect_status === "cancelled";
}

function updateStopButtons(state = lastPipelineState) {
  const trainRunning = state?.train_status === "running";
  const detectRunning =
    state?.detect_status === "running" || state?.detect_status === "cancelling";

  els.stopTrainBtn?.classList.toggle("hidden", !trainRunning);
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
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
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
  if (!trainDone) {
    els.detectBtn.disabled = true;
    return;
  }
  if (detectSourceMode === "lake") {
    els.detectBtn.disabled = lakeVideosReady <= 0;
    return;
  }
  els.detectBtn.disabled = !els.detectVideo.files?.length;
}

async function loadLakeConfig() {
  try {
    const config = await api("/api/pipeline/lake-videos/config");
    if (els.lakeBaseUrl) {
      els.lakeBaseUrl.textContent = `${config.base_url} · ${config.file_prefix}_YYMMDD_HHMM${config.minute_slots?.[0] ?? 0}16.mp4 형식`;
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

function startDetectPolling(result, batchTotal) {
  const running = result.jobs?.find((job) => job.job_id) || result.job;
  activeDetectJobId = running?.job_id || null;
  lastPipelineState = {
    ...(lastPipelineState || {}),
    detect_status: "running",
    detect_batch_total: result.batch_total ?? result.batch_size ?? batchTotal,
    detect_batch_done: result.batch_done ?? 0,
    detect_queue_pending: result.queue_pending ?? 0,
  };
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
  setStatus(els.detectStatus, "탐지 진행 중...");
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
    const pct = state.train_progress_pct || 0;
    const epoch = state.train_epoch || 0;
    const total = state.train_epochs || Number(els.epochs.value) || 1;
    setProgress(
      els.trainProgressWrap,
      els.trainProgressBar,
      els.trainProgressText,
      pct,
      `학습 중 epoch ${epoch}/${total} (${Math.round(pct * 100)}%)`,
    );
    const progressLabel =
      state.train_progress === "cancelling" ? "학습 중지 요청됨..." : state.train_progress || "학습 중...";
    setStatus(els.trainStatus, progressLabel);
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
  }

  if (state.detect_status === "cancelled" || state.detect_status === "cancelling") {
    activeDetectJobId = null;
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    const message =
      state.detect_status === "cancelling"
        ? "탐지 중지 중..."
        : state.detect_error || "탐지가 중지되었습니다.";
    setStatus(els.detectStatus, message, state.detect_status === "cancelled" ? "error" : "");
    if (state.detect_status === "cancelled") {
      detectStopRequested = false;
      updateDetectButtonState();
    }
  }

  updateStopButtons(state);
}

function formatConfidence(confidence) {
  if (confidence == null || Number.isNaN(confidence)) return "-";
  return `${(confidence * 100).toFixed(0)}%`;
}

function formatAreaPx(areaPx) {
  const px = Number(areaPx) || 0;
  if (px <= 0) return "0px";
  if (px >= 1_000_000) return `${(px / 1_000_000).toFixed(2)}M px`;
  if (px >= 10_000) return `${(px / 1000).toFixed(1)}k px`;
  return `${px.toLocaleString()} px`;
}

function formatSegmentStats(segmentOrFrame) {
  const conf = formatConfidence(segmentOrFrame.max_confidence ?? segmentOrFrame.confidence);
  const area = formatAreaPx(segmentOrFrame.max_area_px ?? segmentOrFrame.area_px);
  return `신뢰도 ${conf} · 면적 ${area}`;
}

function openFrameLightbox(imgSrc, titleHtml, objectsHtml) {
  if (!imgSrc || !els.frameLightbox) return;
  els.frameLightboxGallery?.classList.add("hidden");
  els.frameLightboxGallery.innerHTML = "";
  els.frameLightboxImg.classList.remove("hidden");
  els.frameLightboxImg.src = imgSrc;
  els.frameLightboxMeta.innerHTML = `
    <div><strong>${titleHtml}</strong></div>
    <div class="objects">${objectsHtml || "(없음)"}</div>`;
  els.frameLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function previewUrl(jobId, previewPath) {
  if (!jobId || !previewPath) return "";
  return `/api/pipeline/detect/${jobId}/previews/${previewPath}`;
}

function openSegmentLightbox(segment, frames) {
  if (!els.frameLightbox || !frames?.length) return;
  els.frameLightboxImg.classList.add("hidden");
  els.frameLightboxImg.removeAttribute("src");
  const gallery = els.frameLightboxGallery;
  gallery.innerHTML = frames
    .map((frame) => {
      const src = previewUrl(frame.job_id, frame.preview_path);
      const time = frame.absolute_time_label || fmtTime(frame.timestamp_sec);
      const stats = formatSegmentStats(frame);
      return `
        <div class="lightbox-gallery-item">
          <img src="${src}" alt="" loading="lazy" />
          <div class="caption">${time} · ${stats}</div>
        </div>`;
    })
    .join("");
  gallery.classList.remove("hidden");
  const range =
    segment.time_label ||
    `${segment.start_absolute_time_label || ""} – ${segment.end_absolute_time_label || ""}`;
  els.frameLightboxMeta.innerHTML = `
    <div><strong>${segment.class_name}</strong> · ${range}</div>
    <div class="objects">${segment.video_name} · ${frames.length}프레임 · ${formatSegmentStats(segment)}</div>`;
  els.frameLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

async function openSegmentLightboxById(segmentId, segmentSummary) {
  try {
    const { segment, frames } = await api(`/api/pipeline/detect/timeline/segment/${encodeURIComponent(segmentId)}`);
    openSegmentLightbox(segment || segmentSummary, frames);
  } catch (err) {
    console.error("openSegmentLightboxById failed", err);
    alert(err.message || "세그먼트를 불러오지 못했습니다.");
  }
}

function closeFrameLightbox() {
  if (!els.frameLightbox) return;
  els.frameLightbox.classList.add("hidden");
  els.frameLightboxImg.classList.remove("hidden");
  els.frameLightboxImg.removeAttribute("src");
  if (els.frameLightboxGallery) {
    els.frameLightboxGallery.classList.add("hidden");
    els.frameLightboxGallery.innerHTML = "";
  }
  document.body.style.overflow = "";
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
    row.title = "더블클릭하여 탐지 프레임 전체 보기";
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
      const queuePending = lastPipelineState?.detect_queue_pending > 0;
      const batchActive =
        lastPipelineState?.detect_batch_total > 1 &&
        (lastPipelineState?.detect_batch_done || 0) < lastPipelineState.detect_batch_total;
      await loadTimeline({ reset: true });
      if (queuePending || batchActive || lastPipelineState?.detect_status === "running") {
        setStatus(
          els.detectStatus,
          `비디오 완료: ${job.video_name} · 다음 파일 처리 중...`,
        );
        return "running";
      }
      await finalizeDetectJob(job);
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

    if (state.train_status === "cancelled" || state.detect_status === "cancelled") {
      activeDetectJobId = null;
      detectStopRequested = false;
      updateDetectButtonState();
      stopPolling();
      return;
    }

    if (state.detect_status === "cancelling") {
      updateStopButtons(state);
      return;
    }

    const trainRunning = state.train_status === "running";
    if (trainRunning) {
      return;
    }

    const jobId = activeDetectJobId || state.detect_job_id;
    const batchRunning =
      !isDetectStopping(state) &&
      (state.detect_status === "running" ||
        (state.detect_queue_pending || 0) > 0 ||
        ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
          state.detect_status !== "error"));
    const needsDetectUI =
      !!activeDetectJobId ||
      batchRunning ||
      (state.detect_status === "completed" &&
        state.detect_job_id &&
        handledDetectJobId !== state.detect_job_id);

    if (jobId && needsDetectUI) {
      const outcome = await updateDetectUI(jobId);
      if (outcome === "done" && !batchRunning) {
        stopPolling();
        return;
      }
    }

    if (!needsDetectUI && !activeDetectJobId && !batchRunning) {
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
  return (
    state.train_status === "running" ||
    state.detect_status === "running" ||
    (state.detect_queue_pending || 0) > 0 ||
    ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
      state.detect_status !== "error") ||
    !!activeDetectJobId
  );
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

els.frameResults.addEventListener("dblclick", handleFrameCardDblClick);
els.timelineViewport?.addEventListener("dblclick", handleFrameCardDblClick);
els.importFrameResults.addEventListener("dblclick", handleFrameCardDblClick);

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
els.frameLightbox?.addEventListener("click", (event) => {
  if (event.target === els.frameLightbox) closeFrameLightbox();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.frameLightbox?.classList.contains("hidden")) {
    closeFrameLightbox();
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
    updateStopButtons({ train_status: "running" });
    startPolling();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
    els.trainBtn.disabled = false;
  }
});

els.stopTrainBtn?.addEventListener("click", async () => {
  els.stopTrainBtn.disabled = true;
  setStatus(els.trainStatus, "학습 중지 요청...");
  try {
    await api("/api/pipeline/train/stop", { method: "POST" });
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
  detectSession += 1;
  handledDetectJobId = null;
  detectStopRequested = false;

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
    setStatus(els.detectStatus, `${batchTotal}개 Lake 비디오 탐지 시작...`);
    lastPipelineState = {
      ...(lastPipelineState || {}),
      detect_status: "running",
      detect_batch_total: batchTotal,
      detect_batch_done: 0,
      detect_queue_pending: Math.max(0, batchTotal - 1),
    };
    showDetectProgress(0, 0, 0, 0, lastPipelineState);
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
      startDetectPolling(result, batchTotal);
    updateStopButtons({ detect_status: "running", detect_batch_total: batchTotal });
    } catch (err) {
      activeDetectJobId = null;
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
  setStatus(els.detectStatus, `${files.length}개 비디오 탐지 시작...`);
  lastPipelineState = {
    ...(lastPipelineState || {}),
    detect_status: "running",
    detect_batch_total: files.length,
    detect_batch_done: 0,
    detect_queue_pending: Math.max(0, files.length - 1),
  };
  showDetectProgress(0, 0, 0, 0, lastPipelineState);
  try {
    const result = await api("/api/pipeline/detect", { method: "POST", body: form });
    startDetectPolling(result, files.length);
    updateStopButtons({ detect_status: "running", detect_batch_total: files.length });
  } catch (err) {
    activeDetectJobId = null;
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
