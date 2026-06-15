const els = {
  cvatZip: document.getElementById("cvatZip"),
  cvatVideo: document.getElementById("cvatVideo"),
  importBtn: document.getElementById("importBtn"),
  importStatus: document.getElementById("importStatus"),
  importLabels: document.getElementById("importLabels"),
  epochs: document.getElementById("epochs"),
  batch: document.getElementById("batch"),
  trainBtn: document.getElementById("trainBtn"),
  trainStatus: document.getElementById("trainStatus"),
  trainProgressWrap: document.getElementById("trainProgressWrap"),
  trainProgressBar: document.getElementById("trainProgressBar"),
  trainProgressText: document.getElementById("trainProgressText"),
  detectVideo: document.getElementById("detectVideo"),
  frameStride: document.getElementById("frameStride"),
  confidence: document.getElementById("confidence"),
  detectBtn: document.getElementById("detectBtn"),
  resetTimelineBtn: document.getElementById("resetTimelineBtn"),
  detectStatus: document.getElementById("detectStatus"),
  detectProgressWrap: document.getElementById("detectProgressWrap"),
  detectProgressBar: document.getElementById("detectProgressBar"),
  detectProgressText: document.getElementById("detectProgressText"),
  frameResults: document.getElementById("frameResults"),
  frameCount: document.getElementById("frameCount"),
  timelineSummary: document.getElementById("timelineSummary"),
  detectFrameMore: document.getElementById("detectFrameMore"),
  importFramePanel: document.getElementById("importFramePanel"),
  importFrameToggle: document.getElementById("importFrameToggle"),
  importFrameBody: document.getElementById("importFrameBody"),
  importFrameCount: document.getElementById("importFrameCount"),
  importFrameSplit: document.getElementById("importFrameSplit"),
  importFrameResults: document.getElementById("importFrameResults"),
  importFrameMore: document.getElementById("importFrameMore"),
  frameLightbox: document.getElementById("frameLightbox"),
  frameLightboxImg: document.getElementById("frameLightboxImg"),
  frameLightboxMeta: document.getElementById("frameLightboxMeta"),
  frameLightboxClose: document.getElementById("frameLightboxClose"),
};

const POLL_MS = 500;
const IMPORT_FRAME_LIMIT = 48;
const DETECT_FRAME_LIMIT = 48;

let pollTimer = null;
let handledDetectJobId = null;
let activeDetectJobId = null;
let importFrameOffset = 0;
let importFrameTotal = 0;
let detectFrameOffset = 0;
let detectFrameTotal = 0;
let detectFrameJobId = null;
let lastPipelineState = null;
let importFramesLoaded = false;
let importPanelExpanded = false;
let labelSummaryKey = "";
let detectSession = 0;

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
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
    setStatus(els.trainStatus, state.train_progress || "학습 중...");
    els.trainBtn.disabled = true;
  } else if (state.train_status === "completed") {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, 1, "학습 완료");
    setStatus(els.trainStatus, `학습 완료: ${state.train_weights}`, "ok");
    els.trainBtn.disabled = false;
    els.detectBtn.disabled = false;
  } else if (state.train_status === "error") {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, null);
    setStatus(els.trainStatus, state.train_error, "error");
    els.trainBtn.disabled = false;
  } else {
    setProgress(els.trainProgressWrap, els.trainProgressBar, els.trainProgressText, null);
  }
}

function openFrameLightbox(imgSrc, titleHtml, objectsHtml) {
  if (!imgSrc || !els.frameLightbox) return;
  els.frameLightboxImg.src = imgSrc;
  els.frameLightboxMeta.innerHTML = `
    <div><strong>${titleHtml}</strong></div>
    <div class="objects">${objectsHtml || "(없음)"}</div>`;
  els.frameLightbox.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeFrameLightbox() {
  if (!els.frameLightbox) return;
  els.frameLightbox.classList.add("hidden");
  els.frameLightboxImg.removeAttribute("src");
  document.body.style.overflow = "";
}

function renderTimelineSummary(videos) {
  if (!els.timelineSummary) return;
  if (!videos?.length) {
    els.timelineSummary.innerHTML = "";
    return;
  }
  els.timelineSummary.innerHTML = videos
    .map((video) => {
      const start = video.video_start ? video.video_start.replace("T", " ").slice(0, 16) : "시간 미상";
      return `<span class="timeline-chip">${video.video_name} · ${start} · ${video.frames_with_detections}건</span>`;
    })
    .join("");
}

function appendTimelineCards(events) {
  for (const event of events) {
    const card = document.createElement("div");
    card.className = "frame-card";
    card.title = "더블클릭하여 크게 보기";
    const objects = (event.detections || [])
      .map((d) => `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`)
      .join("<br>");
    const absTime = event.absolute_time_label || fmtTime(event.timestamp_sec);
    const imgSrc = event.preview_path
      ? `/api/pipeline/detect/${event.job_id}/previews/${event.preview_path}`
      : "";
    const img = imgSrc ? `<img src="${imgSrc}" alt="" loading="lazy" />` : "";
    const title = `${absTime} · frame #${event.frame_index}`;
    card.innerHTML = `
      ${img}
      <div class="meta">
        <div class="abs-time"><strong>${absTime}</strong></div>
        <div>${event.video_name} · frame #${event.frame_index}</div>
        <div class="objects">${objects || "(없음)"}</div>
      </div>`;
    card.dataset.previewSrc = imgSrc;
    card.dataset.frameTitle = `${absTime} · ${event.video_name} · frame #${event.frame_index}`;
    card.dataset.frameObjects = objects;
    els.frameResults.appendChild(card);
  }
}

async function loadTimeline({ reset = true } = {}) {
  const offset = reset ? 0 : detectFrameOffset;
  try {
    const { timeline } = await api(
      `/api/pipeline/detect/timeline?offset=${offset}&limit=${DETECT_FRAME_LIMIT}`,
    );
    if (reset) {
      els.frameResults.innerHTML = "";
      detectFrameOffset = 0;
    }
    detectFrameTotal = timeline.total || 0;
    if (els.frameCount) els.frameCount.textContent = detectFrameTotal;
    renderTimelineSummary(timeline.videos || []);
    appendTimelineCards(timeline.events || []);
    detectFrameOffset += (timeline.events || []).length;
    els.detectFrameMore?.classList.toggle("hidden", detectFrameOffset >= detectFrameTotal);
    if (timeline.videos?.length) {
      handledDetectJobId = timeline.videos[timeline.videos.length - 1].job_id;
    }
    return timeline;
  } catch (err) {
    console.error("loadTimeline failed", err);
    if (reset) {
      els.frameResults.innerHTML = "";
      if (els.frameCount) els.frameCount.textContent = "0";
      renderTimelineSummary([]);
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
  els.frameResults.innerHTML = "";
  if (els.frameCount) els.frameCount.textContent = "0";
  renderTimelineSummary([]);
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
  try {
    const { job } = await api(`/api/pipeline/detect/${jobId}`);

    if (job.status === "running") {
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

    const trainRunning = state.train_status === "running";
    if (trainRunning) {
      return;
    }

    const jobId = activeDetectJobId || state.detect_job_id;
    const batchRunning =
      state.detect_status === "running" ||
      (state.detect_queue_pending || 0) > 0 ||
      ((state.detect_batch_total || 0) > (state.detect_batch_done || 0) &&
        state.detect_status !== "error");
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
els.importFrameResults.addEventListener("dblclick", handleFrameCardDblClick);

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
    startPolling();
  } catch (err) {
    setStatus(els.trainStatus, err.message, "error");
    els.trainBtn.disabled = false;
  }
});

els.detectBtn.addEventListener("click", async () => {
  const files = els.detectVideo.files;
  if (!files.length) return alert("비디오 파일을 하나 이상 선택하세요.");
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  form.append("frame_stride", els.frameStride.value);
  form.append("confidence", els.confidence.value);
  setStatus(els.detectStatus, `${files.length}개 비디오 탐지 시작...`);
  els.detectBtn.disabled = true;
  detectSession += 1;
  handledDetectJobId = null;
  showDetectProgress(0, 0, 0, 0);
  try {
    const result = await api("/api/pipeline/detect", { method: "POST", body: form });
    const running = result.jobs?.find((job) => job.job_id) || result.job;
    activeDetectJobId = running?.job_id || null;
    startPolling();
  } catch (err) {
    activeDetectJobId = null;
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, err.message, "error");
    els.detectBtn.disabled = false;
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
    const timelineCount = state.detect_timeline?.event_count ?? state.detect_timeline_events ?? 0;
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
