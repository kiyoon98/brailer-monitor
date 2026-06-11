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
  detectStatus: document.getElementById("detectStatus"),
  detectProgressWrap: document.getElementById("detectProgressWrap"),
  detectProgressBar: document.getElementById("detectProgressBar"),
  detectProgressText: document.getElementById("detectProgressText"),
  frameResults: document.getElementById("frameResults"),
  importFramePanel: document.getElementById("importFramePanel"),
  importFrameCount: document.getElementById("importFrameCount"),
  importFrameSplit: document.getElementById("importFrameSplit"),
  importFrameResults: document.getElementById("importFrameResults"),
  importFrameMore: document.getElementById("importFrameMore"),
};

const POLL_MS = 500;
const IMPORT_FRAME_LIMIT = 48;

let pollTimer = null;
let handledDetectJobId = null;
let activeDetectJobId = null;
let importFrameOffset = 0;
let importFrameTotal = 0;

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

function detectProgressText(processed, total, withObjects, pct) {
  const totalLabel = total > 0 ? total : "?";
  return `탐지 중 ${processed}/${totalLabel} 프레임 · 객체 ${withObjects}개 (${Math.round(pct * 100)}%)`;
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
    return;
  }

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
    const objects = (frame.objects || [])
      .map((obj) => `${obj.class_name} (${obj.shape})`)
      .join("<br>");
    card.innerHTML = `
      <img src="${frame.preview_url}" alt="" loading="lazy" />
      <div class="meta">
        <div>
          <strong>frame #${frame.frame_index}</strong>
          <span class="split-tag">${frame.split}</span>
        </div>
        <div class="objects">${objects || "(객체 없음)"}</div>
      </div>`;
    els.importFrameResults.appendChild(card);
  }
}

async function loadImportFrames({ reset = false } = {}) {
  if (reset) {
    importFrameOffset = 0;
    els.importFrameResults.innerHTML = "";
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
    els.importFrameMore.classList.toggle("hidden", importFrameOffset >= importFrameTotal);
  } catch (err) {
    if (reset) {
      els.importFramePanel.classList.add("hidden");
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

function showDetectProgress(processed, total, withObjects, pct) {
  setProgress(
    els.detectProgressWrap,
    els.detectProgressBar,
    els.detectProgressText,
    pct,
    detectProgressText(processed, total, withObjects, pct),
  );
  setStatus(els.detectStatus, "탐지 진행 중...");
  els.detectBtn.disabled = true;
}

function renderPipelineState(state) {
  const meta = state.dataset_meta;

  if (state.import_status === "completed" && meta) {
    setStatus(
      els.importStatus,
      `Import 완료 · train ${meta.train_images} / val ${meta.val_images} · ${meta.task_type}`,
      "ok",
    );
    if (meta.label_summary) {
      renderLabelSummary(meta.label_summary);
    } else if (meta.class_names?.length) {
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
    els.trainBtn.disabled = false;
    loadImportFrames({ reset: true });
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

  if (state.detect_status === "running") {
    showDetectProgress(
      state.detect_processed_frames || 0,
      state.detect_total_frames || 0,
      state.detect_frames_with_objects || 0,
      state.detect_progress_pct || 0,
    );
  } else if (state.detect_status === "completed" && !activeDetectJobId) {
    const processed = state.detect_processed_frames || 0;
    const withObjects = state.detect_frames_with_objects || 0;
    setProgress(
      els.detectProgressWrap,
      els.detectProgressBar,
      els.detectProgressText,
      1,
      `완료: ${processed}프레임 · 객체 탐지 ${withObjects}프레임`,
    );
  } else if (state.detect_status === "error") {
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, state.detect_error, "error");
    els.detectBtn.disabled = false;
  }
}

function renderDetectJob(job) {
  if (job.status === "running") {
    showDetectProgress(
      job.processed_frames || 0,
      job.total_frames || 0,
      job.frames_with_detections || 0,
      job.progress || 0,
    );
  } else if (job.status === "completed") {
    setProgress(
      els.detectProgressWrap,
      els.detectProgressBar,
      els.detectProgressText,
      1,
      `완료: ${job.processed_frames}프레임 · 객체 탐지 ${job.frames_with_detections}프레임`,
    );
    els.detectBtn.disabled = false;
  }
}

function renderDetections(manifest, jobId) {
  const frames = manifest.frames || [];
  const withObjects = frames.filter((f) => f.detections.length > 0);
  els.frameCount.textContent = withObjects.length;

  els.frameResults.innerHTML = "";
  for (const frame of withObjects) {
    const card = document.createElement("div");
    card.className = "frame-card";
    const objects = frame.detections
      .map((d) => `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`)
      .join("<br>");
    const img = frame.preview_path
      ? `<img src="/api/pipeline/detect/${jobId}/previews/${frame.preview_path}" alt="" loading="lazy" />`
      : "";
    card.innerHTML = `
      ${img}
      <div class="meta">
        <div><strong>${fmtTime(frame.timestamp_sec)}</strong> · frame #${frame.frame_index}</div>
        <div class="objects">${objects || "(없음)"}</div>
      </div>`;
    els.frameResults.appendChild(card);
  }

  setStatus(
    els.detectStatus,
    `완료: ${manifest.frames_processed}프레임 처리, ${manifest.frames_with_detections}프레임에서 객체 탐지`,
    "ok",
  );
}

async function pollDetectJob(jobId) {
  const { job } = await api(`/api/pipeline/detect/${jobId}`);
  renderDetectJob(job);

  if (job.status === "completed" && handledDetectJobId !== job.job_id) {
    handledDetectJobId = job.job_id;
    activeDetectJobId = null;
    const { manifest } = await api(`/api/pipeline/detect/${job.job_id}/results`);
    renderDetections(manifest, job.job_id);
    return "done";
  }
  if (job.status === "error") {
    activeDetectJobId = null;
    setStatus(els.detectStatus, job.error, "error");
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    els.detectBtn.disabled = false;
    return "done";
  }
  return "running";
}

async function poll() {
  try {
    const { state } = await api("/api/pipeline/state");
    renderPipelineState(state);

    const trainRunning = state.train_status === "running";
    const jobId = activeDetectJobId || state.detect_job_id;
    const detectRunning =
      state.detect_status === "running" || (activeDetectJobId && activeDetectJobId === jobId);

    if (jobId && detectRunning) {
      const outcome = await pollDetectJob(jobId);
      if (outcome === "done") {
        stopPolling();
        return;
      }
    }

    if (!trainRunning && !detectRunning && !activeDetectJobId) {
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
  return state.train_status === "running" || state.detect_status === "running";
}

els.cvatZip.addEventListener("change", () => {
  previewCvatFile(els.cvatZip.files[0]);
});

els.importFrameSplit.addEventListener("change", () => {
  loadImportFrames({ reset: true });
});

els.importFrameMore.addEventListener("click", () => {
  loadImportFrames();
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
    await loadImportFrames({ reset: true });
    els.trainBtn.disabled = false;
  } catch (err) {
    setStatus(els.importStatus, err.message, "error");
  }
});

els.trainBtn.addEventListener("click", async () => {
  setStatus(els.trainStatus, "학습 시작...");
  els.trainBtn.disabled = true;
  handledDetectJobId = null;
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
  const file = els.detectVideo.files[0];
  if (!file) return alert("비디오 파일을 선택하세요.");
  const form = new FormData();
  form.append("file", file);
  form.append("frame_stride", els.frameStride.value);
  form.append("confidence", els.confidence.value);
  setStatus(els.detectStatus, "탐지 시작...");
  els.detectBtn.disabled = true;
  els.frameResults.innerHTML = "";
  handledDetectJobId = null;
  showDetectProgress(0, 0, 0, 0);
  try {
    const { job } = await api("/api/pipeline/detect", { method: "POST", body: form });
    activeDetectJobId = job.job_id;
    startPolling();
    renderDetectJob({ ...job, status: "running", progress: 0, processed_frames: 0, total_frames: 0 });
  } catch (err) {
    activeDetectJobId = null;
    setProgress(els.detectProgressWrap, els.detectProgressBar, els.detectProgressText, null);
    setStatus(els.detectStatus, err.message, "error");
    els.detectBtn.disabled = false;
  }
});

api("/api/pipeline/state")
  .then(({ state }) => {
    renderPipelineState(state);
    if (state.dataset_meta?.label_summary) {
      renderLabelSummary(state.dataset_meta.label_summary);
    }
    if (state.import_status === "completed" && state.dataset_meta) {
      loadImportFrames({ reset: true });
    }
    if (state.detect_status === "running" && state.detect_job_id) {
      activeDetectJobId = state.detect_job_id;
    }
    if (needsPolling(state) || activeDetectJobId) {
      startPolling();
    }
  })
  .catch(console.error);
