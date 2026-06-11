const state = {
  jobId: null,
  job: null,
  frames: [],
  currentFrameId: null,
  drawingPoints: [], // normalized [x,y]
  image: null,
  imageWidth: 0,
  imageHeight: 0,
};

const els = {
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  jobSelect: document.getElementById("jobSelect"),
  refreshJobsBtn: document.getElementById("refreshJobsBtn"),
  openLocalBtn: document.getElementById("openLocalBtn"),
  videoPlayer: document.getElementById("videoPlayer"),
  timestampInput: document.getElementById("timestampInput"),
  captureBtn: document.getElementById("captureBtn"),
  frameList: document.getElementById("frameList"),
  frameCount: document.getElementById("frameCount"),
  jobInfo: document.getElementById("jobInfo"),
  exportBtn: document.getElementById("exportBtn"),
  canvas: document.getElementById("annotCanvas"),
  emptyCanvas: document.getElementById("emptyCanvas"),
  frameMeta: document.getElementById("frameMeta"),
  undoPointBtn: document.getElementById("undoPointBtn"),
  clearPolygonBtn: document.getElementById("clearPolygonBtn"),
  saveLabelBtn: document.getElementById("saveLabelBtn"),
  deleteLabelBtn: document.getElementById("deleteLabelBtn"),
  metaFrameId: document.getElementById("metaFrameId"),
  metaTime: document.getElementById("metaTime"),
  metaFrameIndex: document.getElementById("metaFrameIndex"),
  metaDrawPoints: document.getElementById("metaDrawPoints"),
  metaSavedPoints: document.getElementById("metaSavedPoints"),
  metaArea: document.getElementById("metaArea"),
  metaSource: document.getElementById("metaSource"),
  metaScore: document.getElementById("metaScore"),
  autoInterval: document.getElementById("autoInterval"),
  autoThreshold: document.getElementById("autoThreshold"),
  autoDetectBtn: document.getElementById("autoDetectBtn"),
  autoDetectStatus: document.getElementById("autoDetectStatus"),
  autoProgressBar: document.getElementById("autoProgressBar"),
  autoProgressText: document.getElementById("autoProgressText"),
};

let autoDetectPoll = null;
let lastAutoDetectStatus = "idle";

const ctx = els.canvas.getContext("2d");

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec % 1) * 10);
  return `${m}:${String(s).padStart(2, "0")}.${ms}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json();
}

function setJobEnabled(enabled) {
  els.captureBtn.disabled = !enabled;
  els.exportBtn.disabled = !enabled;
  els.autoDetectBtn.disabled = !enabled;
}

async function refreshJobs(selectId) {
  const data = await api("/api/jobs");
  els.jobSelect.innerHTML = "";
  if (!data.jobs.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(작업 없음)";
    els.jobSelect.appendChild(opt);
    return;
  }
  for (const job of data.jobs) {
    const opt = document.createElement("option");
    opt.value = job.job_id;
    opt.textContent = `${job.video_name} (${job.annotated_count} labels)`;
    if (job.job_id === selectId) opt.selected = true;
    els.jobSelect.appendChild(opt);
  }
}

function showJobInfo(job) {
  els.jobInfo.classList.remove("hidden");
  els.jobInfo.innerHTML = `
    <div><strong>${job.video_name}</strong></div>
    <div>${fmtTime(job.duration_sec)} · ${job.fps.toFixed(1)} fps · ${job.width}×${job.height}</div>
    <div>레이블 ${job.annotated_count}개</div>`;
}

async function refreshJobMeta(jobId) {
  const { job } = await api(`/api/jobs/${jobId}`);
  state.job = job;
  showJobInfo(job);
}

async function loadJob(jobId, { reloadVideo = true } = {}) {
  if (!jobId) return;
  state.jobId = jobId;
  const { job } = await api(`/api/jobs/${jobId}`);
  state.job = job;
  showJobInfo(job);
  setJobEnabled(true);

  els.videoPlayer.classList.remove("hidden");
  if (reloadVideo) {
    const currentTime = els.videoPlayer.currentTime || 0;
    const wasPaused = els.videoPlayer.paused;
    els.videoPlayer.src = `/api/jobs/${jobId}/media/video?t=${Date.now()}`;
    els.videoPlayer.addEventListener(
      "loadedmetadata",
      () => {
        if (currentTime > 0 && currentTime < job.duration_sec) {
          els.videoPlayer.currentTime = currentTime;
        }
        if (!wasPaused) {
          els.videoPlayer.play().catch(() => {});
        }
      },
      { once: true },
    );
  }
  els.timestampInput.max = job.duration_sec;

  await loadFrames();

  const { state: detectState } = await api(`/api/jobs/${jobId}/auto-detect`);
  lastAutoDetectStatus = detectState.status || "idle";
  syncAutoDetectPolling(detectState);
  renderAutoDetectUI(detectState, jobId);
}

async function loadFrames(selectFrameId) {
  const data = await api(`/api/jobs/${state.jobId}/frames`);
  state.frames = data.frames || [];
  els.frameCount.textContent = state.frames.length;
  renderFrameList();

  const target = selectFrameId || state.currentFrameId;
  if (target) {
    await selectFrame(target);
  } else if (state.frames.length) {
    await selectFrame(state.frames[state.frames.length - 1].frame_id);
  }
}

function renderFrameList() {
  els.frameList.innerHTML = "";
  state.frames.forEach((frame) => {
    const item = document.createElement("div");
    item.className = "frame-item" + (frame.frame_id === state.currentFrameId ? " active" : "");
    const thumb = `/api/jobs/${state.jobId}/media/frames/${frame.image}`;
    const tagClass = frame.source === "auto" ? "auto" : frame.has_label ? "ok" : "no";
    const tagText = frame.source === "auto" ? "auto" : frame.has_label ? "saved" : "draft";
    const scoreText = frame.score != null ? ` · ${(frame.score * 100).toFixed(0)}%` : "";
    item.innerHTML = `
      <img src="${thumb}" alt="" loading="lazy" />
      <div>
        <div class="title">${fmtTime(frame.timestamp_sec)}
          <span class="tag ${tagClass}">${tagText}</span>${scoreText}
        </div>
        <div class="meta">${frame.frame_id}</div>
      </div>`;
    item.addEventListener("click", () => selectFrame(frame.frame_id));
    els.frameList.appendChild(item);
  });
}

function canvasPointFromEvent(event) {
  const rect = els.canvas.getBoundingClientRect();
  const scaleX = state.imageWidth / rect.width;
  const scaleY = state.imageHeight / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return {
    px: [x, y],
    norm: [x / state.imageWidth, y / state.imageHeight],
  };
}

function redrawCanvas() {
  if (!state.image) return;
  const w = state.imageWidth;
  const h = state.imageHeight;
  els.canvas.width = w;
  els.canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(state.image, 0, 0);

  const frame = state.frames.find((f) => f.frame_id === state.currentFrameId);
  const saved = frame?.label?.polygon_norm;

  const drawPoly = (points, { fill, stroke, lineWidth }) => {
    if (!points || points.length < 2) return;
    ctx.beginPath();
    const p0 = points[0];
    const x0 = Array.isArray(p0) ? p0[0] * w : p0.x * w;
    const y0 = Array.isArray(p0) ? p0[1] * h : p0.y * h;
    ctx.moveTo(x0, y0);
    for (let i = 1; i < points.length; i++) {
      const p = points[i];
      const x = (Array.isArray(p) ? p[0] : p.x) * w;
      const y = (Array.isArray(p) ? p[1] : p.y) * h;
      ctx.lineTo(x, y);
    }
    if (points.length >= 3) ctx.closePath();
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    if (points.length >= 3) ctx.fill();
    ctx.stroke();
  };

  if (saved?.length >= 3) {
    drawPoly(saved, {
      fill: "rgba(34, 197, 94, 0.2)",
      stroke: "#22c55e",
      lineWidth: Math.max(2, w / 640),
    });
  }

  if (state.drawingPoints.length) {
    drawPoly(state.drawingPoints, {
      fill: "rgba(59, 130, 246, 0.15)",
      stroke: "#3b82f6",
      lineWidth: Math.max(2, w / 640),
    });
    for (const [nx, ny] of state.drawingPoints) {
      ctx.beginPath();
      ctx.arc(nx * w, ny * h, Math.max(4, w / 320), 0, Math.PI * 2);
      ctx.fillStyle = "#3b82f6";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  updateMeta();
  updateButtons();
}

function updateButtons() {
  const hasFrame = !!state.currentFrameId;
  const hasDraw = state.drawingPoints.length > 0;
  els.undoPointBtn.disabled = !hasDraw;
  els.clearPolygonBtn.disabled = !hasDraw;
  els.saveLabelBtn.disabled = !hasFrame || state.drawingPoints.length < 3;
  const frame = state.frames.find((f) => f.frame_id === state.currentFrameId);
  els.deleteLabelBtn.disabled = !frame?.has_label;
}

function updateMeta() {
  const frame = state.frames.find((f) => f.frame_id === state.currentFrameId);
  if (!frame) {
    els.frameMeta.classList.add("hidden");
    return;
  }
  els.frameMeta.classList.remove("hidden");
  els.metaFrameId.textContent = frame.frame_id;
  els.metaTime.textContent = `${fmtTime(frame.timestamp_sec)} (${frame.timestamp_sec}s)`;
  els.metaFrameIndex.textContent = frame.frame_index;
  els.metaDrawPoints.textContent = state.drawingPoints.length;
  els.metaSavedPoints.textContent = frame.label?.point_count ?? (frame.has_label ? "?" : "-");
  els.metaArea.textContent = frame.label?.area_ratio ?? "-";
  els.metaSource.textContent = frame.source ?? (frame.has_label ? "manual" : "-");
  els.metaScore.textContent = frame.score != null ? `${(frame.score * 100).toFixed(1)}%` : "-";
}

async function selectFrame(frameId) {
  state.currentFrameId = frameId;
  state.drawingPoints = [];
  renderFrameList();

  const frame = state.frames.find((f) => f.frame_id === frameId);
  if (!frame) return;

  els.emptyCanvas.classList.add("hidden");

  const img = new Image();
  img.onload = () => {
    state.image = img;
    state.imageWidth = img.width;
    state.imageHeight = img.height;

    if (frame.label?.polygon_norm?.length) {
      state.drawingPoints = frame.label.polygon_norm.map(([x, y]) => [x, y]);
    }
    redrawCanvas();
  };
  img.src = `/api/jobs/${state.jobId}/media/frames/${frame.image}?t=${Date.now()}`;
}

async function captureAtTimestamp(sec) {
  if (!state.jobId) return;
  const { frame } = await api(`/api/jobs/${state.jobId}/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ timestamp_sec: Number(sec) }),
  });
  await loadJob(state.jobId);
  await selectFrame(frame.frame_id);
}

async function saveLabel() {
  if (!state.currentFrameId || state.drawingPoints.length < 3) return;
  await api(`/api/jobs/${state.jobId}/frames/${state.currentFrameId}/label`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ polygon_norm: state.drawingPoints, class_id: 0 }),
  });
  await loadJob(state.jobId);
  await selectFrame(state.currentFrameId);
}

async function deleteLabel() {
  if (!state.currentFrameId) return;
  await api(`/api/jobs/${state.jobId}/frames/${state.currentFrameId}/label`, { method: "DELETE" });
  state.drawingPoints = [];
  await loadJob(state.jobId);
  await selectFrame(state.currentFrameId);
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { job } = await api("/api/upload", { method: "POST", body: form });
  await refreshJobs(job.job_id);
  await loadJob(job.job_id);
}

// Events
els.dropZone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) uploadFile(e.target.files[0]);
});
els.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); els.dropZone.classList.add("dragover"); });
els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragover"));
els.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});

els.openLocalBtn.addEventListener("click", async () => {
  const { job } = await api("/api/open-local", { method: "POST" });
  await refreshJobs(job.job_id);
  await loadJob(job.job_id);
});

els.refreshJobsBtn.addEventListener("click", () => refreshJobs(state.jobId));
els.jobSelect.addEventListener("change", () => loadJob(els.jobSelect.value));

els.captureBtn.addEventListener("click", () => captureAtTimestamp(els.timestampInput.value));
els.videoPlayer.addEventListener("timeupdate", () => {
  els.timestampInput.value = Number(els.videoPlayer.currentTime.toFixed(1));
});

document.querySelectorAll(".bookmark").forEach((btn) => {
  btn.addEventListener("click", () => {
    const sec = Number(btn.dataset.sec);
    els.timestampInput.value = sec;
    if (state.jobId) {
      els.videoPlayer.currentTime = sec;
      captureAtTimestamp(sec);
    }
  });
});

els.canvas.addEventListener("click", (e) => {
  if (!state.image) return;
  const { norm } = canvasPointFromEvent(e);
  state.drawingPoints.push([norm[0], norm[1]]);
  redrawCanvas();
});

els.undoPointBtn.addEventListener("click", () => {
  state.drawingPoints.pop();
  redrawCanvas();
});

els.clearPolygonBtn.addEventListener("click", () => {
  state.drawingPoints = [];
  redrawCanvas();
});

els.saveLabelBtn.addEventListener("click", saveLabel);
els.deleteLabelBtn.addEventListener("click", deleteLabel);

els.exportBtn.addEventListener("click", async () => {
  const result = await api(`/api/jobs/${state.jobId}/export`, { method: "POST" });
  alert(`학습용 dataset에 ${result.exported}개 레이블을보냈습니다.`);
});

function renderAutoDetectUI(detectState, jobId) {
  const status = detectState?.status || "idle";
  if (status === "idle") {
    els.autoDetectStatus.classList.add("hidden");
    els.autoDetectBtn.disabled = !jobId;
    return;
  }
  els.autoDetectStatus.classList.remove("hidden");
  const pct = Math.round((detectState.progress || 0) * 100);
  els.autoProgressBar.style.width = `${pct}%`;
  if (status === "running") {
    els.autoProgressText.textContent =
      `진행 중 ${pct}% · 처리 ${detectState.processed}/${detectState.total} · 감지 ${detectState.detected}`;
    els.autoDetectBtn.disabled = true;
  } else if (status === "completed") {
    els.autoProgressText.textContent = `완료 · 감지 ${detectState.detected}개`;
    els.autoDetectBtn.disabled = false;
  } else if (status === "error") {
    els.autoProgressText.textContent = `오류: ${detectState.error}`;
    els.autoDetectBtn.disabled = false;
  }
}

function syncAutoDetectPolling(detectState) {
  if (detectState.status === "running") {
    if (!autoDetectPoll) {
      autoDetectPoll = setInterval(pollAutoDetect, 1500);
    }
    return;
  }
  stopAutoDetectPoll();
}

async function handleAutoDetectStatusChange(detectState, jobId) {
  const status = detectState.status || "idle";
  if (status === lastAutoDetectStatus) {
    renderAutoDetectUI(detectState, jobId);
    return;
  }

  const prev = lastAutoDetectStatus;
  lastAutoDetectStatus = status;
  renderAutoDetectUI(detectState, jobId);
  syncAutoDetectPolling(detectState);

  if (prev === "running" && status === "completed") {
    await loadFrames();
    await refreshJobMeta(jobId);
    await refreshJobs(jobId);
  }
}

function stopAutoDetectPoll() {
  if (autoDetectPoll) {
    clearInterval(autoDetectPoll);
    autoDetectPoll = null;
  }
}

async function pollAutoDetect() {
  if (!state.jobId) return;
  try {
    const { state: detectState } = await api(`/api/jobs/${state.jobId}/auto-detect`);
    await handleAutoDetectStatusChange(detectState, state.jobId);
    if (detectState.status === "running") {
      await loadFrames();
      await refreshJobMeta(state.jobId);
    }
  } catch (err) {
    console.error("auto-detect poll failed", err);
    stopAutoDetectPoll();
    els.autoProgressText.textContent = `오류: ${err.message}`;
  }
}

async function startAutoDetect() {
  if (!state.jobId) return;
  const interval = Number(els.autoInterval.value);
  const threshold = Number(els.autoThreshold.value);
  await api(`/api/jobs/${state.jobId}/auto-detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ interval_sec: interval, threshold }),
  });
  lastAutoDetectStatus = "idle";
  els.autoDetectStatus.classList.remove("hidden");
  stopAutoDetectPoll();
  autoDetectPoll = setInterval(pollAutoDetect, 1500);
  await pollAutoDetect();
}

els.autoDetectBtn.addEventListener("click", startAutoDetect);

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    els.undoPointBtn.click();
  }
  if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    els.saveLabelBtn.click();
  }
});

refreshJobs().then(() => {
  if (els.jobSelect.value) loadJob(els.jobSelect.value);
});
