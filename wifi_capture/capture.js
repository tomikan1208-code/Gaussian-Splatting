"use strict";

const byId = (id) => document.getElementById(id);
const camera = byId("camera");
const cameraView = byId("camera-view");
const cameraSettings = byId("camera-settings");
const portrait = matchMedia("(orientation: portrait)");
const canvas = document.createElement("canvas");
const drawing = canvas.getContext("2d", { alpha: false });
const stillCanvas = document.createElement("canvas");
const stillDrawing = stillCanvas.getContext("2d", { alpha: false });
const modes = [...document.querySelectorAll("[data-mode]")];
const queueLimit = 4;
const photoSlots = 2;
const cameraReady = Boolean(window.isSecureContext && navigator.mediaDevices?.getUserMedia);
const captureTransfer = { kind: "capture", queue: [] };
const uploadTransfer = { kind: "upload", queue: [], currentBytes: 0 };
const transfers = { capture: captureTransfer, upload: uploadTransfer };
const trainingStatuses = {
  running: "PCで学習中",
  completed: "学習完了。結果はPCの runs にあります。",
  failed: "学習に失敗しました。PCでログを確認してください。",
};

let code = location.hash.slice(1) || sessionStorage.getItem("pairCode") || "";
if (location.hash) history.replaceState(null, "", location.pathname + location.search);
let server;
let stream;
let cameraState = "closed";
let cameraAttempt = 0;
let previewTimeout;
let mode = new URLSearchParams(location.search).get("mode") === "capture" ? "capture" : "upload";
let frameEncoding = false;
let photoEncoding = false;
let captureBatch;
let uploadBatch;
let motionEnabled = false;
let orientation;
let wakeLock;
let training;
let startingTraining = false;
let checkingTraining = false;
let trainingError = "";

function message(text, error = false, kind = mode) {
  transfers[kind].notice = text;
  transfers[kind].noticeError = error;
  renderMessage();
}

function renderMessage() {
  const { notice = "", noticeError = false } = transfers[mode];
  const note = byId("message");
  note.hidden = !notice;
  note.textContent = notice;
  note.dataset.error = String(noticeError);
  const warning = byId("camera-error");
  const cameraNotice = captureTransfer.noticeError ? captureTransfer.notice : "";
  warning.hidden = !cameraNotice;
  warning.textContent = cameraNotice;
}

async function api(path, body, pairCode = code) {
  const headers = { "X-Pair-Code": pairCode };
  const options = { cache: "no-store", headers };
  if (body !== undefined) {
    options.method = "POST";
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "PCと通信できませんでした。");
  return data;
}

function shareLink() {
  const link = server.links[Number(byId("address").value)];
  if (!link) return;
  const qr = byId("qr");
  qr.hidden = false;
  qr.src = "/pair.png?address=" + encodeURIComponent(link.address);
  const manual = byId("share-link");
  const baseUrl = new URL(link.url);
  baseUrl.hash = "";
  manual.href = baseUrl.href;
  manual.textContent = baseUrl.href;
}

function applyServer(state) {
  server = state;
  training = state.training;
  if (state.local) code = state.code;
  if (state.paired) sessionStorage.setItem("pairCode", code);
  else sessionStorage.removeItem("pairCode");
  const connection = byId("connection");
  connection.textContent = state.paired ? "PC接続済み" : "未接続";
  connection.dataset.online = String(state.paired);
  byId("pair-panel").hidden = state.paired;
  byId("workspace").hidden = !state.paired;
  byId("desktop-panel").hidden = !state.local;
  byId("stop").hidden = !state.local;
  if (state.local) {
    byId("no-network").hidden = Boolean(state.links.length);
    byId("share-layout").hidden = !state.links.length;
    byId("share-code").textContent = state.code;
    byId("address").replaceChildren(...state.links.map((link, index) => new Option(link.address, String(index))));
    shareLink();
  }
  byId("tls-panel").hidden = cameraReady;
  byId("camera-entry").hidden = !cameraReady;
  byId("tls-ready").hidden = !state.tlsPort;
  byId("tls-unavailable").hidden = Boolean(state.tlsPort);
  byId("upload-camera-hint").hidden = cameraReady || !state.tlsPort;
  if (state.tlsPort) {
    const secure = new URL(location.href);
    secure.protocol = "https:";
    secure.port = state.tlsPort;
    secure.hash = code;
    secure.searchParams.set("mode", "capture");
    byId("secure-link").href = secure.href;
    secure.searchParams.set("mode", "upload");
    byId("secure-upload-link").href = secure.href;
  }
  renderProgress();
  renderTraining();
}

async function initialize() {
  try {
    applyServer(await api("/api/state"));
  } catch {
    byId("connection").textContent = "PCに接続できません";
    byId("pair-panel").hidden = false;
    byId("pair-note").textContent = "PCで撮影サーバーが動いていることを確認してください。";
  }
}

byId("pair-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  code = byId("pair-code").value.trim();
  const note = byId("pair-note");
  note.dataset.error = "true";
  try {
    const state = await api("/api/state");
    if (!state.paired) throw new Error("接続コードが違います。PCの画面と照らし合わせてください。");
    note.textContent = "";
    applyServer(state);
  } catch (error) {
    note.textContent = error.message || "PCと通信できませんでした。";
  }
});

byId("address").addEventListener("change", shareLink);
byId("qr").addEventListener("error", () => { byId("qr").hidden = true; });
byId("copy-link").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(server.links[Number(byId("address").value)].url);
    byId("copy-link").textContent = "コピーしました";
    setTimeout(() => { byId("copy-link").textContent = "リンクをコピー"; }, 1600);
  } catch {
    byId("share-link").closest("details").open = true;
  }
});

function renderTraining() {
  const batch = mode === "capture" ? captureBatch : uploadBatch;
  const transfer = transfers[mode];
  const ready = Boolean(batch?.id && batch.saved && !batch.recording && !batch.cancelled
    && !transfer.sending && !transfer.queue.length);
  const canTrain = ready && (batch.saved >= 3 || batch.hasVideo);
  const running = training?.state === "running";
  if (training?.session && captureBatch?.id === training.session) captureBatch.sealed = true;
  if (training?.state === "completed") {
    for (const item of [captureBatch, uploadBatch]) {
      if (item?.id === training.session) item.trained = true;
    }
  }

  const button = byId("train");
  button.hidden = !canTrain || Boolean(batch?.trained);
  button.disabled = running || startingTraining;
  button.textContent = training?.state === "failed" && training.session === batch?.id
    ? "このフォルダで学習を再試行" : "このフォルダで学習開始";
  const job = running || !ready || training?.session === batch?.id ? training : undefined;
  byId("training-control").hidden = !ready && !job && !startingTraining && !trainingError;
  byId("stop").disabled = running || startingTraining;

  const text = startingTraining ? "PCで学習を起動しています。" : trainingError
    || (job ? trainingStatuses[job.state] : "")
    || (ready && !canTrain ? "学習には同じ場所を撮った写真を3枚以上まとめて送るか、動画を送ってください。" : "");
  const status = byId("training-status");
  if (status.textContent !== text) status.textContent = text;
  const detail = byId("training-log");
  detail.hidden = !job;
  const path = job?.state === "failed" ? job.log : job?.folder ?? "";
  if (detail.textContent !== path) detail.textContent = path;
}

async function refreshTraining() {
  if (!server?.paired || startingTraining || checkingTraining || document.hidden) return;
  checkingTraining = true;
  try {
    const { job } = await api("/api/training");
    if (!startingTraining) {
      if (job && job.log !== training?.log && (job.session === captureBatch?.id || job.session === uploadBatch?.id)) {
        trainingError = "";
      }
      training = job;
      renderTraining();
    }
  } catch {
    // Keep the last status while the phone is reconnecting.
  } finally {
    checkingTraining = false;
  }
}

byId("train").addEventListener("click", async () => {
  const batch = mode === "capture" ? captureBatch : uploadBatch;
  startingTraining = true;
  trainingError = "";
  renderTraining();
  try {
    ({ job: training } = await api("/api/training", { id: batch.id, saved: batch.saved }));
  } catch (error) {
    trainingError = error.message;
  } finally {
    startingTraining = false;
    renderTraining();
  }
});
setInterval(refreshTraining, 3000);

function hasCameraFrame() {
  return stream && !camera.paused && camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    && camera.videoWidth > 0 && camera.videoHeight > 0;
}

function controls() {
  const uploading = uploadTransfer.queue.length > 0;
  for (const id of ["upload-label", "files"]) byId(id).disabled = uploading;
  byId("motion").disabled = motionEnabled;
  byId("send").disabled = uploading || byId("files").files.length === 0;
  const recording = Boolean(captureBatch?.recording);
  const frameReady = cameraState === "ready";
  const photo = byId("still-action");
  const full = captureTransfer.queue.length >= queueLimit;
  photo.textContent = full ? "送信待ちがいっぱい" : "写真を撮る";
  photo.disabled = !frameReady || photoEncoding || Boolean(captureTransfer.failed) || full;

  const shutter = byId("capture-action");
  let label = "映像を準備中";
  let disabled = true;
  if (recording) {
    label = "撮影停止";
    disabled = false;
  } else if (captureTransfer.failed) {
    label = "未送信を再送";
    disabled = false;
  } else if (cameraState === "play-needed") {
    label = "映像を表示";
    disabled = false;
  } else if (cameraState === "retry") {
    label = "カメラを再試行";
    disabled = false;
  } else if (frameReady) {
    label = "撮影開始";
    disabled = false;
  }
  byId("capture-action-label").textContent = label;
  shutter.disabled = disabled;
  shutter.dataset.kind = frameReady && !captureTransfer.failed ? "record" : "";
  shutter.dataset.recording = String(recording);
  byId("queue-actions").hidden = !transfers[mode].failed;
  renderTraining();
}

function showMode(value) {
  mode = value;
  trainingError = "";
  byId("capture-panel").hidden = value !== "capture";
  byId("upload-panel").hidden = value !== "upload";
  for (const button of modes) button.setAttribute("aria-pressed", String(button.dataset.mode === value));
  renderMessage();
  renderProgress();
  controls();
}
for (const button of modes) button.addEventListener("click", () => showMode(button.dataset.mode));
for (const id of ["secure-link", "secure-upload-link"]) {
  byId(id).addEventListener("click", (event) => {
    if (!uploadTransfer.queue.length) return;
    event.preventDefault();
    showMode("upload");
    if (!uploadTransfer.failed) message("送信が完了すると撮影ページを開けます。");
  });
}

function renderProgress() {
  const selected = mode === "capture" ? captureBatch : uploadBatch;
  const destination = byId("destination");
  const path = selected?.folder ?? server?.output ?? "";
  if (destination.textContent !== path) destination.textContent = path;
  const percentage = uploadBatch?.totalBytes
    ? 100 * (uploadBatch.sentBytes + uploadTransfer.currentBytes) / uploadBatch.totalBytes : 0;
  const progress = byId("transfer");
  progress.hidden = !uploadBatch || mode !== "upload";
  progress.value = percentage;

  const waiting = captureTransfer.queue.length;
  const captureStatus = captureBatch ? "保存 " + captureBatch.saved + "枚　送信待ち " + waiting + "枚" : "";
  const uploadStatus = uploadTransfer.failed ? "ファイル送信は再送待ち"
    : uploadTransfer.queue.length ? "ファイル送信 " + Math.floor(percentage) + "%" : "";
  const cameraProgress = byId("camera-progress");
  const summary = [captureStatus, uploadStatus].filter(Boolean).join(" ／ ");
  cameraProgress.hidden = !summary;
  if (cameraProgress.textContent !== summary) cameraProgress.textContent = summary;

  const stats = byId("camera-stats");
  stats.hidden = !captureBatch;
  if (captureBatch && cameraSettings.open) {
    const seconds = (captureBatch.lastFrame - captureBatch.firstFrame) / 1000;
    const rate = seconds > 0 ? "　実測 " + ((captureBatch.frames - 1) / seconds).toFixed(1) + " fps" : "";
    stats.textContent = captureStatus + "　取りこぼし " + captureBatch.dropped + "枚" + rate;
  }
}

function showPreviewState(state, text = "") {
  cameraState = state;
  const note = byId("preview-note");
  note.hidden = !text;
  note.textContent = text;
  controls();
}

function showLiveFrame() {
  if (!hasCameraFrame()) return;
  clearTimeout(previewTimeout);
  if (cameraState !== "ready") showPreviewState("ready");
  showSettingsValues();
}

function playPreview() {
  const attempt = cameraAttempt;
  showPreviewState("waiting", "映像を準備しています…");
  clearTimeout(previewTimeout);
  previewTimeout = setTimeout(() => {
    if (attempt !== cameraAttempt || cameraState === "ready") return;
    if (hasCameraFrame()) showLiveFrame();
    else showPreviewState("play-needed", "「映像を表示」を押してください。");
  }, 4000);
  camera.play().then(showLiveFrame).catch(() => {
    if (attempt === cameraAttempt && cameraState === "waiting") {
      clearTimeout(previewTimeout);
      showPreviewState("play-needed", "「映像を表示」を押してください。");
    }
  });
}

async function openCamera() {
  const attempt = ++cameraAttempt;
  releaseCamera();
  showPreviewState("requesting", "カメラへのアクセスを許可してください。");
  try {
    const connected = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        aspectRatio: { ideal: 4 / 3 },
        width: { ideal: 3840 },
        height: { ideal: 2880 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    if (attempt !== cameraAttempt) {
      for (const track of connected.getTracks()) track.stop();
      return;
    }
    stream = connected;
    camera.defaultMuted = true;
    camera.muted = true;
    camera.playsInline = true;
    camera.srcObject = connected;
    connected.getVideoTracks()[0].addEventListener("ended", () => {
      if (stream !== connected) return;
      if (captureBatch?.recording) stopCapture(captureBatch);
      releaseCamera();
      showPreviewState("retry", "カメラの接続が切れました。");
    }, { once: true });
    playPreview();
  } catch (error) {
    if (attempt !== cameraAttempt) return;
    releaseCamera();
    if (error.name === "NotAllowedError") {
      showPreviewState("retry", "Safariのサイト設定でカメラを許可し、再試行してください。");
    } else if (error.name === "NotFoundError") {
      showPreviewState("retry", "カメラが見つかりませんでした。");
    } else {
      showPreviewState("retry", "カメラを開けませんでした。ほかの撮影アプリを閉じて再試行してください。");
    }
  }
}

function releaseCamera() {
  clearTimeout(previewTimeout);
  const previous = stream;
  stream = undefined;
  camera.srcObject = null;
  for (const track of previous?.getTracks() ?? []) track.stop();
}

byId("open-camera").addEventListener("click", () => {
  if (!cameraView.hidden) return;
  if (!history.state?.wifiCamera) history.pushState({ wifiCamera: true }, "");
  cameraView.hidden = false;
  document.body.classList.add("camera-open");
  showSettingsValues();
  openCamera();
});

function closeCamera(goBack = false) {
  if (cameraView.hidden) return;
  cameraAttempt += 1;
  if (captureBatch?.recording) stopCapture(captureBatch);
  releaseCamera();
  cameraState = "closed";
  if (cameraSettings.open) cameraSettings.close();
  cameraView.hidden = true;
  document.body.classList.remove("camera-open");
  if (goBack && history.state?.wifiCamera) history.back();
  controls();
  if (!byId("open-camera").disabled) byId("open-camera").focus({ preventScroll: true });
}
byId("close-camera").addEventListener("click", () => closeCamera(true));
window.addEventListener("popstate", () => { if (!history.state?.wifiCamera) closeCamera(); });
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !cameraView.hidden && !cameraSettings.open) closeCamera(true);
});

function frameCrop(edge) {
  const { videoWidth, videoHeight } = camera;
  if (!videoWidth || !videoHeight) return null;
  const horizontal = portrait.matches ? 3 : 4;
  const vertical = portrait.matches ? 4 : 3;
  const fit = Math.min(videoWidth / horizontal, videoHeight / vertical);
  const scale = Math.floor(edge ? Math.min(fit, edge / 4) : fit);
  const sourceWidth = fit * horizontal;
  const sourceHeight = fit * vertical;
  return {
    left: (videoWidth - sourceWidth) / 2, top: (videoHeight - sourceHeight) / 2,
    sourceWidth, sourceHeight, width: horizontal * scale, height: vertical * scale,
  };
}

const edgeValue = () => Number(byId("edge").value);
const fpsValue = () => Number(byId("fps").value);

function showSettingsValues() {
  const edge = edgeValue();
  const crop = frameCrop(edge);
  const longSide = crop ? Math.max(crop.width, crop.height) : edge;
  byId("quality-value").textContent = longSide ? longSide.toLocaleString("ja-JP") + " px" : "原寸";
  byId("fps-value").textContent = fpsValue() + " fps";
  const size = byId("output-size");
  size.hidden = !crop;
  if (crop) size.textContent = "保存サイズ " + crop.width + " × " + crop.height + " px　PNG";
}

for (const event of ["loadeddata", "canplay", "playing", "resize"]) camera.addEventListener(event, showLiveFrame);
portrait.addEventListener("change", showSettingsValues);
byId("edge").addEventListener("change", () => {
    if (captureBatch?.recording) captureBatch.edge = edgeValue();
  showSettingsValues();
});
byId("fps").addEventListener("change", () => {
  if (captureBatch?.recording) {
    captureBatch.interval = 1000 / fpsValue();
    captureBatch.next = performance.now();
  }
  showSettingsValues();
});
byId("settings-toggle").addEventListener("click", () => {
  cameraSettings.showModal();
  showSettingsValues();
  renderProgress();
});
byId("done-settings").addEventListener("click", () => cameraSettings.close());

const finite = (value) => typeof value === "number" && Number.isFinite(value) ? value : null;
function recordOrientation(event) {
  const alpha = finite(event.alpha);
  const beta = finite(event.beta);
  const gamma = finite(event.gamma);
  const compass = finite(event.webkitCompassHeading);
  if (alpha === null && beta === null && gamma === null && compass === null) return;
  const heading = compass !== null && compass >= 0 ? compass
    : event.absolute && alpha !== null ? (360 - alpha + 360) % 360 : null;
  orientation = { heading_deg: heading, alpha_deg: alpha, beta_deg: beta, gamma_deg: gamma,
    absolute: Boolean(event.absolute || heading !== null), compass_accuracy_deg: finite(event.webkitCompassAccuracy),
    screen_angle_deg: finite(screen.orientation?.angle ?? window.orientation) };
  orientation.time = performance.now();
  const reading = heading === null ? "方角未取得" : "方角 " + heading.toFixed(0) + "°";
  byId("motion-status").textContent = reading + (beta === null ? "" : " ／ 前後傾き " + beta.toFixed(0) + "°");
}
byId("motion").addEventListener("click", async () => {
  const status = byId("motion-status");
  status.hidden = false;
  try {
    const eventType = window.DeviceOrientationEvent;
    if (!eventType) throw new Error("この端末では姿勢を取得できません。");
    if (typeof eventType.requestPermission === "function" && await eventType.requestPermission() !== "granted") {
      throw new Error("姿勢の記録にはモーションの許可が必要です。");
    }
    window.addEventListener("deviceorientation", recordOrientation);
    motionEnabled = true;
    status.textContent = "方角・傾きの情報を待っています。";
    controls();
  } catch (error) {
    status.textContent = error.message;
  }
});

async function sendJob(transfer, job) {
  const activeBatch = job.batch;
  if (!activeBatch.id) {
    const { kind, label, settings } = activeBatch;
    Object.assign(activeBatch, await api("/api/session", { kind, label, settings }));
    renderProgress();
  }
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/file");
    request.responseType = "json";
    request.setRequestHeader("X-Pair-Code", code);
    request.setRequestHeader("X-Session", activeBatch.id);
    request.setRequestHeader("X-Index", String(job.index));
    if (job.info) request.setRequestHeader("X-Frame-Info", JSON.stringify(job.info));
    else request.setRequestHeader("X-File-Name", encodeURIComponent(job.file.name));
    if (transfer.kind === "upload") {
      request.upload.onprogress = (event) => {
        transfer.currentBytes = event.loaded;
        renderProgress();
      };
    }
    request.onload = () => {
      if (request.status >= 200 && request.status < 300 && request.response?.name) resolve();
      else reject(new Error(request.response?.error || "PCへ保存できませんでした。"));
    };
    request.onerror = () => reject(new Error("PCとの接続が切れました。"));
    request.onabort = () => reject(new Error("送信が中断されました。"));
    request.send(job.file);
  });
}

function finished(transfer) {
  const activeBatch = transfer.kind === "capture" ? captureBatch : uploadBatch;
  if (!activeBatch || transfer.failed || transfer.sending || transfer.queue.length
      || activeBatch.recording || activeBatch.cancelled) return;
  const amount = activeBatch.saved.toLocaleString();
  message(activeBatch.kind === "capture"
    ? amount + "枚のPNGをPCに保存しました。"
    : amount + "件のファイルをPCに保存しました。", false, transfer.kind);
  if (activeBatch.kind === "upload") {
    byId("files").value = "";
    displayFiles();
  }
}

async function drain(transfer) {
  if (transfer.sending || transfer.failed || !transfer.queue.length) return;
  transfer.sending = true;
  renderProgress();
  controls();
  let photoEncodingFailed = false;
  try {
    while (transfer.queue.length) {
      const job = transfer.queue[0];
      const activeBatch = job.batch;
      if (job.png) {
        job.file = await job.png;
        job.png = undefined;
      }
      if (!job.file) {
        transfer.queue.shift();
        if (job.info.capture_type === "photo") {
          photoEncodingFailed = true;
          message("写真をPNGにできませんでした。もう一度撮ってください。", true, "capture");
        } else {
          activeBatch.dropped += 1;
        }
        renderProgress();
        controls();
        continue;
      }
      await sendJob(transfer, job);
      transfer.queue.shift();
      activeBatch.saved += 1;
      if (transfer.kind === "upload") {
        activeBatch.sentBytes += job.file.size;
        transfer.currentBytes = 0;
      } else if (job.info.capture_type === "video_frame") {
        activeBatch.frames += 1;
        if (activeBatch.frames === 1) activeBatch.firstFrame = job.takenAt;
        activeBatch.lastFrame = job.takenAt;
      }
      renderProgress();
      controls();
    }
  } catch (error) {
    transfer.failed = true;
    transfer.currentBytes = 0;
    if (transfer.kind === "capture" && captureBatch?.recording) stopCapture(captureBatch);
    message(error.message + " 接続を確認して再送できます。", true, transfer.kind);
  } finally {
    transfer.sending = false;
    if (!photoEncodingFailed) finished(transfer);
    renderProgress();
    controls();
  }
}

function newCaptureBatch(settings, takenAt) {
  captureBatch = { kind: "capture", label: byId("capture-label").value, settings, previewStream: stream,
    start: takenAt, saved: 0, frames: 0, nextIndex: 0, dropped: 0, firstFrame: 0, lastFrame: 0 };
  return captureBatch;
}

function stopCapture(activeBatch) {
  activeBatch.recording = false;
  cancelAnimationFrame(activeBatch.tick);
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = undefined;
  }
  if (captureTransfer.queue.length && !captureTransfer.failed) {
    message("撮影を停止しました。残りを送信しています。", false, "capture");
  }
  finished(captureTransfer);
  renderProgress();
  controls();
}

function captureInfo(crop, type, now, started = now, videoSeconds = null) {
  const info = { capture_type: type, captured_at: new Date().toISOString(),
    elapsed_ms: Math.max(0, Math.round(now - started)), video_seconds: videoSeconds,
    width: crop.width, height: crop.height, orientation: null };
  if (orientation) {
    const { time, ...readings } = orientation;
    info.orientation = { ...readings, sample_age_ms: Math.round(now - time) };
  }
  return info;
}

function capturePng(target, context, crop) {
  if (target.width !== crop.width || target.height !== crop.height) {
    target.width = crop.width;
    target.height = crop.height;
    context.imageSmoothingQuality = "high";
  }
  context.drawImage(camera, crop.left, crop.top, crop.sourceWidth, crop.sourceHeight,
    0, 0, crop.width, crop.height);
  return new Promise((resolve) => target.toBlob(resolve, "image/png")).catch(() => null);
}

function enqueueCapture(activeBatch, png, info, takenAt) {
  captureTransfer.queue.push({ batch: activeBatch, index: ++activeBatch.nextIndex, png, info, takenAt });
  renderProgress();
  controls();
  drain(captureTransfer);
}

function schedule(activeBatch) {
  const sample = (now) => {
    if (!activeBatch.recording) return;
    activeBatch.tick = requestAnimationFrame(sample);
    if (now < activeBatch.next) return;
    const due = Math.floor((now - activeBatch.next) / activeBatch.interval) + 1;
    activeBatch.next += due * activeBatch.interval;
    activeBatch.dropped += due - 1;
    // Keep queue space for manual photos when the network falls behind.
    if (frameEncoding || captureTransfer.queue.length >= queueLimit - photoSlots || !hasCameraFrame()) {
      activeBatch.dropped += 1;
      renderProgress();
      return;
    }
    const crop = frameCrop(activeBatch.edge);
    const info = captureInfo(crop, "video_frame", now, activeBatch.start, camera.currentTime);
    try {
      const png = capturePng(canvas, drawing, crop);
      frameEncoding = true;
      enqueueCapture(activeBatch, png.finally(() => { frameEncoding = false; }), info, now);
    } catch {
      activeBatch.dropped += 1;
      renderProgress();
    }
  };
  activeBatch.tick = requestAnimationFrame(sample);
}

byId("capture-action").addEventListener("click", () => {
  if (captureBatch?.recording) {
    stopCapture(captureBatch);
    return;
  }
  if (captureTransfer.failed) {
    retryBatch(captureTransfer);
    return;
  }
  if (cameraState === "retry") {
    openCamera();
    return;
  }
  if (cameraState === "play-needed") {
    playPreview();
    return;
  }
  if (!hasCameraFrame() || document.hidden) return;
  const now = performance.now();
  const settings = { edge: edgeValue(), fps: fpsValue() };
  let activeBatch = captureBatchFor(stream);
  if (!activeBatch || activeBatch.startedRecording || activeBatch.settings.edge !== settings.edge
      || activeBatch.settings.fps !== settings.fps) {
    activeBatch = newCaptureBatch(settings, now);
  }
  Object.assign(activeBatch, { edge: settings.edge, next: now, interval: 1000 / settings.fps,
    startedRecording: true, recording: true });
  message("撮影中", false, "capture");
  if (navigator.wakeLock) {
    navigator.wakeLock.request("screen").then((lock) => {
      if (activeBatch.recording) wakeLock = lock;
      else lock.release().catch(() => {});
    }).catch(() => {});
  }
  schedule(activeBatch);
  renderProgress();
  controls();
});

function captureBatchFor(previewStream) {
  if (captureBatch && captureBatch.previewStream === previewStream && !captureBatch.cancelled && !captureBatch.sealed) {
    return captureBatch;
  }
  return undefined;
}

byId("still-action").addEventListener("click", () => {
  if (!hasCameraFrame() || byId("still-action").disabled) return;
  try {
    const now = performance.now();
    const settings = { edge: edgeValue(), fps: fpsValue() };
    const crop = frameCrop(settings.edge);
    const activeBatch = captureBatchFor(stream) ?? newCaptureBatch(settings, now);
    const info = captureInfo(crop, "photo", now, activeBatch.start,
      activeBatch.recording ? camera.currentTime : null);
    const png = capturePng(stillCanvas, stillDrawing, crop);
    photoEncoding = true;
    message("", false, "capture");
    enqueueCapture(activeBatch, png.finally(() => {
      photoEncoding = false;
      controls();
    }), info, now);
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      byId("shutter-flash").animate?.([{ opacity: .56 }, { opacity: 0 }], { duration: 150 });
    }
  } catch (error) {
    message(error.message, true, "capture");
    controls();
  }
});

function displayFiles() {
  const files = [...byId("files").files];
  const list = byId("file-list");
  list.hidden = !files.length;
  list.replaceChildren(...files.slice(0, 6).map((file) => {
    const row = document.createElement("li");
    row.textContent = file.name;
    return row;
  }));
  byId("selection").textContent = files.length ? files.length.toLocaleString() + "件を選択"
    : "PNG・JPEG・HEIC・HEIF・MOV・MP4";
  controls();
}
byId("files").addEventListener("change", displayFiles);
byId("send").addEventListener("click", () => {
  const files = [...byId("files").files];
  if (uploadTransfer.queue.length || !files.length) return;
  uploadBatch = { kind: "upload", label: byId("upload-label").value, saved: 0, sentBytes: 0,
    hasVideo: files.some((file) => /\.(mov|mp4)$/i.test(file.name)),
    totalBytes: files.reduce((sum, file) => sum + file.size, 0) };
  trainingError = "";
  uploadTransfer.queue = files.map((file, index) => ({ batch: uploadBatch, file, index: index + 1 }));
  message(files.length.toLocaleString() + "件をPCに送信しています。", false, "upload");
  drain(uploadTransfer);
});

function retryBatch(transfer) {
  if (!transfer.failed) return;
  transfer.failed = false;
  message("未送信のファイルを送っています。", false, transfer.kind);
  drain(transfer);
}
byId("retry").addEventListener("click", () => retryBatch(transfers[mode]));
byId("discard").addEventListener("click", () => {
  const transfer = transfers[mode];
  if (!transfer.failed) return;
  const activeBatch = mode === "capture" ? captureBatch : uploadBatch;
  activeBatch.cancelled = true;
  transfer.failed = false;
  transfer.queue.length = 0;
  message("未送信分を取り消しました。");
  renderProgress();
  controls();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && captureBatch?.recording) stopCapture(captureBatch);
  if (!document.hidden && stream && camera.paused) playPreview();
  if (!document.hidden) refreshTraining();
});
window.addEventListener("pagehide", () => closeCamera());
byId("stop").addEventListener("click", async () => {
  try {
    await api("/api/stop", {});
    server = undefined;
    closeCamera();
    byId("connection").textContent = "サーバー停止済み";
    byId("connection").dataset.online = "false";
    byId("stop").hidden = true;
    byId("workspace").hidden = true;
    byId("desktop-panel").hidden = true;
  } catch (error) {
    message(error.message, true);
  }
});
showMode(mode);
initialize();
