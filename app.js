import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// --- Config ---
const TATTOO_OPACITY = 0.75;
const MIN_FOREARM_PX = 20;
const VISIBILITY_THRESH = 0.5;
// One Euro Filter params
const OEF_MIN_CUTOFF = 0.008; // lower = smoother when still
const OEF_BETA = 0.005;       // higher = less lag when moving
const OEF_ANGLE_CUTOFF = 0.003; // extra-smooth angle filtering
const OEF_ANGLE_BETA = 0.003;
const OEF_SCALE_CUTOFF = 0.005; // extra-heavy smoothing for pxPerM
const OEF_SCALE_BETA = 0.005;
// Outlier gating: max px jump per second before rejecting a frame
const OUTLIER_MAX_SPEED = 800; // px/s
// Dead zone: ignore movements smaller than this (pixels)
const DEAD_ZONE_PX = 2.5;

// --- Embed mode ---
const EMBED = new URLSearchParams(location.search).has("embed");
if (EMBED) document.body.classList.add("embed");

// --- DOM ---
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

// --- State ---
let poseLandmarker = null;
let tattooImg = null;
let tattooWidthCm = 8; // default tattoo width in cm
let bodyPart = "forearm"; // "forearm" or "upperarm"
let placement = 0.5; // 0–1 along the segment (A→B)

// Landmark indices: [pointA, pointB] for left and right
const BODY_PARTS = {
  forearm:  { left: [13, 15], right: [14, 16] }, // elbow→wrist
  upperarm: { left: [11, 13], right: [12, 14] }, // shoulder→elbow
};

// --- One Euro Filter ---
// Attempt to use a widely-known low-latency smoothing filter
// References: http://cristal.univ-lille.fr/~casiez/1euro/
function createOneEuroFilter(minCutoff = OEF_MIN_CUTOFF, beta = OEF_BETA) {
  let xPrev = null, dxPrev = 0, tPrev = null;
  const alpha = (dt, cutoff) => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  };
  return {
    filter(x, t) {
      if (tPrev === null) { xPrev = x; tPrev = t; return x; }
      const dt = Math.max(t - tPrev, 1e-6);
      // Derivative
      const dx = (x - xPrev) / dt;
      const aDeriv = alpha(dt, 1.0);
      const dxSmoothed = aDeriv * dx + (1 - aDeriv) * dxPrev;
      // Adaptive cutoff
      const cutoff = minCutoff + beta * Math.abs(dxSmoothed);
      const a = alpha(dt, cutoff);
      const xSmoothed = a * x + (1 - a) * xPrev;
      xPrev = xSmoothed; dxPrev = dxSmoothed; tPrev = t;
      return xSmoothed;
    },
    reset() { xPrev = null; dxPrev = 0; tPrev = null; },
  };
}

// Per-arm smoothed geometry + shared torso scale
const filters = {
  left: {
    cx: createOneEuroFilter(), cy: createOneEuroFilter(),
    sinA: createOneEuroFilter(OEF_ANGLE_CUTOFF, OEF_ANGLE_BETA),
    cosA: createOneEuroFilter(OEF_ANGLE_CUTOFF, OEF_ANGLE_BETA),
    prevCx: null, prevCy: null, prevT: null,
    lockedCx: null, lockedCy: null,
  },
  right: {
    cx: createOneEuroFilter(), cy: createOneEuroFilter(),
    sinA: createOneEuroFilter(OEF_ANGLE_CUTOFF, OEF_ANGLE_BETA),
    cosA: createOneEuroFilter(OEF_ANGLE_CUTOFF, OEF_ANGLE_BETA),
    prevCx: null, prevCy: null, prevT: null,
    lockedCx: null, lockedCy: null,
  },
  pxPerM: createOneEuroFilter(OEF_SCALE_CUTOFF, OEF_SCALE_BETA),
};

function resetArmFilters() {
  for (const side of [filters.left, filters.right]) {
    side.cx.reset(); side.cy.reset();
    side.sinA.reset(); side.cosA.reset();
    side.prevCx = null; side.prevCy = null; side.prevT = null;
    side.lockedCx = null; side.lockedCy = null;
  }
}

// --- Load tattoo image ---
function loadTattooImage() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => {
      const c = document.createElement("canvas");
      c.width = 200; c.height = 100;
      const cx = c.getContext("2d");
      cx.fillStyle = "#000";
      cx.beginPath(); cx.ellipse(100, 50, 80, 30, 0, 0, Math.PI * 2); cx.fill();
      cx.clearRect(40, 20, 120, 60);
      cx.beginPath(); cx.ellipse(100, 50, 60, 18, 0, 0, Math.PI * 2); cx.fill();
      const fallback = new Image();
      fallback.onload = () => resolve(fallback);
      fallback.src = c.toDataURL();
    };
    img.src = "tattoo.png";
  });
}

// --- Init MediaPipe Pose ---
async function initPose() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

// --- Init camera ---
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadedmetadata = r));
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// --- Draw tattoo on one arm ---
function drawTattoo(ptA, ptB, f, t, pxPerM) {
  if (ptA.visibility < VISIBILITY_THRESH || ptB.visibility < VISIBILITY_THRESH) return;

  const ax = ptA.x * canvas.width, ay = ptA.y * canvas.height;
  const bx = ptB.x * canvas.width, by = ptB.y * canvas.height;
  if (Math.hypot(bx - ax, by - ay) < MIN_FOREARM_PX) return;

  const rawCx = ax + (bx - ax) * placement;
  const rawCy = ay + (by - ay) * placement;

  // P2: Outlier gating — reject single-frame jumps
  if (f.prevCx !== null) {
    const dt = Math.max(t - f.prevT, 1e-6);
    const speed = Math.hypot(rawCx - f.prevCx, rawCy - f.prevCy) / dt;
    if (speed > OUTLIER_MAX_SPEED) return; // skip this frame
  }
  f.prevCx = rawCx; f.prevCy = rawCy; f.prevT = t;

  let cx = f.cx.filter(rawCx, t);
  let cy = f.cy.filter(rawCy, t);

  // Dead zone: lock position when movement is tiny
  if (f.lockedCx !== null && Math.hypot(cx - f.lockedCx, cy - f.lockedCy) < DEAD_ZONE_PX) {
    cx = f.lockedCx; cy = f.lockedCy;
  } else {
    f.lockedCx = cx; f.lockedCy = cy;
  }

  // P0: Filter angle via sin/cos to avoid ±π discontinuity
  const rawAngle = Math.atan2(by - ay, bx - ax);
  const angle = Math.atan2(
    f.sinA.filter(Math.sin(rawAngle), t),
    f.cosA.filter(Math.cos(rawAngle), t)
  );

  const tattooW = (tattooWidthCm / 100) * pxPerM;
  const tattooH = tattooW * (tattooImg.height / tattooImg.width);

  ctx.save();
  ctx.globalAlpha = TATTOO_OPACITY;
  ctx.globalCompositeOperation = "multiply";
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  ctx.drawImage(tattooImg, -tattooW / 2, -tattooH / 2, tattooW, tattooH);
  ctx.restore();
}

// --- Main render loop ---
let lastTime = 0;

function renderLoop() {
  const now = performance.now();
  if (now - lastTime < 16) { requestAnimationFrame(renderLoop); return; }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (poseLandmarker && video.readyState >= 2) {
    const results = poseLandmarker.detectForVideo(video, now);
    lastTime = now;

    if (results.landmarks && results.landmarks.length > 0 && results.worldLandmarks && results.worldLandmarks.length > 0) {
      const lm = results.landmarks[0];
      const wl = results.worldLandmarks[0];
      const t = now / 1000;

      // Use torso (right shoulder→right hip) for pxPerM: always faces camera
      // 2D: landmarks 12 (R shoulder) → 24 (R hip)
      // 3D: worldLandmarks same indices
      const torso2d = Math.hypot(
        (lm[12].x - lm[24].x) * canvas.width,
        (lm[12].y - lm[24].y) * canvas.height
      );
      const torso3d = Math.hypot(wl[12].x - wl[24].x, wl[12].y - wl[24].y, wl[12].z - wl[24].z);
      const rawPxPerM = torso2d / Math.max(torso3d, 0.01);
      const pxPerM = filters.pxPerM.filter(rawPxPerM, t);

      const bp = BODY_PARTS[bodyPart];
      drawTattoo(lm[bp.left[0]], lm[bp.left[1]], filters.left, t, pxPerM);
      drawTattoo(lm[bp.right[0]], lm[bp.right[1]], filters.right, t, pxPerM);
    }
  }

  requestAnimationFrame(renderLoop);
}

// --- UI handlers ---
document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => { tattooImg = img; };
  img.src = URL.createObjectURL(file);
});

document.getElementById("size-input").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  if (v > 0) tattooWidthCm = v;
});

document.getElementById("part-select").addEventListener("change", (e) => {
  bodyPart = e.target.value;
  // Reset filters when switching body part
  resetArmFilters();
});

document.getElementById("placement-input").addEventListener("input", (e) => {
  placement = parseFloat(e.target.value);
});

// --- postMessage API (embed mode) ---
if (EMBED) {
  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || !d.type) return;
    if (d.type === "setTattoo") {
      const img = new Image();
      img.onload = () => { tattooImg = img; };
      img.src = d.dataUrl;
      if (d.widthInch > 0) tattooWidthCm = d.widthInch * 2.54;
      else if (d.widthCm > 0) tattooWidthCm = d.widthCm;
    } else if (d.type === "setSize") {
      if (d.cm > 0) tattooWidthCm = d.cm;
    } else if (d.type === "setBodyPart" && BODY_PARTS[d.part]) {
      bodyPart = d.part;
      resetArmFilters();
    } else if (d.type === "setPlacement") {
      placement = parseFloat(d.value);
    }
  });
}

// --- Start ---
async function main() {
  try {
    statusEl.textContent = "Loading tattoo...";
    tattooImg = await loadTattooImage();
    statusEl.textContent = "Starting camera...";
    await initCamera();
    statusEl.textContent = "Loading AI model...";
    await initPose();
    statusEl.textContent = "Ready";
    setTimeout(() => (statusEl.style.opacity = "0"), 1500);
    if (EMBED) window.parent.postMessage({ type: "ready" }, "*");
    renderLoop();
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
    if (EMBED) window.parent.postMessage({ type: "error", message: err.message }, "*");
    console.error(err);
  }
}

main();
