import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// --- Config ---
const EMA_ALPHA = 0.4; // smoothing factor: lower = smoother but laggier
const TATTOO_OPACITY = 0.75;
const TATTOO_WIDTH_RATIO = 0.55; // tattoo width relative to forearm length

// --- DOM ---
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

// --- State ---
let poseLandmarker = null;
let tattooImg = null;
// Smoothed landmarks (EMA) for left and right forearms
let smoothed = {
  left: { elbow: null, wrist: null },
  right: { elbow: null, wrist: null },
};

// --- Load tattoo image ---
function loadTattooImage() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => {
      // Generate a simple fallback tattoo if tattoo.png is missing
      const c = document.createElement("canvas");
      c.width = 200;
      c.height = 100;
      const cx = c.getContext("2d");
      // Draw a simple tribal-style shape
      cx.fillStyle = "#000";
      cx.beginPath();
      cx.ellipse(100, 50, 80, 30, 0, 0, Math.PI * 2);
      cx.fill();
      cx.clearRect(40, 20, 120, 60);
      cx.beginPath();
      cx.ellipse(100, 50, 60, 18, 0, 0, Math.PI * 2);
      cx.fill();
      // Convert to image
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
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
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
  // Match canvas size to video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// --- EMA smoothing ---
function ema(prev, curr, alpha) {
  if (!prev) return { ...curr };
  return {
    x: prev.x + alpha * (curr.x - prev.x),
    y: prev.y + alpha * (curr.y - prev.y),
    z: prev.z + alpha * (curr.z - prev.z),
    visibility: curr.visibility,
  };
}

// --- Draw tattoo on one forearm ---
function drawTattoo(elbow, wrist) {
  // Both landmarks must be visible enough
  if (elbow.visibility < 0.5 || wrist.visibility < 0.5) return;

  const ex = elbow.x * canvas.width;
  const ey = elbow.y * canvas.height;
  const wx = wrist.x * canvas.width;
  const wy = wrist.y * canvas.height;

  // Forearm vector
  const dx = wx - ex;
  const dy = wy - ey;
  const forearmLen = Math.sqrt(dx * dx + dy * dy);
  if (forearmLen < 20) return; // too small, skip

  const angle = Math.atan2(dy, dx);

  // Tattoo dimensions
  const tattooW = forearmLen * 0.7;
  const tattooH = tattooW * (tattooImg.height / tattooImg.width);

  // Center of forearm
  const cx = (ex + wx) / 2;
  const cy = (ey + wy) / 2;

  ctx.save();
  ctx.globalAlpha = TATTOO_OPACITY;
  ctx.globalCompositeOperation = "multiply";

  // Transform: translate to center, rotate to forearm angle
  ctx.translate(cx, cy);
  ctx.rotate(angle);

  // Draw tattoo centered
  ctx.drawImage(tattooImg, -tattooW / 2, -tattooH / 2, tattooW, tattooH);

  ctx.restore();
}

// --- Main render loop ---
let lastTime = 0;

function renderLoop() {
  const now = performance.now();
  // Throttle to avoid calling detectForVideo with same timestamp
  if (now - lastTime < 16) {
    requestAnimationFrame(renderLoop);
    return;
  }

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw camera frame to canvas (for multiply blend to work on)
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (poseLandmarker && video.readyState >= 2) {
    const results = poseLandmarker.detectForVideo(video, now);
    lastTime = now;

    if (results.landmarks && results.landmarks.length > 0) {
      const lm = results.landmarks[0];

      // MediaPipe Pose landmark indices:
      // 13=left elbow, 15=left wrist, 14=right elbow, 16=right wrist
      const rawLeftElbow = lm[13];
      const rawLeftWrist = lm[15];
      const rawRightElbow = lm[14];
      const rawRightWrist = lm[16];

      // Apply EMA smoothing
      smoothed.left.elbow = ema(smoothed.left.elbow, rawLeftElbow, EMA_ALPHA);
      smoothed.left.wrist = ema(smoothed.left.wrist, rawLeftWrist, EMA_ALPHA);
      smoothed.right.elbow = ema(smoothed.right.elbow, rawRightElbow, EMA_ALPHA);
      smoothed.right.wrist = ema(smoothed.right.wrist, rawRightWrist, EMA_ALPHA);

      // Draw tattoo on both forearms
      drawTattoo(smoothed.left.elbow, smoothed.left.wrist);
      drawTattoo(smoothed.right.elbow, smoothed.right.wrist);
    }
  }

  requestAnimationFrame(renderLoop);
}

// --- Start ---
async function main() {
  try {
    statusEl.textContent = "正在加载纹身图案...";
    tattooImg = await loadTattooImage();

    statusEl.textContent = "正在初始化摄像头...";
    await initCamera();

    statusEl.textContent = "正在加载 AI 模型...";
    await initPose();

    statusEl.textContent = "就绪 ✓";
    setTimeout(() => (statusEl.style.opacity = "0"), 1500);

    renderLoop();
  } catch (err) {
    statusEl.textContent = "错误: " + err.message;
    console.error(err);
  }
}

main();
