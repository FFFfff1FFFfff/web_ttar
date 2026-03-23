# AR Tattoo

A browser-based AR tattoo preview tool. Upload a tattoo design and see it rendered on your arm in real time using your webcam.

## Features

- **Real-time pose tracking** via MediaPipe Pose Landmarker (GPU-accelerated)
- **Body part selection** — switch between forearm and upper arm
- **Position slider** — move the tattoo along the limb
- **Size control** — set tattoo width in centimeters
- **Custom designs** — upload any image as a tattoo
- **Stable scaling** — uses torso measurements for distance-independent sizing
- **Smooth tracking** — One Euro Filter reduces jitter while keeping low latency

## Usage

Serve the project with any static file server:

```bash
npx serve .
```

Then open the URL in a browser with webcam access (Chrome/Edge recommended for best WebGPU support).

1. Allow camera access when prompted
2. The default tattoo appears on both arms automatically
3. Use **Upload Design** to load your own tattoo image
4. Adjust **Width** (cm) to set the real-world size
5. Switch between **Forearm** / **Upper Arm**
6. Drag the **Position** slider to move the tattoo along the limb

## Embed Mode

Add `?embed` to the URL to use as an iframe. The built-in UI is hidden; control everything via `postMessage`.

```html
<iframe src="https://your-domain.com/?embed" allow="camera" />
```

### Parent → iframe

```js
const frame = document.querySelector("iframe");

// Set tattoo image (data URL)
frame.contentWindow.postMessage({ type: "setTattoo", dataUrl: "data:image/png;..." }, "*");

// Set width in cm
frame.contentWindow.postMessage({ type: "setSize", cm: 10 }, "*");

// Set body part: "forearm" | "upperarm"
frame.contentWindow.postMessage({ type: "setBodyPart", part: "forearm" }, "*");

// Set position along limb (0–1)
frame.contentWindow.postMessage({ type: "setPlacement", value: 0.5 }, "*");
```

### iframe → Parent

```js
window.addEventListener("message", (e) => {
  if (e.data.type === "ready") { /* AR is running */ }
  if (e.data.type === "error") { /* e.data.message */ }
});
```

## Tech Stack

- MediaPipe Pose Landmarker (vision tasks WASM/GPU)
- Canvas 2D rendering
- One Euro Filter for landmark smoothing
- Vanilla JS, no build step
