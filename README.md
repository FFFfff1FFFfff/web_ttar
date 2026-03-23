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

## Tech Stack

- MediaPipe Pose Landmarker (vision tasks WASM/GPU)
- Canvas 2D rendering
- One Euro Filter for landmark smoothing
- Vanilla JS, no build step
