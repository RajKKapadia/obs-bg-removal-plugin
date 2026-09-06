# OBS background removal: RMBG-1.4 and RVM

A native C++ OBS **Effect Filter** that runs BRIA RMBG-1.4 and Robust Video Matting
(RVM MobileNetV3) locally with ONNX Runtime.
The initial supported platform is native OBS on **Linux x86-64**, with SDR sources.
This is a personal prototype; model weights have BRIA's separate usage terms.

## Try it in OBS

1. Build and install using the commands below, then restart OBS.
2. Right-click your webcam source → **Filters**.
3. Under **Effect Filters**, click **+** → **Background Removal (RMBG / RVM)**.
4. The model is selected automatically when installed with `RMBG_INSTALL_MODEL=ON`.
   Otherwise select `data/models/rmbg-1.4.onnx` from this checkout.
5. Leave **Inference device** on **Automatic**, or choose **CUDA GPU** to require it.
6. Click **Refresh status / retry model** after loading. Confirm **Ready: CUDA**.
7. Add a Color Source or image **below** your webcam to clearly see the transparency.

Start with maximum mask updates **15**, temporal smoothing **0**, threshold **0.5**,
and edge softness **0.5**. These are the defaults for new filters. Existing filters can
retain saved values: set smoothing to **0** manually for the most responsive edges.
Try **30** mask updates for more frequent cutout updates if the GPU has spare capacity;
check mask age and OBS rendering lag, since contention can increase delay.
The update setting is a ceiling, not guaranteed throughput. Try smoothing **0.05–0.15**
if the edges flicker, at the cost of some responsiveness. Increase threshold to
remove more foreground; reduce it to keep more. Reduce softness for firmer edges.
**Show mask instead of video** displays white foreground and black background.

The original source, including its background, stays visible while loading, on errors,
or when the last mask is older than the configured timeout. The filter is not a privacy
barrier. A missing model can be fixed by selecting the correct file and clicking retry.

## Use RVM MobileNetV3

RVM is the human video-matting model available in the author's TensorFlow.js/WebGL demo.
This native plugin uses the author's **ONNX export** with CUDA or CPU. Model formats for
TensorFlow.js cannot be loaded directly into ONNX Runtime.

1. Download the official models: `python3 scripts/download-rvm.py`.
2. Build this version and restart OBS after installing it. Existing filter instances and
   their saved settings are retained; the registered filter ID is unchanged.
3. In **Model file**, select `rvm_mobilenetv3_fp16.onnx` for CUDA, or
   `rvm_mobilenetv3_fp32.onnx` for CUDA/CPU. The model signature is detected automatically.
4. Start with **30** mask updates, **0** smoothing, **0.5** threshold, **0.5** softness,
   **1280** maximum input long edge, and **Automatic** downsample ratio.
5. Refresh status: it should show **Ready: CUDA; model: RVM FP16** and increasing
   **recurrent frames (state on GPU)**. These are snapshots refreshed by the button.

RVM capture preserves aspect ratio and never upscales the source. The maximum input size
limits GPU capture/readback work; the separate downsample ratio controls RVM's coarse
processing stage before refinement. Automatic uses an internal long edge of approximately
480 pixels: for a 1280×720 capture it selects 0.375. Try a smaller ratio for less processing
or a larger one for different framing, such as full-body shots; larger is not always better.
The RVM controls have no effect on RMBG's fixed 1024×1024 input.

The four recurrent states are carried across processed frames and remain on CUDA when
CUDA is selected. They reset on source/capture dimension changes, model/settings reload,
non-increasing timestamps, or a gap of at least one second. Overload replaces the single
waiting frame rather than growing a video queue. Keep extra temporal smoothing at zero
initially because RVM already has temporal memory.

RVM's `pha` output is used directly as alpha, without RMBG's per-frame min/max normalization.
Default threshold/softness preserve this alpha; adjusted controls remap it linearly.
The current source supplies RGB; the model's estimated `fgr` RGB is not used. This keeps
the visible video current while the newest completed alpha is applied. Outlines can still
trail motion because capture, processing, and display take time.

To include both downloaded RVM files in an install, configure with
`-DRVM_INSTALL_MODELS=ON` in addition to the other build options. This option defaults off.
The download script pins release **v1.0.0** and checks SHA256; provenance and the upstream
GPL-3.0 license are in `data/licenses/rvm/` and are included with the plugin data.
See the [official RVM project](https://github.com/PeterL1n/RobustVideoMatting) and its
[inference documentation](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md).

## Build (Linux)

Required: CMake 3.24+, a C++17 compiler, OBS 30+ development files, libpng development
files, and Python 3.11+ with tarfile's safe `data` extraction filter (Python 3.12 on Mint 22).
Common Ubuntu packages: `build-essential cmake libobs-dev libpng-dev libsimde-dev`.
The installed OBS application and development libraries should come from the same source.

CPU-only build:

```sh
python3 scripts/bootstrap.py --backend cpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

CUDA build used on the development machine:

```sh
python3 scripts/bootstrap.py --backend cuda --simde
python3 scripts/prepare-cuda.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="$PWD/.deps/onnxruntime-linux-x64-gpu_cuda12-1.29.0" \
  -DRMBG_CUDA_LIBRARY_DIR="$PWD/.deps/cuda/lib" \
  -DRMBG_INSTALL_MODEL=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

`--simde` extracts missing SIMDe headers into `.deps/sysroot`, without installing
system packages. Omit it when the headers are installed already.
`prepare-cuda.py` verifies and reuses matching NVIDIA wheels from the local uv cache
when present, or downloads pinned wheels from PyPI. `--cache-only` disables downloads.
These private libraries avoid the older CUDA 12.0 libraries supplied by Mint 22.
The NVIDIA driver still needs to work (`nvidia-smi`). No global CUDA installation is changed.

ONNX Runtime is pinned to **1.29.0**. CUDA dependencies are pinned to runtime **12.9.79**,
cuBLAS **12.9.1.4**, and cuDNN **9.10.2.21**. Downloaded libraries and models are ignored
by Git. The GPU SDK plus private libraries use several GB of disk space.

## Install for your user

For the native Linux OBS application:

```sh
cmake --install build --prefix "${XDG_CONFIG_HOME:-$HOME/.config}/obs-studio/plugins"
```

This creates only `obs-rmbg/` inside OBS's user plugin directory. No `sudo` is required.
The installed library uses paths relative to itself, including the matching private CUDA
libraries when configured. Restart OBS after installing or rebuilding.
This package has **not** been prepared for Flatpak OBS, Windows, or macOS.

For an inspectable package before installation:

```sh
cmake --install build --prefix "$PWD/dist"
```

Model installation defaults **off**. `RMBG_INSTALL_MODEL=ON` is intended for your personal
installation, not a public distribution package. Library license notices are installed
under the plugin's `data/licenses/` directory. Keep BRIA's model terms separate.

## Model download

The bootstrap script downloads the original FP32 ONNX file from the official repository:

- [BRIA RMBG-1.4 ONNX files](https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx)
- Destination: `data/models/rmbg-1.4.onnx`
- Revision: `2ceba5a5efaec153162aedea169f76caf9b46cf8`
- SHA256: `8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f`

If access is required, download **model.onnx** in your browser and save it at that
destination. The plugin accepts FP32 or FP16 input/output tensors for the supported
model signatures. Internal model precision can differ from these types: the official
RMBG **model_fp16.onnx** preserves FP32 inputs and outputs. No Python model code or
Hugging Face token is used inside OBS.

The official alternative files at the same revision were downloaded and checksum verified:

| Upstream file | Local filename in `data/models/` | SHA256 |
| --- | --- | --- |
| `model_fp16.onnx` | `rmbg-1.4-fp16.onnx` | `9fdfdb41866d872e0acf4a010c35c1a8547bf0eebe0d1544406bbf1c824cb59d` |
| `model_quantized.onnx` | `rmbg-1.4-quantized.onnx` | `a6648479275dfd0ede0f3a8abc20aa5c437b394681b05e5af6d268250aaf40f3` |

Select a variant using the filter's model file picker; changing the selection reloads
the model automatically. The original model remains the default. The bootstrap and CMake
install commands above still download/install only that original model; copy alternative
files separately or select them directly from this checkout.

On 2026-09-06, the unchanged C++ CLI accepted all three files with CUDA enabled. A short
five-iteration comparison on the RTX 5070 Ti, excluding the first iteration, averaged
**42.4 ms FP32**, **23.4 ms FP16**, and **834.1 ms quantized**. These are inference-pipeline
times from one static fixture under shared machine load, not OBS end-to-end latency.
The quantized run reported 342 graph copy nodes and nodes outside the preferred provider;
the `CUDA` status identifies an enabled provider and does not guarantee all operations
run on the GPU. FP16 is the useful alternative in this comparison.

The installed plugin also passed the real-libobs rendering, transparency, foreground-color,
and missing-model recovery checks with FP16. Its CLI alpha mask differed from FP32 by
an average of **0.0036 on the 0–255 scale** on this fixture. Hair detail and live motion
still need evaluation on representative webcam footage.

BRIA's current [model card](https://huggingface.co/briaai/RMBG-1.4) advertises noncommercial
use. Its linked agreement was unavailable during development; an older agreement limits
use to evaluation. Follow the terms attached to your model access. The code's license
does not grant any additional rights to model weights.

## Test an image without OBS

The CLI uses exactly the same C++ inference and mask normalization as the plugin:

```sh
build/rmbg-image data/models/rmbg-1.4.onnx input.png output.png \
  --device cuda --iterations 5
```

It writes a transparent PNG and `output-mask.png`, and prints per-frame processing time.
The average excludes the first iteration when multiple iterations are requested.
Use `--device cpu` for a CPU test. These timings include tensor preparation, inference,
and mask normalization; they exclude OBS capture, graphics readback, and compositing.

## OBS integration test

Install `xvfb` for an isolated display. Supply an opaque PNG with a clear foreground subject:

```sh
xvfb-run -a build/obs-rmbg-smoke \
  "$PWD/build/obs-rmbg.so" "$PWD/data" \
  "$PWD/data/models/rmbg-1.4.onnx" "$PWD/input.png" "$PWD/obs-result.png" cuda
```

This initializes real libobs/OpenGL, loads the module, renders a source through the
filter, checks for transparent and opaque pixels, changes to a missing model, verifies
opaque passthrough, and destroys everything cleanly. It uses a private scene and does
not change the user's OBS scene collection. It stretches the fixture to a 640×360
canvas for the test; normal source transforms remain controlled by OBS.

Append `30 0 10` to measure 10 seconds at 30 mask updates with smoothing disabled.
The benchmark reports completed mask updates per second, masked video FPS, and mean
displayed mask age (time from source capture to rendering with that mask). It excludes
camera buffering, display latency, and the visual effect of temporal smoothing.

For RVM, use a portrait fixture with an opaque background. Add the opposite model family's
ONNX path after the measurement arguments to also verify live landscape/portrait resizing,
switching from RVM to RMBG, and switching back. The capture and mask textures must resize
correctly at each transition. The isolated scene never alters your OBS scene collection.

The model integration executable separately checks actual recurrence, state residency,
and resets for pauses, source changes, orientation changes, and configuration generations:

```sh
build/rvm-integration data/models/rvm_mobilenetv3_fp16.onnx portrait.png cuda
build/rvm-integration data/models/rvm_mobilenetv3_fp32.onnx portrait.png cpu
```

The image CLI also accepts `--rvm-max-size 1280` and `--rvm-downsample 0` (automatic).
Repeated image iterations reuse RVM state; use independent runs for unrelated still images.

## Implementation

- `src/model.cpp`: detects model signatures, converts premultiplied RGBA to RGB NCHW
  (RMBG: `channel / 255 - 0.5`; RVM: `channel / 255`), handles FP32/FP16 tensors, and
  runs ONNX. RVM recurrent tensors use device I/O binding and its alpha range is preserved.
- `src/worker.cpp`: loads models off the render thread and processes one in-flight
  frame with one replaceable waiting frame. New captures replace older waiting frames;
  the queue never grows. Configuration generations invalidate both pending work and results.
- `src/plugin.cpp`: captures at the model's selected size on the GPU, maps a staging transfer on a later
  video tick while inference runs, uploads masks, and exposes OBS controls and diagnostic
  status, including mask age. Capture scheduling follows the OBS video clock without
  accumulating drift from render-thread timing variations.
- `data/rmbg.effect`: applies the mask to both RGB and alpha for correct premultiplied
  compositing, with threshold, softness, and mask preview controls.

Model inference is asynchronous. The current source is combined with the newest completed
mask; this favors uninterrupted video but can cause outlines to trail fast motion.
Temporal smoothing is adjusted for elapsed time, resets across source-size changes or
long gaps, and cannot eliminate all flicker from an image segmentation model.
RMBG identifies salient objects, so it may preserve chairs and other objects as well as people.
RVM is designed for human matting.

## Current verification

On Linux Mint 22.3, OBS 32.2.0, Intel Core Ultra 7 265K and NVIDIA RTX 5070 Ti:

- C++ build and core checks passed.
- Deterministic worker concurrency checks cover replacement of waiting frames, overlapping
  capture/inference, and model changes while work is in flight.
- Official model download and cached CUDA library checksums verified.
- RMBG FP32 inference: approximately **29–32 ms** per frame on CUDA after warmup;
  approximately **1.2 seconds** on CPU with two inference threads.
- Real libobs rendering, mask transparency, foreground color preservation, and missing-model
  passthrough tested using the installed package in the personal OBS plugin directory.
- GPU compute used the RTX card; virtual-display rendering used Mesa llvmpipe.
- In a 10-second static-image benchmark at 30 video FPS, with the user's OBS also running,
  the previous default pipeline produced **10.1 mask updates/s** with **130 ms** mean
  displayed mask age. Overlapping capture/inference at the same 15/0.15 settings produced
  **15.0 updates/s** with **83 ms** mean age. At 30 mask updates and zero smoothing,
  the new pipeline produced **24.0 updates/s** with **113 ms** mean age; the old pipeline
  produced **10.1 updates/s** with **132 ms** mean age. More frequent updates can increase
  GPU contention; 15 remains the conservative default. Output pixels were identical
  for the static fixture. These are shared-GPU observations, not a live-camera latency guarantee.
- RVM FP16, installed v0.2.0: approximately **7–9 ms** processing on the 512×600 portrait
  fixture after warmup, **30 mask updates/s**, and **66.7 ms** average displayed mask age
  in a five-second OBS test at 30 video FPS. A larger 850×1280 input took approximately
  **12–14 ms** in the short CLI check. Capture/display timing remains part of the delay.
- RVM FP16/FP32 CUDA and FP32 CPU passed actual recurrent-state reuse/reset checks.
  The OBS test passed source resizing in both orientations, RVM → RMBG → RVM switching,
  transparency, color preservation, and missing-model passthrough. Both RMBG precisions
  produced pixel-identical CLI output to their pre-RVM baselines.
- Live webcam motion, recording load, and additional OBS plugin combinations still need
  testing on your actual scenes. The inference result is not an end-to-end 30 FPS guarantee.

## Troubleshooting

- **Filter missing:** restart OBS, check the native/Flatpak installation distinction,
  and inspect Help → Log Files → View Current Log for `obs-rmbg`.
- **Ready: CPU:** automatic CUDA initialization fell back. Rebuild with the CUDA SDK and
  private libraries above. Explicit CUDA mode surfaces initialization errors.
- **cuDNN unavailable / old libcudart symbol error:** use the complete private CUDA build;
  the CUDA SDK by itself does not supply all matching NVIDIA runtime libraries.
- **Error after inference starts:** the original source is shown. Fix the runtime/model
  issue and retry, or explicitly select CPU. Initialization fallback does not hide
  runtime inference errors.
- **Cutout trails movement:** confirm **Ready: CUDA** and set smoothing to **0**.
  Try **30** mask updates if there is spare GPU capacity; return to **15** if mask age or
  OBS rendering lag increases. Refresh status to inspect processing time and mask age. The model
  still needs time to process each frame; increasing the limit above its throughput cannot
  remove that delay. Lowering webcam resolution does not shrink the fixed 1024×1024 model input.
- **Whole video stutters:** check OBS's Stats window for rendering/encoding lag and GPU
  contention. Reducing mask updates to **15** can free GPU time, at the cost of slower
  cutout updates. CPU mode is a functionality fallback for this model.
