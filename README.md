# OBS RMBG-1.4 background removal

A native C++ OBS **Effect Filter** that runs BRIA RMBG-1.4 locally with ONNX Runtime.
The initial supported platform is native OBS on **Linux x86-64**, with SDR sources.
This is a personal prototype; model weights have BRIA's separate usage terms.

## Try it in OBS

1. Build and install using the commands below, then restart OBS.
2. Right-click your webcam source → **Filters**.
3. Under **Effect Filters**, click **+** → **RMBG-1.4 Background Removal**.
4. The model is selected automatically when installed with `RMBG_INSTALL_MODEL=ON`.
   Otherwise select `data/models/rmbg-1.4.onnx` from this checkout.
5. Leave **Inference device** on **Automatic**, or choose **CUDA GPU** to require it.
6. Click **Refresh status / retry model** after loading. Confirm **Ready: CUDA**.
7. Add a Color Source or image **below** your webcam to clearly see the transparency.

Start with maximum mask updates **15**, temporal smoothing **0.15**, threshold **0.5**,
and edge softness **0.5**. The update setting is a ceiling, not guaranteed throughput.
Try **30** mask updates after confirming GPU processing works. Increase threshold to
remove more foreground; reduce it to keep more. Reduce softness for firmer edges.
**Show mask instead of video** displays white foreground and black background.

The original source, including its background, stays visible while loading, on errors,
or when the last mask is older than the configured timeout. The filter is not a privacy
barrier. A missing model can be fixed by selecting the correct file and clicking retry.

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
destination. The first version supports FP32 tensors only; FP16 model files are rejected
with an explanation. No Python model code or Hugging Face token is used inside OBS.

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

## Implementation

- `src/model.cpp`: validates model input/output, converts premultiplied RGBA to RGB NCHW
  (`channel / 255 - 0.5`), runs ONNX, and safely normalizes the foreground mask.
- `src/worker.cpp`: loads models off the render thread and processes at most one in-flight
  frame. Configuration generations prevent old model results from reappearing.
- `src/plugin.cpp`: captures at 1024×1024 on the GPU, maps a staging transfer on a later
  video tick, uploads masks, and exposes OBS controls and diagnostic status.
- `data/rmbg.effect`: applies the mask to both RGB and alpha for correct premultiplied
  compositing, with threshold, softness, and mask preview controls.

Model inference is asynchronous. The current source is combined with the newest completed
mask; this favors uninterrupted video but can cause outlines to trail fast motion.
Temporal smoothing is adjusted for elapsed time, resets across source-size changes or
long gaps, and cannot eliminate all flicker from an image segmentation model.
RMBG identifies salient objects, so it may preserve chairs and other objects as well as people.

## Current verification

On Linux Mint 22.3, OBS 32.2.0, Intel Core Ultra 7 265K and NVIDIA RTX 5070 Ti:

- C++ build and core checks passed.
- Official model download and cached CUDA library checksums verified.
- FP32 inference: approximately **29–32 ms** per frame on CUDA after warmup;
  approximately **1.2 seconds** on CPU with two inference threads.
- Real libobs rendering, mask transparency, foreground color preservation, and missing-model
  passthrough tested using the installed package in the personal OBS plugin directory.
- GPU compute used the RTX card; virtual-display rendering used Mesa llvmpipe.
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
- **Slow updates:** check the status processing time, disable other heavy GPU tasks,
  and try 15 mask updates per second. CPU mode is a functionality fallback for this model.
