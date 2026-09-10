# OBS background removal: RMBG-1.4 and RVM

A native C++ OBS **Effect Filter** that runs BRIA RMBG-1.4 and Robust Video Matting
(RVM MobileNetV3 / ResNet50) locally with ONNX Runtime.
The initial supported platform is native OBS on **Linux x86-64**, with SDR sources.
This is an experimental community plugin. Performance and supported platforms are
documented below; model licenses are separate from the plugin's source license.

## License and model choice

The plugin source is licensed under **GPL-2.0-or-later**; see [LICENSE](LICENSE) and
[COPYING](COPYING). Model weights and third-party runtime libraries have their own terms.
This source repository does not include downloaded models or runtime binaries.

The setup examples below use **RVM MobileNetV3** and leave BRIA RMBG-1.4 optional.
RVM's upstream GPL-3.0 license and provenance are retained in
[data/licenses/rvm/](data/licenses/rvm/NOTICE.txt).
BRIA's [RMBG-1.4 model card](https://huggingface.co/briaai/RMBG-1.4) describes
noncommercial use and requires a separate agreement for commercial use. Check the
terms attached to your access before downloading or using those weights. This project's
GPL license does not extend to them or grant permission to redistribute them.

For contributions, see [CONTRIBUTING.md](CONTRIBUTING.md). Maintainers preparing source
archives or binaries should read [the public release guide](docs/PUBLIC_RELEASE.md).

## Platform support and setup on another computer

**The current build and installation scripts support native Linux x86-64 only.**
OBS and ONNX Runtime also run on other operating systems, but this repository does
not yet produce a Windows or macOS plugin package.

| Target system | Current status | Setup path |
| --- | --- | --- |
| Native Linux x86-64, NVIDIA GPU | Implemented; validated on Linux Mint 22 | Follow [Linux prerequisites](#build-linux), the CUDA build, and [user installation](#install-for-your-user). |
| Native Linux x86-64, CPU or AMD/Intel graphics | CPU inference implemented | Follow the CPU build and select **CPU** with an FP32 model. AMD/Intel GPU inference is not implemented. Measure throughput before using it live. |
| Other Linux distributions on x86-64 | Requires compatible dependencies; not individually validated | Build locally against OBS 30+ development files from the same source as your OBS installation. Package names vary by distribution. |
| Flatpak OBS | No Flatpak package supplied | Use native OBS for the documented installation. A Flatpak extension/build is separate work; copying the native plugin into the sandbox is not a supported installation. |
| Linux ARM64, including Raspberry Pi/Jetson | Not packaged or validated | The download scripts fetch x86-64 libraries. An ARM64 runtime, build, and target-device validation are required. |
| macOS, Apple silicon or Intel | Not implemented or validated | See [macOS port requirements](#macos-port-requirements). |
| Windows | Not implemented or validated | See [Windows port requirements](#windows-port-requirements). WSL-built Linux libraries cannot be loaded by Windows OBS. |

On a new Linux computer, clone the source and run the setup there. Models, SDKs, and
private CUDA libraries are ignored by Git and must be downloaded separately. The
`.onnx` model files can be reused after checksum verification; compiled plugin/runtime
libraries must match the target OS, architecture, and OBS installation. Do not copy an
old CMake build directory, which contains paths from the original computer.

- [Build on Linux](#build-linux) and [install for your user](#install-for-your-user)
- [Download and verify models](#model-download)
- [Repair failed model downloads](#repair-failed-model-downloads)
- [Troubleshoot loading and runtime errors](#troubleshooting)

## Try it in OBS

1. Build and install using the commands below, then restart OBS.
2. Right-click your webcam source → **Filters**.
3. Under **Effect Filters**, click **+** → **Background Removal (RMBG / RVM)**.
4. Select `rvm_mobilenetv3_fp32.onnx` in the installed `data/models/` directory or
   this checkout. RVM must be selected explicitly; installing its files does not change
   the model picker default. If you separately install RMBG with `RMBG_INSTALL_MODEL=ON`,
   new filters can select it automatically. Download locations are listed [below](#model-download).
5. Leave **Inference device** on **Automatic**, or choose **CUDA GPU** to require it.
   On a CPU-only installation, select **CPU** and start with an FP32 model.
6. Click **Refresh status / retry model** after loading. Confirm **Ready: CUDA** or
   **Ready: CPU**, matching your intended device.
7. Add a Color Source or image **below** your webcam to clearly see the transparency.

For a human webcam on the development machine, select **RVM MobileNetV3 FP32**, choose
**CUDA GPU**, refresh after loading, then click **Apply 1080p30 quality settings**.
This selects 30 mask updates, matching, immediate readback, 1920 input limit, automatic
RVM downsampling, and neutral alpha controls. It leaves the model and device unchanged.
The model must be loaded as RVM before the button is enabled; refresh status if necessary.

New-filter defaults remain 15 updates, matching off, smoothing 0, threshold 0.5, and
softness 0.5. Existing filters retain saved values until you edit them or use the preset.
See [the complete tuning guide](#tuning-guide) for every control and camera setup.

The original source, including its background, stays visible while loading, on errors,
or when the last mask is older than the configured timeout. The filter is not a privacy
barrier. A missing model can be fixed by selecting the correct file and clicking retry.

## Lower delay and align motion (v0.3.0)

**Low latency readback** is enabled by default, including for existing filters without a
saved value. It submits the captured image to inference during the same OBS video tick,
removing the previous mandatory one-tick wait. Inference remains on the worker thread;
it never waits for a model result on the rendering thread. The graphics readback itself
can still wait for the GPU. If OBS reports increased rendering lag on your scene, disable
this option to restore deferred readback. **Refresh status** shows readback duration and
mask age; these are snapshots, not a continuously updating meter.

**Match video to mask** is optional and off by default. Enable it to display each completed
mask with the exact video frame that produced it. The retained video stays at the source's
full resolution on the GPU, even when RVM uses a smaller inference image. Until the next
result arrives, the last complete video/mask pair repeats. This removes temporal offset
between video and mask, but delays video and limits visible motion updates to the completed
mask rate. It does not fix errors in the model's predicted outline. Audio timing is unchanged;
you may need to adjust your microphone sync offset if the added video delay is noticeable.

The matching cache allocates at most six source-sized RGBA textures (up to about 47.5 MiB
at 1080p or 190 MiB at 4K, plus other processing buffers). Active inference and displayed
results pin their own frame; dropped waiting captures release theirs. If every slot is
occupied, the next capture is skipped instead of overwriting a retained frame or growing
a queue. Turning matching off releases its GPU cache. Loading, invalid/missing pairs,
model errors, source-size mismatches, and stale results show the original source.

Extra **Temporal smoothing** is bypassed for matched captures to avoid mixing masks from
different images. Your saved smoothing value is retained for live-video mode. RVM's own
recurrent state remains enabled. Start with low latency readback on, matching on, and
30 mask updates for RVM; reduce the camera size or RVM input limit if inference cannot
keep up. Both options also work with RMBG.

The worker reuses RGBA capture buffers instead of allocating and clearing them each frame.
FP16 models receive a directly prepared FP16 input tensor; RVM FP16 alpha converts directly
to bytes. This removes the previous full-size FP32 intermediates. Matching reuses GPU
video textures for compositing without sending the full-resolution video through the CPU.
The inference path still performs GPU/CPU transfers; it is not a zero-copy CUDA pipeline.

## Use RVM

RVM is the human video-matting model available in the author's TensorFlow.js/WebGL demo.
This native plugin uses the author's **ONNX export** with CUDA or CPU. Model formats for
TensorFlow.js cannot be loaded directly into ONNX Runtime.

1. Download the official models: `python3 scripts/download-rvm.py`.
2. Build this version and restart OBS after installing it. Existing filter instances and
   their saved settings are retained; the registered filter ID is unchanged.
3. In **Model file**, select `rvm_mobilenetv3_fp32.onnx` for CUDA/CPU; FP16 is also
   supported on CUDA. Benchmark both precisions: FP32 is faster in this build on the
   development machine. The model signature is detected automatically.
4. Refresh after loading, then use **Apply 1080p30 quality settings**. Use 1280 as the
   input limit if full-resolution processing cannot meet the measured target.
5. Refresh status: it should show **Ready: CUDA; model: RVM FP32** and increasing
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
The captured source supplies RGB by default. The experimental **RVM foreground colors**
option instead uses the model's estimated `fgr` RGB with its own alpha and matching
source capture. It requires matching, is off by default, and can change foreground color
or texture even in opaque areas. Live-video
mode combines the current image with the newest completed alpha, which can trail motion.
Matching mode instead uses the full-resolution source image belonging to that alpha.

To include both downloaded RVM files in an install, configure with
`-DRVM_INSTALL_MODELS=ON` in addition to the other build options. This option defaults off.
The download script pins release **v1.0.0** and checks SHA256; provenance and the upstream
GPL-3.0 license are in `data/licenses/rvm/` and are included with the plugin data.
See the [official RVM project](https://github.com/PeterL1n/RobustVideoMatting) and its
[inference documentation](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md).

## Tuning guide

The filter captures an image, predicts an alpha matte, and composites the result.
A good outline requires both an accurate matte and the correct video frame. A sharper
model cannot correct an old matte applied to a newer frame, and matching cannot recover
detail that the camera already blurred during exposure.

### Every parameter

Defaults below are **new-filter defaults**, not the values in the quality preset.
Controls apply to both model families unless the table identifies an RVM requirement.
The RVM group is named **RVM settings (used only with an RVM model)**. Changes to model,
device, CPU threads, RVM input size, downsample ratio, or effective foreground-color mode
reload the model and reset temporal state. Other controls do not reload it.

| Exact UI label | Default / allowed values | Purpose, tuning direction, and interactions |
| --- | --- | --- |
| Model file (RMBG-1.4 or RVM ONNX) | Installed `rmbg-1.4.onnx`; existing supported `.onnx` file | RVM targets people and remembers preceding processed frames. RMBG targets salient objects and may retain chairs. Model family is detected from its signature. ResNet50 uses the RVM interface. File name is shown in status; renaming a file does not establish its architecture. |
| Inference device | Automatic (CUDA, then CPU); CPU; CUDA GPU (requires compatible runtime and libraries) | Automatic reports a warning when CUDA initialization falls back to CPU. Explicit CUDA makes initialization failure visible instead of switching to a slower CPU session. `Ready: CUDA` does not imply every graph operation executes on GPU. |
| CPU threads | 2; 1–32, step 1 | ONNX intra-operation CPU threads, including CPU work in a CUDA session. More can help CPU inference but compete with OBS and decoding. This is not a CUDA thread count. |
| Maximum mask updates per second | 15; 1–60, step 1 | Capture/inference-request ceiling. Actual throughput depends on processing and OBS cadence. Increase to 30 for this 30 FPS camera. Setting 60 does not create 60 unique camera frames. |
| Low latency readback (disable if rendering stalls) | On; On/Off | Maps the captured GPU image in the same video tick. Disable only when measured OBS rendering stalls improve with deferred mapping; deferral adds one OBS tick before submission. |
| Match video to mask (buffers video; disables extra smoothing) | Off; On/Off | Displays the full-resolution capture that produced each matte. Repeats the last pair until another completes. Removes video/mask temporal offset, adds video delay, and does not delay audio. |
| Temporal smoothing (live video mode only) | 0; 0–0.95, step 0.05 | Blends preceding and current alpha in live mode. Larger values reduce flicker but can trail movement. At a 30 Hz update cadence, 0.1 retains 10% of the preceding matte; weights adjust with elapsed capture time. Bypassed in matched mode. RVM's internal recurrence still runs. |
| Mask threshold | 0.5; 0–1, step 0.01 | Increase to remove more uncertain foreground; decrease to retain more. High values can erase hair and fingers. This remaps alpha; it does not improve model understanding or latency. |
| Edge softness | 0.5; 0.001–0.5, step 0.01 | Width of alpha transition around threshold, not a spatial blur radius. Lower values harden edges. RVM at threshold/softness 0.5/0.5 preserves predicted alpha; RMBG uses a smoothstep remap. |
| Maximum input long edge (pixels) | 1280; UI choices 640, 1280, 1920 | RVM only. Preserve aspect ratio without upscaling. Higher values retain refinement detail but increase readback, preprocessing, inference, and optional foreground transfer cost. API/CLI additionally accept integer limits 320–1920. Does not reduce retained matching-video resolution. |
| Downsample ratio | Automatic (480-pixel internal long edge); 0.125, 0.25, 0.375, 0.5, 0.6, 0.75, 1.0 | RVM only. Controls the coarse stage before high-resolution refinement. Automatic is `min(1,480/input_long_edge)`. Higher ratios cost more and can help smaller/full-body subjects, but are not always better. CLI/API accept 0 (auto) or 0.1–1. |
| RVM foreground colors (experimental; matched video only) | Off; On/Off | Requests RVM's estimated foreground RGB to reduce possible original-background color contamination. Requires a ready RVM model and matching; refresh after model load. Turning matching off suspends it while retaining the saved checkbox. May alter skin, clothing, or opaque detail and transfers an extra RGB image. Compare before adopting. |
| Apply 1080p30 quality settings | Button; ready RVM only | Sets updates 30, matching/readback on, smoothing 0, input limit 1920, ratio auto, threshold/softness 0.5/0.5, timeout 2000, and experimental foreground colors off. Preserves model, device, CPU threads, and preview selection. Takes effect when clicked; is not an automatic settings migration. |
| Discard mask older than (milliseconds) | 2000; 250–10000, step 250 | Fallback threshold measured from plugin capture. An expired matte shows the original source. Lowering it does not accelerate inference and can cause background flashes during stalls. |
| Show mask instead of video | Off; On/Off | White = retained foreground, black = removed background, gray = partial alpha. Shows the alpha after threshold/softness, useful for distinguishing matte errors from RGB fringes. |
| Refresh status / retry model | Button | Refreshes the status snapshot and control availability. Retries a failed model; it does not restart a healthy one. Status is not a continuously updating meter. |

Status includes the selected file, backend, input tensor precision, effective capture
dimensions, actual downsample ratio, recurrence, readback, and timing. RVM input precision
is FP32 or FP16; this label does not describe every internal operator in an arbitrary graph.
RMBG's official FP16 variant can still have FP32 input/output tensors.

### The five resolutions

| Stage | Example from this setup | Effect on the filter |
| --- | --- | --- |
| Camera capture | 1920×1080 MJPEG | Determines source detail, camera bandwidth, decoding work, and full-resolution matching textures. |
| OBS base canvas | 2560×1440 | Scene composition size. Changing it affects scene rendering, not this webcam's base texture dimensions. |
| OBS output | 1920×1080 | Encoded/output size after scene scaling. Lowering it can reduce output/encoding load without reducing model input. |
| RVM capture/input | 1280×720 or 1920×1080 | Actual image read back and passed to RVM, limited by **Maximum input long edge**. |
| RVM coarse stage | Approximately 480×270 in both examples | Input dimensions multiplied by the ratio. Refinement still uses the selected RVM input resolution. |

At 1280×720, automatic ratio is **0.375**; at 1920×1080 it is **0.25**. Both coarse
stages are approximately 480×270, but 1920 input offers more refinement detail. A 640×360
input uses 0.75 automatically; reducing only the input limit therefore does not always
reduce coarse-stage work. Reducing a manual ratio can reduce coarse work at a quality cost.

Dragging the webcam smaller in the scene does not shrink its base source texture before
this filter. To reduce filter work, change its RVM input limit or camera capture mode.
The 1920 limit applies to the **long edge**, including portrait sources. RVM controls do
not change RMBG's fixed 1024×1024 inference size. A higher output resolution cannot restore
detail lost in the camera or inference input. Matching retains up to six source-sized
RGBA textures: about 47.5 MiB at 1080p or 190 MiB at 4K, plus model and processing buffers.

### FPS and latency are different measurements

| Rate | One frame/update every | Meaning |
| --- | --- | --- |
| 15 FPS | 66.67 ms | At most one requested matte for every two frames of a 30 FPS camera. |
| 30 FPS | 33.33 ms | Target camera and mask cadence for the Brio 100. |
| 60 FPS | 16.67 ms | OBS can render other scene content more smoothly; a 30 FPS camera still supplies only 30 new images. |

The **camera FPS** is the source cadence. **OBS FPS** is the render/output cadence.
**Maximum mask updates** caps requests; **completed masks/s** measures actual inference
completions. In matched mode, **distinct pairs/s** counts newly presented plugin capture
identities, while **repeated pairs** counts reuse of the same completed pair. Neither is a
measurement of unique camera sensor frames: the plugin can capture the same source image
twice when OBS runs faster than the camera. A 60 FPS file is not evidence of a 60 FPS camera.

At 30 masks/s with 60 FPS OBS output, repeating each matched pair for roughly two video
ticks is normal. If the model completes only 15 masks/s, matched motion updates only about
15 times/s. Turning matching off shows newer video but can misalign its outline.

Five-second rolling diagnostics report mean/p95 processing time, queue wait, and displayed
mask age. p95 is the nearest-rank 95th percentile; startup rates use elapsed time until a
full window exists. Wait at least ten seconds after loading or changing settings before
judging steady-state performance. Windows expire after inactivity and storage is capped
at 4096 samples per metric. Rates reset on model reload; presented-pair windows also reset
when matching changes. Lifetime counters stay cumulative for the filter instance.

- **Processing** includes tensor preparation, ONNX execution/transfers, alpha conversion,
  and optional foreground conversion; it excludes extra live-mode smoothing.
- **Queue wait** is worker acceptance to inference start. Replaced waiting frames are
  counted separately; the queue remains one active plus one replaceable waiting frame.
- **Displayed mask age** is plugin capture to rendering with that matte, including
  readback, waiting, processing, and repeated presentations. In matched mode this is also
  the plugin's buffered video age. It is not whole-system camera-to-screen latency.
- **Readback** covers GPU mapping, CPU copy, and worker submission. Source rendering and
  staging are included in mask age, not in this timer.

Lowering the timeout, increasing the FPS ceiling beyond throughput, or adding smoothing
does not cure an overloaded pipeline. First examine queue wait, completed rate, mask age,
and **View → Stats** in OBS for rendering/encoding lag.

### Camera setup: Logitech Brio 100 on Linux

On the development machine, V4L2 advertised **MJPEG 1920×1080 at 30 FPS**. All advertised
formats topped out at 30 FPS; this camera did not offer a true 60 FPS mode, including 720p.
Its advertised YUYV 1080p mode was only 5 FPS, so choose MJPEG for 1080p30. Verify your own
device instead of copying another webcam's modes:

```sh
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video0 --list-ctrls-menus
```

Set the webcam source's resolution and frame rate in OBS, then check the OBS log for the
actual negotiated values. Prefer MJPEG/1920×1080/30 for this Brio. Higher compression or
gain noise can damage fine detail before the model sees it. A small camera image in the
scene does not require changing the whole OBS canvas or screen-capture resolution.

For fast movement, improve lighting before increasing model complexity. Use steady front
lighting, avoid a very dark face against a bright background, and keep gain as low as the
available light permits. Avoid excessive camera sharpening, which can create pale edge
rings. Lock white balance after lighting is stable if automatic color changes are visible.

Exposure is the time each camera image collects light. Near-1/30-second exposure smears
moving hands even when frame delivery is 30 FPS. As a starting experiment under 50 Hz
lighting, try **1/100 second (10 ms)** with enough light. Test for LED flicker/banding,
brightness, noise, and actual frame cadence; some lighting requires a different exposure.
1/50 second admits more light but more motion blur. FPS and shutter time are separate.

The controls below were present on this Brio, not guaranteed on every V4L2 device. Close
other camera applications first, record the current values, and verify the control names
and menu values before applying changes. OBS can reapply saved camera controls on startup.
The plugin never changes these controls.

```sh
# Save values to a file you can use to restore the previous configuration.
v4l2-ctl -d /dev/video0 --get-ctrl=auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,power_line_frequency,gain,sharpness,white_balance_automatic,white_balance_temperature > camera-controls-before.txt

# On this Brio: manual exposure=1; 100 exposure units = 10 ms = 1/100 second.
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=100,exposure_dynamic_framerate=0,power_line_frequency=1
v4l2-ctl -d /dev/video0 --get-ctrl=auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,power_line_frequency
```

For restoration, read `camera-controls-before.txt`. While still in manual exposure, restore
the saved exposure time, gain, sharpness, and power-line setting with `--set-ctrl=name=value`.
Restore saved automatic-exposure mode and dynamic-framerate value afterward. If restoring
manual white balance, disable automatic white balance before setting its saved temperature;
then restore its saved automatic flag. Do not assume the current device's reported exposure
under automatic control was the exact exposure during an earlier recording.
Linux defines absolute exposure in **100 µs units** and permits dynamic FPS when automatic
exposure priority allows it. [Kernel camera-control reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)

### Recommended recipes

| Recipe | Configuration | When to use |
| --- | --- | --- |
| Clean 1080p30 | Brio MJPEG 1080p30; RVM MobileNetV3 FP32; CUDA; quality button; foreground colors off | Starting point on the RTX 5070 Ti. OBS output can remain 60 FPS for other scene content. |
| Lower processing cost | Same configuration, RVM input limit 1280 | Use when 1920 input causes processing/rendering contention. Matching video remains full-resolution; matte refinement is 720p. |
| Lowest added video delay | Matching off; immediate readback on; 30 mask updates; smoothing 0; input 1280 initially | Accepts possible outline displacement during rapid motion. Enable foreground colors only when returning to matching. |
| Edge-color comparison | Quality recipe, then enable RVM foreground colors | Compare skin, clothing, hair, and opaque detail against black, white, and colored backgrounds. Keep only if it visibly helps. |

For RVM, increase threshold in small 0.01 steps only after fixing timing and exposure.
Begin at 0.5/0.5 to preserve alpha. Try live-mode smoothing 0.05–0.15 only for residual
flicker; it is intentionally bypassed when matching is on. CPU mode may require a lower
input size and rate. Increasing CPU threads is not a substitute for checking measured FPS.

### Troubleshoot in this order

| Symptom | First checks | Next step |
| --- | --- | --- |
| Outline trails a moving head/hand | Matching on? Requested and completed mask rates near 30? | Inspect the unfiltered source for blur; inspect queue time and mask age. |
| Motion looks stepped or freezes | Distinct pair rate, repeated pairs, worker replacements, OBS Stats | Use 1280 input; compare immediate/deferred readback only if rendering stalls. |
| Hands are blurred with filter disabled | Exposure time, lighting, gain, actual camera FPS | More light and shorter exposure; the filter cannot restore missing sensor detail. |
| Pale/dark fringe around hair | Mask preview against original/composite; camera sharpening | Compare RVM foreground colors; avoid simply hardening all edges and erasing hair. |
| Fingers/hair disappear | Neutral threshold/softness, subject size, 1920 input | Compare a higher coarse ratio or ResNet50 using the same unfiltered sequence. |
| Outline flickers while still | Lighting, gain noise, automatic exposure/white balance, recurrent state | Check recurrence resets; only then try a small amount of live-mode smoothing. |
| Original background flashes | Model error, stale timeout, dimensions or HDR warning | Fix the reported cause; do not hide it by extending the timeout indefinitely. |
| GPU selected but slow | Actual backend, input precision, processing and queue times | Benchmark FP32/FP16; CUDA can still execute shape/unsupported operations on CPU. |

### Audio sync

Matching delays video and leaves the microphone unchanged. Record several visible claps
with your normal scene and settings after warmup. In an editor, compare the frame where
hands meet with the audio transient. If audio occurs first, add a **positive microphone
Sync Offset** in **Advanced Audio Properties** by the measured difference and record again.
At 30 FPS, one video frame is 33.33 ms; at 60 it is 16.67 ms. Measure several claps because
delivery jitter can vary. Do not copy the plugin's mask-age number directly: camera,
microphone, output, and any pre-existing sync offsets also contribute to the recording.

### Additional model options

Download ResNet50 explicitly; the existing command still downloads only MobileNetV3:

```sh
python3 scripts/download-rvm.py --variant resnet50 --precision both
# Optional personal packaging, in addition to your existing configure arguments:
cmake -S . -B build -DRVM_INSTALL_RESNET_MODELS=ON
```

`--variant` accepts `mobilenetv3`, `resnet50`, or `both`; `--precision` accepts `fp32`,
`fp16`, or `both`. Both ResNet files are pinned to official release v1.0.0 and verified
against the SHA256 values in `data/licenses/rvm/NOTICE.txt`. The install option defaults off.
RVM ResNet50 is a larger alternative with small upstream-reported improvements, not a
guaranteed fast-motion fix. Compare it before accepting extra cost. Model signature checks
support the official recurrent interface for both variants.
[Official RVM project](https://github.com/PeterL1n/RobustVideoMatting)

MatAnyone 2 is a future research candidate, not supported by this plugin. Its first-frame
segmentation mask and memory initialization need a different workflow; it is not a file
picker replacement for RVM. [Official MatAnyone 2 project](https://github.com/pq-yang/MatAnyone2)

## Build (Linux)

Required: CMake 3.24+, a C++17 compiler, OBS 30+ development files, libpng development
files, and Python with `hashlib.file_digest` and tarfile's safe `data` extraction filter.
**Python 3.12+ is recommended** (3.12 was used on Mint 22); older Python 3.11 patch
releases may lack the extraction filter. See [Python's extraction-filter documentation](https://docs.python.org/3.12/library/tarfile.html#extraction-filters).
The installed OBS application and development libraries should come from the same source.

### Prerequisites on a fresh Linux machine

For Ubuntu/Mint with repositories providing OBS 30+ and the required tool versions:

```sh
sudo apt update
sudo apt install git build-essential cmake python3 obs-studio libobs-dev libpng-dev libsimde-dev
git clone https://github.com/RajKKapadia/obs-bg-removal-plugin.git
cd obs-bg-removal-plugin
uname -m
obs --version
cmake --version
python3 --version
```

`uname -m` must report `x86_64` for the supplied SDK downloads. On another distribution,
install equivalent compiler, CMake, Python, OBS development, libpng, and SIMDe packages
through its package manager. If the OBS application came from a different repository,
obtain matching development files there too. A Flatpak installation does not supply the
native `libobs` development package used by these commands.

Run all following shell commands from the repository root. Choose **one** backend below.
Both examples download the runtime and the two RVM MobileNetV3 models, and include the
RVM files in a personal installation. `--skip-model` skips BRIA's RMBG download;
`RMBG_INSTALL_MODEL=OFF` excludes it from the install. Existing RMBG functionality and
download support remain available through [the optional setup](#optional-rmbg-14-setup).

### CPU-only build

```sh
python3 scripts/bootstrap.py --backend cpu --skip-model
python3 scripts/download-rvm.py
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="$PWD/.deps/onnxruntime-linux-x64-1.29.0" \
  -DRMBG_CUDA_LIBRARY_DIR= \
  -DRMBG_INSTALL_MODEL=OFF -DRVM_INSTALL_MODELS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

`--fresh` clears cached SDK/library paths when switching between CPU and CUDA builds;
include any additional configure options again. Start with RVM MobileNetV3 **FP32**, an input
limit of 640 or 1280, and a modest update rate; CPU performance depends on the machine.

### NVIDIA CUDA build

Confirm `nvidia-smi` works before starting. The following is the build used on the
development machine:

```sh
python3 scripts/bootstrap.py --backend cuda --skip-model --simde
python3 scripts/prepare-cuda.py
python3 scripts/download-rvm.py
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DONNXRUNTIME_ROOT="$PWD/.deps/onnxruntime-linux-x64-gpu_cuda12-1.29.0" \
  -DRMBG_CUDA_LIBRARY_DIR="$PWD/.deps/cuda/lib" \
  -DRMBG_INSTALL_MODEL=OFF -DRVM_INSTALL_MODELS=ON
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

### Optional RMBG-1.4 setup

After reviewing BRIA's model terms, download the pinned RMBG file using the existing
bootstrap command without `--skip-model`. Match the backend to the build you configured:

```sh
# CPU build; use --backend cuda instead for a CUDA build.
python3 scripts/bootstrap.py --backend cpu
cmake -S . -B build -DRMBG_INSTALL_MODEL=ON
```

Then use the installation command below. This adds RMBG to a personal install and
keeps the existing RVM configuration. To use a file directly from the checkout, leave
the install option off and select it in OBS. The bootstrap script's original behavior
is unchanged: omitting `--skip-model` downloads RMBG as well as the selected SDK.

## Install for your user

For the native Linux OBS application, close OBS before replacing an installed library:

```sh
cmake --install build --prefix "${XDG_CONFIG_HOME:-$HOME/.config}/obs-studio/plugins"
```

This creates only `obs-rmbg/` inside OBS's user plugin directory. No `sudo` is required.
The installed library uses paths relative to itself, including the matching private CUDA
libraries when configured. Restart OBS after installing or rebuilding.
This package has **not** been prepared for Flatpak OBS, Windows, or macOS.

Expected layout inside the native Linux plugin directory:

```text
obs-rmbg/
  bin/64bit/
    obs-rmbg.so
    libonnxruntime.so*                 (including installed versioned links/files)
    libonnxruntime_providers*.so       (when supplied by the selected SDK)
    cuda/                             (when private CUDA libraries are configured)
  data/
    rmbg.effect
    locale/en-US.ini
    licenses/
    models/                           (only models enabled at configure time)
```

After installation, restart OBS and follow [Try it in OBS](#try-it-in-obs). Existing
filters keep their saved model paths: if the path points into your old computer's home
directory or a moved checkout, select the file on the new computer explicitly.
Downloaded checkout models and installed models are separate copies.

For an inspectable package before installation:

```sh
cmake --install build --prefix "$PWD/dist"
```

This stages a **personal installation**, including any enabled models and private
runtime dependencies. It is not a reviewed public release package. Keep it local;
follow [the release guide](docs/PUBLIC_RELEASE.md) before publishing downloadable assets.

Model installation defaults **off**. `RMBG_INSTALL_MODEL=ON` is intended for your personal
installation, not a public distribution package. Library license notices are installed
under the plugin's `data/licenses/` directory. Keep BRIA's model terms separate.

For another Linux machine, building there is preferred. If transferring a personal
staged package to a compatible machine, keep the entire `dist/obs-rmbg/` directory
together, including runtime libraries, data, and notices. Preserve symlinks when copying.
Rebuild on the destination if the loader reports incompatible GLIBC/GLIBCXX or OBS
symbols. After moving between CUDA and CPU builds, stage into a fresh directory to avoid
retaining old runtime files from an earlier install.

### macOS port requirements

There is currently **no working macOS installation command for this checkout**.
Installing CMake or renaming `obs-rmbg.so` does not make the Linux package load on macOS.
A port needs:

- An OBS development SDK, ONNX Runtime SDK, and libpng matching the architecture of
  the running OBS application (`arm64` or `x86_64`), plus Xcode command-line tools.
- Platform-aware CMake rules for an OBS `.plugin` bundle, `.dylib` dependencies, and
  macOS library lookup paths instead of Linux `.so` packaging and `$ORIGIN`.
- A CPU inference baseline and native rendering/model-load tests. The plugin currently
  selects only CPU or CUDA; Apple GPU/CoreML acceleration would require additional code.

A completed macOS port would install its bundle under
`~/Library/Application Support/obs-studio/plugins/`. This is an OBS convention, not a
package generated by this repository. See the [OBS plugins guide](https://obsproject.com/kb/plugins-guide)
and [official plugin template](https://github.com/obsproject/obs-plugintemplate) for
bundle, build, and signing guidance.

### Windows port requirements

There is currently **no working Windows installation command for this checkout**.
A port needs:

- Visual Studio C++ build tools and matching OBS, ONNX Runtime, and libpng development
  files for the architecture of the running OBS application.
- CMake changes for MSVC options, the ONNX Runtime import library and DLLs, and Windows
  plugin installation. The current scripts download Linux archives and NVIDIA `.so`
  libraries; running them in PowerShell or WSL does not produce a Windows plugin.
- CPU model-load and native OBS rendering tests first, followed by CUDA testing with
  matching Windows runtime DLLs and an NVIDIA driver. DirectML acceleration is not
  implemented by this plugin.

A completed Windows port would normally install `obs-rmbg.dll` in
`C:\ProgramData\obs-studio\plugins\obs-rmbg\bin\64bit\`, with effects, locale,
licenses, and models under the sibling `data\` directory. The Linux `.so` cannot be
used there. See the [OBS plugins guide](https://obsproject.com/kb/plugins-guide) and
[official plugin template](https://github.com/obsproject/obs-plugintemplate) for the
Windows build and package conventions.

## Model download

### Download locations and installation options

Downloads require an internet connection; inference in OBS does not. Use a writable
checkout and allow space for the model files plus a temporary download. SDK/CUDA setup
also needs several GB. The scripts verify SHA256 before accepting a download.

| Model | Download command on Linux | Include in a personal install |
| --- | --- | --- |
| RMBG-1.4 FP32 | `python3 scripts/bootstrap.py --backend cpu` (or `--backend cuda`) | `-DRMBG_INSTALL_MODEL=ON` |
| RVM MobileNetV3 FP32 and FP16 | `python3 scripts/download-rvm.py` | `-DRVM_INSTALL_MODELS=ON` |
| RVM ResNet50 FP32 and FP16 | `python3 scripts/download-rvm.py --variant resnet50` | `-DRVM_INSTALL_RESNET_MODELS=ON` |

All models go into `data/models/` in the checkout. CMake options only **copy** already
downloaded models; they do not download them. Each RVM install option requires **both**
FP32 and FP16 files for that variant. If you download only `--precision fp32`, leave
the corresponding install option off and select that file directly in OBS, or download
both before installing. RVM installation does not change the default model picker to RVM.

For model-file preparation on macOS, the RVM script can be run with Python 3.12+ using
the same `python3` command. In Windows PowerShell with the Python launcher, use:

```powershell
py -3.12 scripts/download-rvm.py --variant mobilenetv3 --precision both
```

This only downloads portable model files; macOS/Windows plugin support still requires
the ports described above. `bootstrap.py` and `prepare-cuda.py` stage Linux dependencies
and are not macOS/Windows setup tools.

### RMBG provenance

The bootstrap script downloads the original FP32 ONNX file from the official repository:

- [BRIA RMBG-1.4 ONNX files](https://huggingface.co/briaai/RMBG-1.4/tree/main/onnx)
- Destination: `data/models/rmbg-1.4.onnx`
- Revision: `2ceba5a5efaec153162aedea169f76caf9b46cf8`
- SHA256: `8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f`

If access is required, download the [pinned **model.onnx**](https://huggingface.co/briaai/RMBG-1.4/resolve/2ceba5a5efaec153162aedea169f76caf9b46cf8/onnx/model.onnx?download=true)
in your browser and save it at that destination. Verify its SHA256 using the commands
below. The plugin accepts FP32 or FP16 input/output tensors for the supported
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

### Verify a downloaded model

Use the hash for the **exact variant** you downloaded. RVM files are available as
assets in the [official v1.0.0 release](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0).
These pins also appear in [the download script](scripts/download-rvm.py) and
[RVM provenance notice](data/licenses/rvm/NOTICE.txt).

| RVM filename | SHA256 |
| --- | --- |
| `rvm_mobilenetv3_fp32.onnx` | `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828` |
| `rvm_mobilenetv3_fp16.onnx` | `6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64` |
| `rvm_resnet50_fp32.onnx` | `25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c` |
| `rvm_resnet50_fp16.onnx` | `a9266f5046411d604bbff38e18c54bf8c70d85d93bbc400564697590f5712738` |

Linux (replace the filename to check another model):

```sh
sha256sum data/models/rvm_mobilenetv3_fp32.onnx
```

macOS:

```sh
shasum -a 256 data/models/rvm_mobilenetv3_fp32.onnx
```

Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\data\models\rvm_mobilenetv3_fp32.onnx
```

The complete hash must match the corresponding value above, ignoring letter case.
Check the **actual file selected in OBS**, including the installed copy if applicable.
A matching checkout file does not prove an older installed copy is intact. A filename
or nonzero size alone is not verification. A tiny file containing HTML, a login message,
or `version https://git-lfs.github.com/spec/v1` is a page/pointer, not the model weights.

### Repair failed model downloads

1. **Capture the failure.** Note the exact command, model filename, and error. Check
   connectivity, free disk space, and write access to the checkout. Do not run downloads
   with `sudo`; use a directory owned by your user.
2. **Rerun the same downloader.** Valid existing files print `Verified existing` and
   are reused. Missing files or files with a wrong hash are downloaded again. Downloads
   use a `.part` file and replace the destination only after verification. Interrupted
   transfers restart from the beginning; there is no resume option. Normal failures
   clean up `.part`; a leftover after a killed process can be removed once no downloader
   is running. There is no need to delete all models or edit the checksum pins.
3. **Verify the resulting file** with the commands above. If downloading in a browser,
   use the pinned RMBG link or the RVM v1.0.0 release asset, not a repository HTML page,
   source-code archive, TensorFlow.js file, or PyTorch checkpoint. Save it with the exact
   `.onnx` filename. For offline setup, transfer a verified copy and hash it again on
   the destination; the native SDK/runtime dependencies must also be present to build.
4. **Update the file OBS actually uses.** Select the repaired checkout file explicitly,
   or close OBS and rerun the configured `cmake --install build --prefix
   "${XDG_CONFIG_HOME:-$HOME/.config}/obs-studio/plugins"` to update installed copies.
   Reinstalling a model requires its corresponding CMake install option to be enabled.
5. **Reload and check status.** Reopen OBS if you reinstalled. Select the intended model,
   click **Refresh status / retry model** on an error, wait for loading, and refresh
   again. If a ready session still holds an older file replaced at the same path,
   restart OBS to force it to reopen the file. Expect **Ready: CPU** or **Ready: CUDA**.

Examples from the repository root (run only the command for the file being repaired):

```sh
python3 scripts/download-rvm.py --variant mobilenetv3 --precision fp32
python3 scripts/download-rvm.py --variant resnet50 --precision both
python3 scripts/bootstrap.py --backend cpu
```

For a CUDA installation, use `--backend cuda` in the last command. Bootstrap verifies
and extracts the selected Linux SDK before downloading RMBG; it does not rebuild or
change the installed plugin. Use the pinned browser link if you only need the RMBG file.

| Symptom | Likely cause and recovery |
| --- | --- |
| Timeout, connection reset, DNS failure, or HTTP 429/5xx | Retry later or use a stable connection. For proxy networks, configure the required `HTTPS_PROXY`/`HTTP_PROXY` settings before running Python. |
| `CERTIFICATE_VERIFY_FAILED` | Check the system clock and Python/OS trust store; install the network's required CA certificate through its supported setup. Keep TLS verification enabled. |
| HTTP 401/403 for RMBG | Open the pinned link in a browser and complete any required access steps. The script does not accept a Hugging Face token argument. If access remains unavailable, use RVM with bootstrap `--skip-model` and `RMBG_INSTALL_MODEL=OFF`. |
| `SHA256 mismatch ... refusing to use it` | The bytes differ from the pinned asset, possibly due to truncation, an error page, a proxy, or a changed upstream asset. Retry the official download and compare its hash; keep the pin unchanged until the asset's provenance is checked. |
| `No space left on device` / `Permission denied` | Free space or use a writable checkout, then rerun. `.deps/`, temporary downloads, and optional installed copies also consume space. |
| `hashlib` has no `file_digest`, or tar extraction rejects `filter` | Use Python 3.12+ and rerun. Do not remove the extraction filter to work around an old interpreter. |
| CMake install cannot find an `.onnx` file | An install option is enabled but its files were not downloaded. Download the required model(s), including both precisions for an enabled RVM variant, or disable that option and choose your file manually. |
| `Choose an existing RMBG-1.4 or RVM ONNX model file` | The selected path is empty, missing, or still points to another machine. Pick an existing local file, then retry. |
| ONNX/protobuf parse error or `INVALID_PROTOBUF` | Verify the selected file's hash first. An incomplete file, HTML download, or Git LFS pointer can cause this. Re-download the exact official ONNX asset. |
| `Unsupported model signature` or RVM tensor/dimension errors | Select a supported RMBG-1.4 or official RVM ONNX export. Renaming another model to one of these filenames does not convert it. |
| CUDA/provider/library error with a matching model hash | Check the runtime installation in [Troubleshooting](#troubleshooting). Re-downloading valid weights will not repair missing CUDA libraries. Try an FP32 model with **CPU** to isolate the backend. |

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
camera buffering, display latency, and the visual effect of temporal smoothing. It also
reports mean/max readback time, matched frames, cache size, and cache misses. Readback time
covers mapping, the CPU copy, and submission; source rendering/staging time is included
in mask age, not the readback timer.

Set `RMBG_TEST_MATCH_VIDEO=1` to test matched video, or `RMBG_TEST_READBACK=deferred` to
compare the compatibility path. For example, prefix the `xvfb-run` command with either
or both environment assignments.

A deterministic motion test uses a tiny generated ONNX graph whose mask is the red input
channel. This separates frame-pairing correctness from a trained model's accuracy:

```sh
uv run --with onnx python tests/make-motion-model.py artifacts/motion.onnx
RMBG_TEST_MOTION=1 RMBG_TEST_MATCH_VIDEO=1 RMBG_TEST_TOGGLE=1 \
  xvfb-run -a build/obs-rmbg-smoke \
  "$PWD/build/obs-rmbg.so" "$PWD/data" "$PWD/artifacts/motion.onnx" \
  "$PWD/input.png" "$PWD/artifacts/motion-result.png" cpu 15 0.9 3
```

The input PNG supplies dimensions (at least 64 pixels per side); the test replaces it
with a moving red foreground on a blue background. It checks every captured frame for
incorrect background showing through the foreground mask. It also switches both options
while processing, checks live video as a negative control, and verifies missing-model
passthrough. The high saved smoothing value must not affect matched output.

To check exact pairing under overload, generate the same alpha with extra CPU work:

```sh
uv run --with onnx python tests/make-motion-model.py artifacts/slow-motion.onnx --work-layers 32
RMBG_TEST_MOTION=1 RMBG_TEST_MATCH_VIDEO=1 RMBG_TEST_OVERLOAD=1 RMBG_TEST_SOURCE_FPS=30 RMBG_TEST_OBS_FPS=60 \
  build/obs-rmbg-smoke "$PWD/build/obs-rmbg.so" "$PWD/data" "$PWD/artifacts/slow-motion.onnx" "$PWD/input.png" "$PWD/artifacts/overload.png" cpu 30 0 5
```

Increase `--work-layers` on faster CPUs until replacements occur. This test requires
measured overload and a mismatch fraction below 0.1%; it does not apply the 29 masks/s
throughput gate. The worker test separately gates inference deterministically and checks
that only the newest waiting frame survives overload and old generations are discarded.

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

## Reproduce motion and model comparisons

Record an **unfiltered** clip with the background-removal filter disabled: hold still,
turn your head quickly both ways, move your shoulders, wave an open hand, spread fingers,
and leave/re-enter the frame. Keep exposure and lighting fixed between comparisons.
Re-enable your filter afterward. A previously composited recording can test throughput
but cannot recover the original background or establish matte accuracy.

The sequence tool processes every supplied frame in order, using source-rate timestamps
and recurrent state. Its output is an offline quality comparison, not a live latency test:

```sh
mkdir -p artifacts/raw-motion
ffmpeg -i raw-webcam.mkv -t 10 -vf fps=30 -compression_level 1 artifacts/raw-motion/%06d.png

build/rmbg-sequence data/models/rvm_mobilenetv3_fp32.onnx artifacts/raw-motion artifacts/mobile-1280 --device cuda --fps 30 --rvm-max-size 1280
build/rmbg-sequence data/models/rvm_mobilenetv3_fp32.onnx artifacts/raw-motion artifacts/mobile-1920 --device cuda --fps 30 --rvm-max-size 1920
build/rmbg-sequence data/models/rvm_resnet50_fp32.onnx artifacts/raw-motion artifacts/resnet-1920 --device cuda --fps 30 --rvm-max-size 1920
build/rmbg-sequence data/models/rvm_mobilenetv3_fp32.onnx artifacts/raw-motion artifacts/mobile-colors --device cuda --fps 30 --rvm-max-size 1920 --rvm-foreground 1

ffmpeg -framerate 30 -i artifacts/mobile-1920/comparison/%06d.png -c:v libx264 -crf 18 -pix_fmt yuv420p artifacts/mobile-1920-comparison.mp4
ffmpeg -framerate 30 -i artifacts/mobile-1920/alpha/%06d.png -c:v libx264 -crf 18 -pix_fmt yuv420p artifacts/mobile-1920-alpha.mp4
```

Repeat the two encoding commands for each output directory. Each contains full-size
`alpha/` previews, straight-alpha `cutout/` PNGs, and `comparison/` strips against black,
white, and blue backgrounds. PNGs are named in presentation order; input filenames must
sort in frame order (use zero padding). Output directories must be new or empty. The first
ten processed frames warm recurrent/model state and are excluded from timing summaries;
they remain in the output for inspection. `--warmup N` changes this exclusion.
`timings.csv` records each processing time, input dimensions, ratio, and recurrence count;
`summary.txt` identifies model/backend/precision and reports mean/p95. Disk I/O, resizing,
PNG generation, and video encoding are outside these processing measurements.

For live OBS timing with separately controlled source and render cadence:

```sh
python3 scripts/benchmark-motion.py raw-webcam.mkv artifacts/live-1080p30 --input-kind raw --seconds 600 --source-fps 30 --obs-fps 60 --encode-load
```

Run from your normal graphical Linux session after building. This uses an isolated libobs
scene on the available OpenGL display; it never opens or modifies your OBS scene collection.
The optional load is a separate paced NVENC H.264 encoder, not a recording from your own OBS
scene. The script saves source metadata and SHA256, model SHA256, exact options, original
stream properties, extracted frames, `obs.log`, `encoding.log`, a screenshot, one-second
timing samples, and `summary.json`. By default it extracts up to ten seconds of PNGs and
loops them for the measurement; allow disk space for several hundred full-resolution PNGs.
Use `--input-kind composite` only for explicitly labeled performance-only input.

The numeric acceptance gate requires at least 29 completed masks/s, displayed-age p95
below 75 ms after the first ten seconds, valid matching, and at most six cached textures.
The replay also fails on decoder errors or missed source-frame deliveries. Each reported
p95 covers a rolling five-second window, not the whole run. Inspect RSS for a plateau and
OBS rendering/encoding statistics as well; memory slope is reported for review, not an
automatic pass/fail threshold. Retain a candidate only if it improves the raw footage and
passes the performance check at representative load.

The underlying harness additionally accepts these environment variables:

| Variable | Default / meaning |
| --- | --- |
| `RMBG_TEST_SEQUENCE` | Optional directory of ordered PNGs, decoded off the render thread with a three-frame cache. |
| `RMBG_TEST_SOURCE_FPS` / `RMBG_TEST_OBS_FPS` | 30 / 30; independent source and output cadences (1–240). |
| `RMBG_TEST_WIDTH` / `RMBG_TEST_HEIGHT` | 640 / 360 output canvas; set 1920 / 1080 for the main target. Source dimensions still come from the PNG sequence. |
| `RMBG_TEST_RVM_SIZE` | 1280; RVM input limit (320–1920). |
| `RMBG_TEST_FOREGROUND` | Set to request the optional foreground colors; also enable matching. |
| `RMBG_TEST_PRESET` | Exercise the quality button and verify its settings/model/device preservation. |
| `RMBG_TEST_STALE` | Exercise stale-mask passthrough and fresh-mask recovery. |
| `RMBG_TEST_ACCEPTANCE` | Enforce the 29 masks/s and p95 <75 ms gates after warmup. |

Existing `RMBG_TEST_MOTION`, `RMBG_TEST_MATCH_VIDEO`, `RMBG_TEST_READBACK`, and
`RMBG_TEST_TOGGLE` remain supported. Measurement duration accepts 1–3600 seconds.
Motion fixtures reverse direction rapidly and can run a 30 FPS source in 30 or 60 FPS OBS.
The CTest suite covers numeric/concurrency behavior and bounded PNG replay.
Generate additional tiny ONNX fixtures for foreground validation:

```sh
uv run --with onnx python tests/make-rvm-fixtures.py artifacts/rvm-fixtures
build/rmbg-foreground-tests artifacts/rvm-fixtures
RMBG_TEST_MOTION=1 RMBG_TEST_MATCH_VIDEO=1 RMBG_TEST_FOREGROUND=1 RMBG_TEST_GREEN_FOREGROUND=1 RMBG_TEST_OBS_FPS=60 \
  build/obs-rmbg-smoke "$PWD/build/obs-rmbg.so" "$PWD/data" "$PWD/artifacts/rvm-fixtures/fp32.onnx" "$PWD/input.png" "$PWD/artifacts/foreground-test.png" cpu 30 0 3
```

These fixtures deliberately turn red input into green foreground, so shader use and
pairing can be checked independently of learned quality. `RMBG_TEST_FOREGROUND_TOGGLE=1`
also tests on/off transitions. `RMBG_TEST_PREMULT=1` checks a half-transparent source;
use it with the deterministic RVM foreground fixture. They are test-only controls.

### Status procedure additions

The existing `rmbg_status` procedure keeps all previous outputs. New numeric outputs are
`mask_rate`, `pair_rate`, `distinct_pairs`, `repeated_presentations`, `repeated_window`,
`replaced_frames`, `replaced_window`, `processing_mean_ms`, `processing_p95_ms`,
`queue_mean_ms`, `queue_p95_ms`, `age_mean_ms`, `age_p95_ms`, and `downsample_ratio`.
String outputs are `precision`, `backend`, and `model_file`. Fields ending in `_window`
count the last five seconds; `distinct_pairs`, `repeated_presentations`, `replaced_frames`,
and the original `completed` count the instance's lifetime. Live mode reports no matched
pair rate/repeats. This is a synchronous snapshot API; it does not introduce a background
poller or continuous log stream into OBS.

## Implementation

- `src/model.cpp`: detects model signatures, converts premultiplied RGBA to RGB NCHW
  (RMBG: `channel / 255 - 0.5`; RVM: `channel / 255`), handles FP32/FP16 tensors, and
  runs ONNX. FP16 preprocessing and RVM alpha conversion avoid FP32 intermediate buffers.
  RVM recurrent tensors use device I/O binding and its alpha range is preserved.
- `src/worker.cpp`: loads models off the render thread and processes one in-flight
  frame with one replaceable waiting frame. New captures replace older waiting frames;
  the queue never grows. A bounded buffer pool reuses RGBA storage. Capture identity tokens
  keep matching GPU frames alive without handing graphics objects to the worker.
  Configuration generations invalidate both pending work and results.
- `src/plugin.cpp`: captures at the model's selected size on the GPU, reads back immediately
  or on the next video tick, and uploads masks. Optional full-resolution GPU frame retention
  pairs video and mask by capture identity. Status includes mask age and readback duration.
  Capture scheduling follows the OBS video clock without accumulating timing drift.
- `data/rmbg.effect`: applies the mask to both RGB and alpha for correct premultiplied
  compositing, with threshold, softness, and mask preview controls.

Model inference is asynchronous. Live-video mode combines the current source with the
newest completed mask. Matching mode renders the original captured video with that mask.
Extra temporal smoothing applies only to live-video captures; it adjusts for elapsed time
and resets across source-size changes or long gaps. It cannot eliminate all model flicker.
RMBG identifies salient objects, so it may preserve chairs and other objects as well as people.
RVM is designed for human matting.

## v0.4.0 verification

The dated implementation report is in [docs/v0.4.0-validation.md](docs/v0.4.0-validation.md).
It distinguishes deterministic correctness tests, offline model timings, the isolated
ten-minute NVIDIA/OpenGL replay, and the unfiltered-webcam quality checks still needed.
Do not use older small-image/llvmpipe figures below as predictions for your camera scene.

## Historical verification (v0.1–v0.3)

On Linux Mint 22.3, OBS 32.2.0, Intel Core Ultra 7 265K and NVIDIA RTX 5070 Ti:

- C++ build and core checks passed.
- Deterministic worker concurrency checks cover replacement of waiting frames, overlapping
  capture/inference, model changes while work is in flight, video-token lifetimes, and safe
  capture-buffer reuse. Fused FP16 preprocessing is bit-identical to the former path; direct
  alpha conversion matches it for every finite FP16 value.
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
- Installed v0.3.0, same 512×600 portrait and 30 FPS test: low latency readback plus
  matching video produced **30.0 masks/s** and **34.5 ms** average mask/video age, versus
  **68.3 ms** with deferred readback in the comparison run. The cache used **3** textures
  with no missed captures; map/copy/submission averaged **0.10 ms** (maximum **0.19 ms**,
  rounded up). The earlier live-video low-latency run averaged **35.0 ms** mask age.
  These five-second isolated tests use llvmpipe graphics and NVIDIA CUDA inference;
  hardware graphics readback and your live-camera/recording load can differ.
- A deterministic moving-foreground OBS test produced **zero mismatched opaque foreground
  pixels** in matching mode, including transitions between immediate/deferred readback
  and live/matched output with smoothing saved at 0.9. Live-video controls exposed the
  expected temporal mismatch. Installed v0.3.0 also passed source resizing, RVM/RMBG
  switching, foreground color preservation, missing-model fallback, and CUDA recurrence checks.
- RVM FP16/FP32 CUDA and FP32 CPU passed actual recurrent-state reuse/reset checks.
  The OBS test passed source resizing in both orientations, RVM → RMBG → RVM switching,
  transparency, color preservation, and missing-model passthrough. Both RMBG precisions
  produced pixel-identical CLI output to their pre-RVM baselines.
- Live webcam motion, recording load, and additional OBS plugin combinations still need
  testing on your actual scenes. The inference result is not an end-to-end 30 FPS guarantee.

## Troubleshooting

For incomplete downloads, checksum failures, missing models, and ONNX parse errors,
start with [Repair failed model downloads](#repair-failed-model-downloads).

### Build or plugin loading failures

| Error or symptom | What to check |
| --- | --- |
| CMake cannot find `libobs` / `libobsConfig.cmake` | Install matching OBS 30+ development files. For a custom SDK, pass `-Dlibobs_DIR=/absolute/path/to/directory/containing/libobsConfig.cmake`. The OBS executable alone is not a development SDK. |
| CMake cannot find `onnxruntime_cxx_api.h` or `ORT_LIBRARY` | Bootstrap must finish successfully; point `ONNXRUNTIME_ROOT` at the extracted SDK containing `include/` and `lib/`, not its `.tgz` archive or a Python package. Use the CPU or CUDA path from the build examples. |
| Missing `png.h`, `PNG_LIBRARY`, or `simde/...` headers | Install the libpng/SIMDe development packages. On Ubuntu/Mint, bootstrap's optional `--simde` can extract the pinned SIMDe package locally. |
| `wrong ELF class`, architecture error, `GLIBC_*`/`GLIBCXX_* not found`, or undefined OBS symbols | Check architecture and dependency compatibility. Rebuild on the target Linux machine against its matching OBS SDK; do not use a Linux binary in Windows/macOS or an x86-64 SDK on ARM64. |
| Missing `libonnxruntime.so`, CUDA library, effect, or locale | Reinstall the complete plugin directory with the matching libraries and `data/`, then restart OBS. Copying just `obs-rmbg.so` is insufficient. |
| A CPU setting does not help because the module fails to load | The private CUDA build links NVIDIA libraries directly. Restore those libraries or rebuild/install the CPU-only configuration; model initialization fallback cannot fix an unloaded plugin. |

On native Linux, inspect dependencies of your own installed module:

```sh
ldd "${XDG_CONFIG_HOME:-$HOME/.config}/obs-studio/plugins/obs-rmbg/bin/64bit/obs-rmbg.so"
```

Resolve any `not found` entries. For a CUDA build, also inspect the provider's dependencies:

```sh
ldd "${XDG_CONFIG_HOME:-$HOME/.config}/obs-studio/plugins/obs-rmbg/bin/64bit/libonnxruntime_providers_cuda.so"
nvidia-smi
```

The second library exists only in the CUDA SDK. These checks diagnose library/driver
availability; they do not validate model inference. Use [the image CLI](#test-an-image-without-obs)
with a real local PNG and the verified model, first with `--device cpu` and FP32, then
with `--device cuda` if configured. If the CLI works but OBS fails, compare the model
path and libraries in the installed package with those used by the build.

### OBS status and performance

- **Filter missing:** restart OBS, check the native/Flatpak installation distinction,
  and inspect Help → Log Files → View Current Log for `obs-rmbg`.
- **Ready: CPU:** automatic CUDA initialization fell back. Rebuild with the CUDA SDK and
  private libraries above. Explicit CUDA mode surfaces initialization errors.
- **cuDNN unavailable / old libcudart symbol error:** use the complete private CUDA build;
  the CUDA SDK by itself does not supply all matching NVIDIA runtime libraries.
- **Error after inference starts:** the original source is shown. Fix the runtime/model
  issue and retry, or explicitly select CPU. Initialization fallback does not hide
  runtime inference errors.
- **Cutout trails movement:** confirm **Ready: CUDA**, enable **Low latency readback**, and
  set smoothing to **0** in live-video mode. Enable **Match video to mask** for aligned edges
  with buffered video; visible motion then updates at the completed mask rate.
  Try **30** mask updates if there is spare GPU capacity; return to **15** if mask age or
  OBS rendering lag increases. Refresh status to inspect processing time and mask age. The model
  still needs time to process each frame; increasing the limit above its throughput cannot
  remove that delay. Lowering webcam resolution does not shrink RMBG's fixed 1024×1024 input;
  RVM uses the source dimensions subject to its separate input limit.
- **Whole video stutters:** check OBS's Stats window for rendering/encoding lag and GPU
  contention. Reducing mask updates to **15** can free GPU time, at the cost of slower
  cutout updates. CPU mode is a functionality fallback for this model.

When reporting an unresolved issue, include the OS/distribution and architecture, OBS
version and installation type (native/Flatpak/portable), plugin version or Git commit,
chosen CPU/CUDA build, GPU/driver if applicable, model filename and SHA256, and exact
error. For setup failures, include the failing command and Python/CMake versions.
In OBS, copy the filter's **Status** after refreshing and relevant entries from
**Help → Log Files → View Current Log** (look for `obs-rmbg`, ONNX Runtime, or CUDA).
Check copied paths/logs for personal information before sharing them.
