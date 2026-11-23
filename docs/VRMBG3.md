# Experimental BRIA V-RMBG 3.0 integration

The plugin has an experimental adapter and ONNX export workflow for
[BRIA V-RMBG 3.0](https://huggingface.co/briaai/VRMBG-3.0). The implementation follows
BRIA's published preprocessing and temporal-feedback example. **The gated architecture
and weights were unavailable during development on 2026-10-02. Real-model conversion,
ONNX operator compatibility, memory use, webcam quality, and live performance remain
unverified.** Synthetic fixtures verify the integration, not BRIA's learned model.
Keep the current RVM model for live use until those checks pass.

## Access and export

BRIA must approve access through the official repository. Use the terms attached to your
approval; commercial use requires a BRIA agreement. No weights or upstream architecture
are bundled, and the plugin's GPL license grants no model rights. The script does not
request access or accept terms for you.

Check your existing Hugging Face login without installing export dependencies:

```sh
python3 scripts/export-vrmbg3.py --check-access
```

The check reads `HF_TOKEN` or the cached Hugging Face token and reports an HTTP status
without printing credentials. HTTP 403 can mean access is still pending. Re-run after
approval. Export will also stop on denied access.

For an initial FP32 CPU export, create a private Python environment in this checkout:

```sh
uv venv --python 3.12 .deps/vrmbg3/export-venv
uv pip install --python .deps/vrmbg3/export-venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .deps/vrmbg3/export-venv/bin/python transformers huggingface_hub safetensors timm einops numpy onnx onnxruntime
```

These cover the exporter and common backbone dependencies. The gated architecture may
require additional packages or compatible versions; that dependency set is not yet
validated. The synthetic exporter test passed with PyTorch 2.8.0+cpu, ONNX 1.23.1 and
Python ONNX Runtime 1.30.0. The native plugin was tested with ONNX Runtime 1.29.0.

Get a **full 40-character commit SHA** from the approved repository's Files and versions
history. Replace the placeholder below with that SHA:

```sh
VRMBG_REVISION=REPLACE_WITH_APPROVED_40_CHARACTER_COMMIT_SHA
.deps/vrmbg3/export-venv/bin/python scripts/export-vrmbg3.py --revision "$VRMBG_REVISION" --device cpu --precision fp32 --output data/models/vrmbg3_fp32.onnx
```

The script downloads only the named official files at that revision into `.deps/vrmbg3/`.
It executes the pinned upstream architecture with `trust_remote_code=True` and loads
safetensors. Inspect that authorized source before running it. An authorized local
snapshot can instead be supplied with `--source-dir /absolute/path/to/snapshot`; it must
contain `config.json`, `model.py`, `vrmbg3_config.py` and `model.safetensors`. For local
snapshots, the supplied revision is your provenance claim; the script additionally
records file hashes but cannot authenticate the directory's origin.

CUDA export needs a CUDA-enabled PyTorch installation and `onnxruntime-gpu` in place of
`onnxruntime`, with compatible CUDA/cuDNN libraries. Use the corresponding
[PyTorch installation instructions](https://pytorch.org/get-started/locally/) and
[ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Then use `--device cuda --precision fp16 --output data/models/vrmbg3_fp16.onnx`.
FP16 export requires CUDA; a failed CUDA initialization is not silently validated on CPU.
The FP16 adapter has fixture coverage, but real FP16 export has not been tested.

The exporter writes an ONNX model and adjacent `.validation.json` only after ONNX
checking and three-frame numerical parity with the upstream PyTorch call succeed.
Validation includes first-frame zero state, feedback reuse, and a zero-state restart.
The report records tolerances, errors, versions, revision and SHA256 hashes. Existing
output files are never overwritten. Choose a new path to retry an export.

## Select and compare in OBS

Build and install the plugin using the [README](../README.md#build-linux), then restart
OBS. Select the exported ONNX file through **Model file** and refresh status. It should
identify **V-RMBG 3.0 (experimental)** and its precision. Models remain at their selected
local path; CMake does not bundle V-RMBG exports into an installation.

Start with CUDA, immediate readback, matching enabled, smoothing 0, threshold 0.5 and
softness 0.5. Set the update limit according to measured throughput. The RVM input limit,
downsample ratio, foreground-color option and quality preset do not apply. Capture is
fixed at 1024 × 1024 regardless of camera resolution; full-resolution video is retained
for matching. The source supplies RGB and the model supplies direct alpha.

Feedback tensors stay on CUDA between processed frames when CUDA is selected. State
resets after a source resize, model/settings reload, non-increasing timestamp or a gap
of at least one second. Worker/UI smoothing and threshold changes never enter model
feedback. Matching uses the existing bounded cache and latest-frame queue.

Compare the same **unfiltered** motion footage against the current RVM baseline:

```sh
build/rmbg-sequence data/models/vrmbg3_fp32.onnx artifacts/raw-motion artifacts/vrmbg3-comparison --device cuda --fps 30
python3 scripts/benchmark-motion.py raw-webcam.mkv artifacts/live-vrmbg3 --model data/models/vrmbg3_fp32.onnx --input-kind raw --seconds 600 --encode-load
```

Use the [existing comparison and live benchmark workflow](../README.md#reproduce-motion-and-model-comparisons)
to inspect hair, hands, fast turns and background leakage, and measure OBS mask age,
rendering/encoding load and memory. A successful export does not establish a quality
improvement or 30 FPS. The existing native CUDA arena limit is 2 GiB; actual V-RMBG
requirements may require follow-up work after profiling the authorized model.

## Reproduce synthetic checks

Development checks on 2026-10-02 passed: native build and all four configured CTest
checks; FP32/FP16 streaming fixtures on CPU and CUDA; the synthetic exporter and its
output loaded by the native adapter; isolated OBS motion/matching and saved-RVM-setting
checks on CPU with llvmpipe graphics; and real RVM FP32 regressions on CPU and CUDA.
The OBS fixture had zero mismatched opaque foreground pixels across 90 measured frames
and passed missing-model passthrough. Its timings are not measurements of BRIA's model.

From the repository root, after configuring the native build:

```sh
uv run --python 3.12 --with onnx python tests/make-vrmbg3-fixtures.py artifacts/vrmbg3-fixtures
cmake -S . -B build -DVRMBG3_TEST_FIXTURES="$PWD/artifacts/vrmbg3-fixtures"
cmake --build build -j4
ctest --test-dir build --output-on-failure
build/vrmbg3-integration artifacts/vrmbg3-fixtures cuda
```

The tests cover FP32/FP16 normalization, raw-alpha feedback, device residency, continuity
resets, invalid contracts, non-finite alpha, display smoothing and matched-frame bypass.
For exporter regression checks, use a new output directory each time:

```sh
.deps/vrmbg3/export-venv/bin/python tests/vrmbg3-export-tests.py artifacts/vrmbg3-export-new
```

For a synthetic moving red subject on blue, provide any local opaque PNG at least
64 pixels per side; the harness uses its dimensions and replaces its pixels:

```sh
RMBG_TEST_VRMBG3=1 RMBG_TEST_MOTION=1 RMBG_TEST_MATCH_VIDEO=1 RMBG_TEST_FOREGROUND=1 \
  build/obs-rmbg-smoke "$PWD/build/obs-rmbg.so" "$PWD/data" "$PWD/artifacts/vrmbg3-fixtures/render.onnx" /absolute/path/to/opaque-input.png "$PWD/artifacts/vrmbg3-obs.png" cpu 15 0.8 3
```

This checks registration, direct-alpha compositing, matching, ignoring a saved RVM
foreground-color setting, and missing-model passthrough. It requires a working OpenGL
display and uses an isolated libobs scene.

## ONNX contract

This plugin-specific wrapper exposes two inputs and outputs around BRIA's six-channel
paired input. Arbitrary upstream or third-party ONNX exports are not accepted.

| Tensor | Direction | Fixed NCHW shape | Meaning |
| --- | --- | --- | --- |
| `src` | Input | `[1,3,1024,1024]` | RGB in [0,1], normalized with means `[0.485,0.456,0.406]` and standard deviations `[0.229,0.224,0.225]`. |
| `prev_foreground` | Input | `[1,3,1024,1024]` | Previous normalized RGB multiplied by previous raw predicted alpha; zeros initially. |
| `pha` | Output | `[1,1,1024,1024]` | Sigmoid of the final model logits; no per-frame min/max normalization. |
| `next_foreground` | Output | `[1,3,1024,1024]` | `src * pha`, retained for the next processed frame. |

All tensors have the same FP32 or FP16 precision. Required custom metadata is
`obs_bg_removal.model=bria-vrmbg-3.0` and `obs_bg_removal.contract=1`.
