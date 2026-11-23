#!/usr/bin/env python3
"""Export an authorized BRIA V-RMBG 3.0 snapshot to the plugin's ONNX contract.

Does not request access or accept license terms. Downloads only from the official
repository at an explicitly supplied full commit SHA. --check-access needs only
Python's standard library; exporting needs torch, transformers, onnx and ORT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "briaai/VRMBG-3.0"
CONTRACT = {"obs_bg_removal.model": "bria-vrmbg-3.0", "obs_bg_removal.contract": "1"}
FILES = ["config.json", "model.py", "vrmbg3_config.py", "model.safetensors", "README.md"]


def check_access():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    token_path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token"))
    if not token and token_path.is_file():
        token = token_path.read_text().strip()
    request = urllib.request.Request(
        f"https://huggingface.co/{REPOSITORY}/resolve/main/config.json",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.loads(response.read(4096))
        print("V-RMBG 3.0 access granted. Export requires a pinned revision and authorized use.")
        return 0
    except urllib.error.HTTPError as error:
        # Never print tokens, request headers, URLs with credentials, or HTTP bodies.
        print(f"V-RMBG 3.0 access unavailable (HTTP {error.code}). Check your Hugging Face login and BRIA access approval.")
        return 1
    except (urllib.error.URLError, ValueError):
        print("V-RMBG 3.0 access check failed: network unavailable or invalid response.")
        return 1


def make_wrapper(model):
    import torch

    class StreamingExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, src, prev_foreground):
            # src has BRIA's ImageNet normalization. Its reference implementation
            # feeds normalized previous RGB * unmodified predicted alpha, not raw
            # RGB and not the UI-remapped or byte-quantized display mask.
            paired = torch.cat((src, prev_foreground), dim=1)
            logits = self.model(paired.contiguous(memory_format=torch.channels_last))[-1]
            pha = logits.sigmoid()
            return pha, src * pha

    return StreamingExport().eval()


def export_checked(model, output, *, device, revision, source_hashes=None):
    """Publish only after ONNX checking and multi-frame PyTorch/ORT parity pass."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    output = Path(output)
    report_path = output.with_suffix(".validation.json")
    if output.exists() or report_path.exists():
        raise RuntimeError("Output or validation report already exists; choose a new output path.")
    wrapper = make_wrapper(model)
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float32, torch.float16):
        raise RuntimeError("Only FP32 and FP16 exports are supported.")
    if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA validation requires onnxruntime-gpu with a working CUDA provider.")
    src = torch.zeros((1, 3, 1024, 1024), device=device, dtype=dtype)
    state = torch.zeros_like(src)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vrmbg3-export-", dir=output.parent) as work:
        candidate = Path(work) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(wrapper, (src, state), str(candidate),
                              input_names=["src", "prev_foreground"],
                              output_names=["pha", "next_foreground"],
                              opset_version=17, dynamo=False, external_data=False)
        graph = onnx.load(candidate)
        onnx.helper.set_model_props(graph, {
            **CONTRACT, "obs_bg_removal.revision": revision,
            "obs_bg_removal.source": REPOSITORY,
        })
        onnx.checker.check_model(graph)
        onnx.save(graph, candidate)
        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(candidate), providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA initialization failed; refusing to validate a CPU fallback.")
        rng = np.random.default_rng(20261002)
        ort_state = state.cpu().numpy()
        rows = []
        half = dtype == torch.float16
        atol, rtol = (0.005, 0.01) if half else (0.0002, 0.002)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
        for index in range(3):
            # Frame 3 also checks zero-state restart after a preceding sequence.
            if index == 2:
                state = torch.zeros_like(state)
                ort_state = np.zeros_like(ort_state)
            rgb = rng.random((1, 3, 1024, 1024), dtype=np.float32)
            normalized = ((rgb - mean) / std).astype(np.float16 if half else np.float32)
            src = torch.from_numpy(normalized).to(device)
            with torch.inference_mode():
                # Compare against the upstream call directly, independently of
                # the export wrapper, including the upstream final sigmoid.
                ref_alpha = model(torch.cat((src, state), 1).contiguous(memory_format=torch.channels_last))[-1].sigmoid()
                ref_state = src * ref_alpha
            actual = session.run(["pha", "next_foreground"], {"src": normalized, "prev_foreground": ort_state})
            reference = [ref_alpha.cpu().numpy(), ref_state.cpu().numpy()]
            errors = {}
            for name, predicted, expected in zip(("pha", "next_foreground"), actual, reference):
                shape = (1, 1 if name == "pha" else 3, 1024, 1024)
                if predicted.shape != shape or predicted.dtype != normalized.dtype or not np.isfinite(predicted).all():
                    raise RuntimeError(f"Invalid {name} output on validation frame {index + 1}.")
                np.testing.assert_allclose(predicted, expected, atol=atol, rtol=rtol,
                                           err_msg=f"{name} parity failed on frame {index + 1}")
                errors[name] = float(np.max(np.abs(predicted.astype(np.float32) - expected.astype(np.float32))))
            if np.min(actual[0]) < 0 or np.max(actual[0]) > 1:
                raise RuntimeError("Export returned alpha outside [0,1].")
            rows.append({"frame": index + 1, "state_reset": index != 1, "max_abs_error": errors})
            state, ort_state = ref_state, actual[1]
        report = {
            "repository": REPOSITORY, "revision": revision, "contract": CONTRACT,
            "precision": "fp16" if half else "fp32", "providers": session.get_providers(),
            "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "atol": atol, "rtol": rtol, "frames": rows,
            "onnx_sha256": digest(candidate),
            "source_sha256": source_hashes or {},
            "scope": "Synthetic multi-frame numerical parity only; not webcam quality or OBS throughput validation.",
        }
        # No model file is exposed at the destination until validation succeeds.
        del session
        candidate_report = Path(work) / "validation.json"
        candidate_report.write_text(json.dumps(report, indent=2) + "\n")
        candidate_report.replace(report_path)
        try:
            candidate.replace(output)
        except OSError:
            report_path.unlink()
            raise
    return report


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-access", action="store_true")
    parser.add_argument("--revision", help="Full 40-character commit SHA of the authorized upstream snapshot")
    parser.add_argument("--source-dir", type=Path, help="Use an already downloaded authorized snapshot")
    parser.add_argument("--output", type=Path, default=ROOT / "data/models/vrmbg3_fp32.onnx")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.check_access:
        return check_access()
    if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must pin a full upstream commit SHA; floating revisions are not accepted")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 export/validation requires --device cuda")
    if args.output.exists() or args.output.with_suffix(".validation.json").exists():
        parser.error("Output already exists; choose a new output path")
    try:
        # Keep generated modules/backbone caches inside this ignored checkout.
        os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".deps/vrmbg3/modules"))
        os.environ.setdefault("TORCH_HOME", str(ROOT / ".deps/vrmbg3/torch"))
        from huggingface_hub import snapshot_download
        import torch
        from transformers import AutoModelForImageSegmentation
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch.")
        source = args.source_dir
        if source is None:
            source = Path(snapshot_download(REPOSITORY, revision=args.revision,
                          cache_dir=ROOT / ".deps/vrmbg3/hub", allow_patterns=FILES))
        for name in FILES[:-1]:
            if not (source / name).is_file():
                raise RuntimeError(f"Authorized snapshot is missing {name}.")
        source_hashes = {name: digest(source / name) for name in FILES if (source / name).is_file()}
        # This executes the authorized pinned architecture; it does not load
        # legacy pickle weights or silently fall back to another model family.
        model = AutoModelForImageSegmentation.from_pretrained(
            str(source), trust_remote_code=True, local_files_only=True, use_safetensors=True,
        ).eval().to(device=args.device, dtype=torch.float16 if args.precision == "fp16" else torch.float32)
        export_checked(model, args.output, device=args.device, revision=args.revision, source_hashes=source_hashes)
        report_path = args.output.with_suffix(".validation.json")
        print(f"Validated ONNX export: {args.output}\nNumerical parity/provenance report: {report_path}")
        print("Choose the ONNX file in OBS. Measure real footage before adopting it for live use.")
        return 0
    except ImportError:
        print("Missing export dependencies. See docs/VRMBG3.md for the private environment setup.")
        return 1
    except Exception as error:
        # Hub errors can include authenticated request metadata. Report their
        # type only; ORT/PyTorch/checker failures carry useful local diagnostics.
        if type(error).__module__.startswith(("huggingface_hub", "httpx", "requests")):
            print(f"Model download/access failed ({type(error).__name__}). BRIA must approve access first.")
        else:
            print(f"Export failed ({type(error).__name__}): {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
