#!/usr/bin/env python3
"""Download pinned upstream binaries into this checkout (no system changes)."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORT_VERSION = "1.29.0"
ORT_ASSETS = {
    "cpu": ("onnxruntime-linux-x64-1.29.0", "c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba"),
    "cuda": ("onnxruntime-linux-x64-gpu_cuda12-1.29.0", "4ca594a0da83927befbd73fe020d7f569be151d70bb4fe9741ad405f4882e2ad"),
}
MODEL_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"
MODEL_SHA256 = "8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == expected:
        print(f"Verified existing {path}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name} ...", flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "obs-rmbg-bootstrap/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if digest(partial) != expected:
            raise RuntimeError(f"SHA256 mismatch for {path.name}; refusing to use it")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Verified {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=ORT_ASSETS, default="cpu")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--simde", action="store_true", help="Extract Ubuntu's missing SIMDe headers locally")
    args = parser.parse_args()
    name, sha = ORT_ASSETS[args.backend]
    archive = ROOT / ".deps" / (name + ".tgz")
    download(f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.tgz", archive, sha)
    # Python's data filter rejects unsafe archive paths and links.
    with tarfile.open(archive) as tar:
        tar.extractall(ROOT / ".deps", filter="data")
    if not args.skip_model:
        download(f"https://huggingface.co/briaai/RMBG-1.4/resolve/{MODEL_REVISION}/onnx/model.onnx?download=true",
                 ROOT / "data/models/rmbg-1.4.onnx", MODEL_SHA256)
    if args.simde:
        package = ROOT / ".deps/libsimde-dev_0.7.2-6_all.deb"
        download("https://archive.ubuntu.com/ubuntu/pool/universe/s/simde/libsimde-dev_0.7.2-6_all.deb", package,
                 "153006966f7b0049f9ebe6ec32641a71c064f9812fcccfa2cc72b2808e73b970")
        subprocess.run(["dpkg-deb", "--extract", str(package), str(ROOT / ".deps/sysroot")], check=True)
    print(f"Configure with: cmake -S . -B build -DONNXRUNTIME_ROOT={ROOT / '.deps' / name}")


if __name__ == "__main__":
    main()
