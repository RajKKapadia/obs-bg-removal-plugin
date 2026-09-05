#!/usr/bin/env python3
"""Stage private CUDA libraries from pinned NVIDIA wheels or an existing uv cache."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from bootstrap import ROOT, download

PACKAGES = {
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cublas-cu12": "12.9.1.4",
    "nvidia-cudnn-cu12": "9.10.2.21",
}


def cached(package, version):
    cache = Path.home() / ".cache/uv/archive-v0"
    name = package.replace("-", "_")
    return next(cache.glob(f"*/{name}-{version}.dist-info/METADATA"), None)


def copy_cached(metadata, destination):
    root = metadata.parent.parent
    # Validate cached library bytes using the wheel's RECORD hashes before copying.
    with (metadata.parent / "RECORD").open(newline="") as stream:
        for name, checksum, _ in csv.reader(stream):
            source = root / name
            if "/lib/" not in name or ".so" not in source.name:
                continue
            if not checksum.startswith("sha256="):
                raise RuntimeError(f"Missing SHA256 for {source}")
            expected = checksum.split("=", 1)[1]
            with source.open("rb") as file:
                actual = base64.urlsafe_b64encode(hashlib.file_digest(file, "sha256").digest()).decode().rstrip("=")
            if actual != expected:
                raise RuntimeError(f"Cached CUDA library checksum mismatch: {source}")
            shutil.copy2(source, destination / source.name)
    licenses = metadata.parent / "licenses"
    if licenses.exists():
        shutil.copytree(licenses, destination.parent / "licenses" / metadata.parent.name, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Never download; fail if pinned wheels aren't cached")
    args = parser.parse_args()
    target = ROOT / ".deps/cuda/lib"
    target.mkdir(parents=True, exist_ok=True)
    for package, version in PACKAGES.items():
        metadata = cached(package, version)
        if metadata:
            print(f"Verifying cached {package} {version}", flush=True)
            copy_cached(metadata, target)
        else:
            if args.cache_only: raise RuntimeError(f"Not cached: {package}=={version}")
            with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
                info = json.load(response)
            wheel = next(f for f in info["urls"] if "manylinux" in f["filename"] and "x86_64" in f["filename"])
            archive = ROOT / ".deps" / wheel["filename"]
            download(wheel["url"], archive, wheel["digests"]["sha256"])
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    path = Path(name)
                    if "/lib/" in name and ".so" in path.name:
                        with z.open(name) as source, (target / path.name).open("wb") as output:
                            shutil.copyfileobj(source, output)
                    elif "/licenses/" in name and not name.endswith("/"):
                        directory = target.parent / "licenses" / package
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / path.name).write_bytes(z.read(name))
    (target.parent / "versions.json").write_text(json.dumps(PACKAGES, indent=2) + "\n")
    print(f"Private CUDA libraries ready: {target}")
    print(f"Configure with -DRMBG_CUDA_LIBRARY_DIR={target}")


if __name__ == "__main__":
    main()
