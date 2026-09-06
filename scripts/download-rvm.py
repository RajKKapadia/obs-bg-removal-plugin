#!/usr/bin/env python3
"""Download the official RVM MobileNetV3 ONNX release and its license."""
import argparse
from bootstrap import download, ROOT

RELEASE = "v1.0.0"
SOURCE_REVISION = "53d74c6826735f01f4406b5ca9075eee27bec094"
MODELS = {
    "fp32": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
    "fp16": "6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    args = parser.parse_args()
    for precision, checksum in MODELS.items():
        if args.precision not in (precision, "both"):
            continue
        name = f"rvm_mobilenetv3_{precision}.onnx"
        download(f"https://github.com/PeterL1n/RobustVideoMatting/releases/download/{RELEASE}/{name}",
                 ROOT / "data/models" / name, checksum)
    download(f"https://raw.githubusercontent.com/PeterL1n/RobustVideoMatting/{SOURCE_REVISION}/LICENSE",
             ROOT / "data/licenses/rvm/LICENSE", "8b1ba204bb69a0ade2bfcf65ef294a920f6bb361b317dba43c7ef29d96332b9b")
    print("Choose an RVM ONNX file in the filter's model picker. Use FP16 with CUDA; FP32 also supports CPU.")


if __name__ == "__main__":
    main()
