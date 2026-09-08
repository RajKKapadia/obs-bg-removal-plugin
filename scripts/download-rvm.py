#!/usr/bin/env python3
"""Download pinned official RVM ONNX models and their license (MobileNetV3 by default)."""
import argparse
from bootstrap import download, ROOT

RELEASE = "v1.0.0"
SOURCE_REVISION = "53d74c6826735f01f4406b5ca9075eee27bec094"
MODELS = {
    "mobilenetv3": {
        "fp32": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
        "fp16": "6a0d5ce6cc17702613be548559879b4521ed424cfe14ddc48d1acaa44d616f64",
    },
    "resnet50": {
        "fp32": "25db300fcb6ee27f941a1b52c97856e8d1f13c7f35817f81a612f89af0e8a85c",
        "fp16": "a9266f5046411d604bbff38e18c54bf8c70d85d93bbc400564697590f5712738",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    parser.add_argument("--variant", choices=["mobilenetv3", "resnet50", "both"], default="mobilenetv3")
    args = parser.parse_args()
    for variant, models in MODELS.items():
        if args.variant not in (variant, "both"):
            continue
        for precision, checksum in models.items():
            if args.precision not in (precision, "both"):
                continue
            name = f"rvm_{variant}_{precision}.onnx"
            download(f"https://github.com/PeterL1n/RobustVideoMatting/releases/download/{RELEASE}/{name}",
                     ROOT / "data/models" / name, checksum)
    download(f"https://raw.githubusercontent.com/PeterL1n/RobustVideoMatting/{SOURCE_REVISION}/LICENSE",
             ROOT / "data/licenses/rvm/LICENSE", "8b1ba204bb69a0ade2bfcf65ef294a920f6bb361b317dba43c7ef29d96332b9b")
    print("Choose an RVM ONNX file in the filter's model picker. Benchmark FP32 and FP16 on CUDA; FP32 also supports CPU.")


if __name__ == "__main__":
    main()
