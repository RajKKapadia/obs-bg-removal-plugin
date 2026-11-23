"""Exercise the exporter with an untrained tiny model, without BRIA weights/access."""
import importlib.util
import json
from pathlib import Path
import sys
import onnx
import torch

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vrmbg3_export", root / "scripts/export-vrmbg3.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


class TinyVideoModel(torch.nn.Module):
    def __init__(self, *, invalid=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float("nan") if invalid else 0.125))

    def forward(self, paired):
        return [(paired[:, :1] + paired[:, 3:4]) * self.scale + 0.0007]


output = destination / "fp32.onnx"
report = exporter.export_checked(TinyVideoModel().eval(), output, device="cpu", revision="0" * 40)
assert output.is_file() and len(report["frames"]) == 3
assert [row["state_reset"] for row in report["frames"]] == [True, False, True]
assert json.loads(output.with_suffix(".validation.json").read_text())["onnx_sha256"] == exporter.digest(output)
graph = onnx.load(output)
assert {prop.key: prop.value for prop in graph.metadata_props}.items() >= exporter.CONTRACT.items()
assert [value.name for value in graph.graph.input] == ["src", "prev_foreground"]
assert [value.name for value in graph.graph.output] == ["pha", "next_foreground"]

try:
    exporter.export_checked(TinyVideoModel().eval(), output, device="cpu", revision="0" * 40)
    raise AssertionError("An existing model must not be overwritten")
except RuntimeError as error:
    assert "already exists" in str(error)

invalid = destination / "invalid.onnx"
try:
    exporter.export_checked(TinyVideoModel(invalid=True).eval(), invalid, device="cpu", revision="0" * 40)
    raise AssertionError("A non-finite model must fail validation")
except RuntimeError as error:
    assert "Invalid pha output" in str(error)
assert not invalid.exists() and not invalid.with_suffix(".validation.json").exists()
print("Synthetic exporter: three-frame/reset parity, contract metadata, overwrite guard and failed-export cleanup passed.")
