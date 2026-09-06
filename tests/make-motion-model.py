"""Generate a tiny deterministic RMBG-shaped ONNX fixture, not a trained model.

Run: uv run --with onnx python tests/make-motion-model.py artifacts/motion.onnx
The red channel is the foreground label, so moving video/mask mismatches can be
measured independently of a learned model's accuracy or recurrent state.
"""
import sys
from pathlib import Path
import onnx
from onnx import TensorProto, helper

graph = helper.make_graph(
    [helper.make_node("Gather", ["input", "red"], ["mask"], axis=1)],
    "moving-foreground-test",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1024, 1024])],
    [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 1024, 1024])],
    [helper.make_tensor("red", TensorProto.INT64, [1], [0])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, destination)
