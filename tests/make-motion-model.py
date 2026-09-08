"""Generate a tiny deterministic RMBG-shaped ONNX fixture, not a trained model.

Run: uv run --with onnx python tests/make-motion-model.py artifacts/motion.onnx
The red channel is the foreground label, so moving video/mask mismatches can be
measured independently of a learned model's accuracy or recurrent state.
"""
import argparse
from pathlib import Path
import onnx
from onnx import TensorProto, helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--work-layers", type=int, default=0,
                    help="Add identity convolutions to exercise CPU overload without changing alpha")
args = parser.parse_args()
if not 0 <= args.work_layers <= 256:
    parser.error("--work-layers must be 0..256")
nodes = [helper.make_node("Gather", ["input", "red"], ["red_channel"], axis=1)]
constants = [helper.make_tensor("red", TensorProto.INT64, [1], [0])]
previous = "red_channel"
if args.work_layers:
    constants.append(helper.make_tensor("identity_kernel", TensorProto.FLOAT, [1, 1, 3, 3],
                                        [0, 0, 0, 0, 1, 0, 0, 0, 0]))
for layer in range(args.work_layers):
    output = f"work_{layer}"
    nodes.append(helper.make_node("Conv", [previous, "identity_kernel"], [output], pads=[1, 1, 1, 1]))
    previous = output
nodes.append(helper.make_node("Identity", [previous], ["mask"]))
graph = helper.make_graph(
    nodes,
    "moving-foreground-test",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1024, 1024])],
    [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 1024, 1024])],
    constants,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
destination = args.destination
destination.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, destination)
