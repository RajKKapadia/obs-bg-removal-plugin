"""Generate tiny deterministic RVM-signature fixtures; no trained weights.

uv run --with onnx python tests/make-rvm-fixtures.py artifacts/rvm-fixtures
Alpha follows input red; foreground is green. Invalid fixtures test rejection.
"""
import sys
from pathlib import Path
import onnx
from onnx import TensorProto as T, helper as h

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
for name in ("fp32", "fp16", "nan", "shape", "type"):
    dtype = T.FLOAT16 if name == "fp16" else T.FLOAT
    inputs = [h.make_tensor_value_info("src", dtype, [1, 3, "h", "w"]),
              h.make_tensor_value_info("downsample_ratio", T.FLOAT, [1])]
    outputs = [h.make_tensor_value_info("pha", dtype, [1, 1, "h", "w"])]
    constants = [h.make_tensor("red", T.INT64, [1], [0]), h.make_tensor("zero", dtype, [1], [0])]
    nodes = [h.make_node("Gather", ["src", "red"], ["pha"], axis=1),
             h.make_node("Mul", ["pha", "zero"], ["black"]),
             h.make_node("Concat", ["black", "pha", "black"], ["rgb"], axis=1)]
    for i in range(1, 5):
        inputs.append(h.make_tensor_value_info(f"r{i}i", dtype, [1, "c", "rh", "rw"]))
        outputs.append(h.make_tensor_value_info(f"r{i}o", dtype, [1, "c", "rh", "rw"]))
        nodes.append(h.make_node("Identity", [f"r{i}i"], [f"r{i}o"]))
    output_type = dtype
    if name == "nan":
        constants.append(h.make_tensor("nan_value", dtype, [1], [float("nan")]))
        nodes.append(h.make_node("Mul", ["rgb", "nan_value"], ["fgr"]))
    elif name == "shape":
        nodes.append(h.make_node("Gather", ["rgb", "red"], ["fgr"], axis=3))
    elif name == "type":
        output_type = T.FLOAT16
        nodes.append(h.make_node("Cast", ["rgb"], ["fgr"], to=output_type))
    else:
        nodes.append(h.make_node("Identity", ["rgb"], ["fgr"]))
    outputs.append(h.make_tensor_value_info("fgr", output_type, [1, 3, "h", "fw"]))
    model = h.make_model(h.make_graph(nodes, name, inputs, outputs, constants),
                         opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, destination / f"{name}.onnx")
