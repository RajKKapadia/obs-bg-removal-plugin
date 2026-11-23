"""Generate untrained V-RMBG streaming-contract fixtures for adapter tests."""
import sys
from pathlib import Path
import onnx
from onnx import TensorProto as T, helper as h

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
for name in ("fp32", "fp16", "render", "nan", "missing-metadata", "wrong-contract", "wrong-size", "wrong-type"):
    dtype = T.FLOAT16 if name == "fp16" else T.FLOAT
    size = 64 if name == "wrong-size" else 1024
    inputs = [h.make_tensor_value_info(key, dtype, [1, 3, size, size]) for key in ("src", "prev_foreground")]
    state_type = T.FLOAT16 if name == "wrong-type" else dtype
    outputs = [h.make_tensor_value_info("pha", dtype, [1, 1, size, size]),
               h.make_tensor_value_info("next_foreground", state_type, [1, 3, size, size])]
    constants = [h.make_tensor("red", T.INT64, [1], [0]),
                 h.make_tensor("weight", dtype, [1], [32 if name == "render" else 0.125]),
                 h.make_tensor("offset", dtype, [1], [-51.2 if name == "render" else 0.0007])]
    # Output intentionally stays in a narrow range: min/max normalization would
    # destroy it. Feedback changes subsequent alpha and detects quantization,
    # stale source RGB, state mixing between channels, and missing state reuse.
    nodes = [h.make_node("Gather", ["src", "red"], ["current_red"], axis=1),
             h.make_node("Gather", ["prev_foreground", "red"], ["previous_red"], axis=1)]
    prior = "previous_red"
    if name == "render":
        # Give the motion harness a nearly binary red matte. Its opaque-color
        # oracle deliberately excludes interpolated boundary pixels. Keep a
        # small state dependency without widening the synthetic outline.
        constants.append(h.make_tensor("prior_weight", dtype, [1], [0.001]))
        nodes.append(h.make_node("Mul", [prior, "prior_weight"], ["small_prior"]))
        prior = "small_prior"
    nodes += [h.make_node("Add", ["current_red", prior], ["sum"]),
             h.make_node("Mul", ["sum", "weight"], ["scaled"]),
             h.make_node("Add", ["scaled", "offset"], ["logits"]),
             h.make_node("Sigmoid", ["logits"], ["alpha"])]
    if name == "nan":
        constants.append(h.make_tensor("invalid", dtype, [1], [float("nan")]))
        nodes.append(h.make_node("Mul", ["alpha", "invalid"], ["pha"]))
    else:
        nodes.append(h.make_node("Identity", ["alpha"], ["pha"]))
    nodes += [h.make_node("Mul", ["src", "pha"], ["feedback"]),
              h.make_node("Cast", ["feedback"], ["next_foreground"], to=state_type)]
    model = h.make_model(h.make_graph(nodes, name, inputs, outputs, constants),
                         opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    if name != "missing-metadata":
        h.set_model_props(model, {"obs_bg_removal.model": "bria-vrmbg-3.0",
                                 "obs_bg_removal.contract": "2" if name == "wrong-contract" else "1"})
    onnx.checker.check_model(model)
    onnx.save(model, destination / f"{name}.onnx")
