#pragma once
#include "model.hpp"

namespace rmbg {
// Person gate: an RVM matte limits where a salient-object matte (RMBG or V-RMBG)
// may keep pixels. Objects in front of the person stay; objects beside them go.
struct GateShape {
    float threshold = 0.5f; // RVM alpha at or above this counts as person.
    float expand = 2.0f;    // Grow the person region, in percent of frame height.
    float feather = 1.0f;   // Soften the gate edge, in percent of frame height.
};
constexpr uint32_t kGateLongEdge = 640; // RVM input for the gate; edges come from the main model.

// Area-average resample of premultiplied RGBA. Copies timing and source metadata,
// never the capture token, so the gate never pins a cached video frame.
Frame resize_frame(const Frame &frame, ImageSize size);
// Binarize, dilate, then box-feather an 8-bit matte into an 8-bit gate.
std::vector<uint8_t> person_gate(const std::vector<uint8_t> &alpha, uint32_t width, uint32_t height, const GateShape &shape);
// Model settings for the RVM gate session, inheriting device and threads.
ModelConfig gate_model_config(const ModelConfig &main);
// Runs the gate model on a resampled copy of frame and stores the gate in mask.
void apply_person_gate(Model &gate, const Frame &frame, Mask &mask, const GateShape &shape);
}
