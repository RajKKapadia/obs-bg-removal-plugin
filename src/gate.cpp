#include "gate.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace rmbg {
namespace {
struct Tap { uint32_t index; float weight; };
// For each output sample, the source samples it covers and their area weights.
std::vector<std::vector<Tap>> area_taps(uint32_t source, uint32_t target)
{
    std::vector<std::vector<Tap>> taps(target);
    const double scale = double(source) / target;
    for (uint32_t o = 0; o < target; ++o) {
        double begin = o * scale, end = (o + 1) * scale;
        if (scale < 1) { // Upsampling: interpolate between the two nearest centers.
            const double center = std::clamp((o + 0.5) * scale - 0.5, 0.0, double(source - 1));
            const auto low = uint32_t(center), high = std::min(low + 1, source - 1);
            const float t = float(center - low);
            taps[o] = {{low, 1 - t}, {high, t}};
            continue;
        }
        float total = 0;
        for (auto s = uint32_t(begin); s < source && s < end; ++s) {
            const float weight = float(std::min(end, s + 1.0) - std::max(begin, double(s)));
            if (weight > 0) { taps[o].push_back({s, weight}); total += weight; }
        }
        for (auto &tap : taps[o]) tap.weight /= total;
    }
    return taps;
}
// Sliding extrema on a grayscale matte. Truncate the window at the frame edges.
template<class Compare>
void extrema(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, int radius, Compare compare)
{
    std::vector<uint8_t> scratch(pixels.size());
    std::vector<int> candidates(std::max(width, height));
    auto line = [&](const uint8_t *in, uint8_t *out, int count, size_t stride) {
        int head = 0, tail = 0, next = 0;
        for (int i = 0; i < count; ++i) {
            while (head < tail && candidates[head] < i - radius) ++head;
            for (; next < count && next <= i + radius; ++next) {
                while (head < tail && compare(in[next * stride], in[candidates[tail - 1] * stride])) --tail;
                candidates[tail++] = next;
            }
            out[i * stride] = in[candidates[head] * stride];
        }
    };
    for (uint32_t y = 0; y < height; ++y) line(&pixels[size_t(y) * width], &scratch[size_t(y) * width], width, 1);
    for (uint32_t x = 0; x < width; ++x) line(&scratch[x], &pixels[x], height, width);
}
// Box blur of one line; edge samples average only the pixels inside the image.
void blur_line(const uint8_t *in, uint8_t *out, size_t count, size_t stride, int radius)
{
    uint32_t sum = 0;
    const int n = int(count);
    int low = 0, high = std::min(radius, n - 1);
    for (int i = low; i <= high; ++i) sum += in[i * stride];
    for (int i = 0; i < n; ++i) {
        out[i * stride] = uint8_t((sum + (high - low + 1) / 2) / (high - low + 1));
        if (i - radius >= 0) { sum -= in[(i - radius) * stride]; ++low; }
        if (i + radius + 1 < n) { sum += in[(i + radius + 1) * stride]; ++high; }
    }
}
template<class Line> void separable(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, int radius, Line line)
{
    if (radius <= 0) return;
    std::vector<uint8_t> scratch(pixels.size());
    for (uint32_t y = 0; y < height; ++y) line(&pixels[size_t(y) * width], &scratch[size_t(y) * width], width, 1, radius);
    for (uint32_t x = 0; x < width; ++x) line(&scratch[x], &pixels[x], height, width, radius);
}
}

Frame resize_frame(const Frame &frame, ImageSize size)
{
    if (!frame.width || !frame.height || frame.rgba.size() != size_t(frame.width) * frame.height * 4)
        throw std::invalid_argument("Invalid RGBA frame dimensions or buffer size");
    if (!size.width || !size.height) throw std::invalid_argument("Resize target must be nonzero");
    Frame result;
    result.width = size.width; result.height = size.height;
    result.source_width = frame.source_width; result.source_height = frame.source_height;
    result.timestamp_ns = frame.timestamp_ns; result.generation = frame.generation;
    result.capture_id = frame.capture_id; result.submitted_ns = frame.submitted_ns;
    const auto columns = area_taps(frame.width, size.width), rows = area_taps(frame.height, size.height);
    // Vertical pass first: whole contiguous rows vectorize well and cut the row count early.
    const size_t source_row = size_t(frame.width) * 4;
    std::vector<float> vertical(source_row);
    result.rgba.resize(size_t(size.width) * size.height * 4);
    for (uint32_t y = 0; y < size.height; ++y) {
        std::fill(vertical.begin(), vertical.end(), 0.0f);
        for (const auto &tap : rows[y]) {
            const uint8_t *in = frame.rgba.data() + tap.index * source_row;
            for (size_t i = 0; i < source_row; ++i) vertical[i] += tap.weight * in[i];
        }
        uint8_t *out = result.rgba.data() + size_t(y) * size.width * 4;
        for (uint32_t x = 0; x < size.width; ++x) {
            float r = 0, g = 0, b = 0, a = 0;
            for (const auto &tap : columns[x]) {
                const float *px = vertical.data() + size_t(tap.index) * 4;
                r += tap.weight * px[0]; g += tap.weight * px[1]; b += tap.weight * px[2]; a += tap.weight * px[3];
            }
            out[x * 4 + 0] = uint8_t(std::clamp(r, 0.0f, 255.0f) + 0.5f);
            out[x * 4 + 1] = uint8_t(std::clamp(g, 0.0f, 255.0f) + 0.5f);
            out[x * 4 + 2] = uint8_t(std::clamp(b, 0.0f, 255.0f) + 0.5f);
            out[x * 4 + 3] = uint8_t(std::clamp(a, 0.0f, 255.0f) + 0.5f);
        }
    }
    return result;
}

std::vector<uint8_t> person_gate(const std::vector<uint8_t> &alpha, uint32_t width, uint32_t height, const GateShape &shape)
{
    if (!width || !height || alpha.size() != size_t(width) * height)
        throw std::invalid_argument("Invalid person gate dimensions");
    const float cut = std::clamp(shape.threshold, 0.05f, 0.95f);
    // Keep a soft transition around the selected threshold, with guaranteed
    // transparent and opaque endpoints even at the extremes of the slider.
    const float low = std::max(0.0f, cut - 0.35f), high = std::min(1.0f, cut + 0.35f);
    std::vector<uint8_t> gate(alpha.size());
    std::array<uint8_t, 256> coverage{};
    for (size_t i = 0; i < coverage.size(); ++i) {
        const float value = float(i) / 255;
        const float t = value < cut ? 0.5f * (value - low) / (cut - low) : 0.5f + 0.5f * (value - cut) / (high - cut);
        coverage[i] = uint8_t(std::lround(std::clamp(t, 0.0f, 1.0f) * 255));
    }
    std::transform(alpha.begin(), alpha.end(), gate.begin(), [&](uint8_t value) { return coverage[value]; });
    const auto pixels = [&](float percent) { return int(std::lround(std::clamp(percent, 0.0f, 25.0f) / 100 * height)); };
    const int close = pixels(shape.close);
    if (close > 0) {
        // Close gaps between nearby foreground areas without leaving a dilated
        // band around the silhouette where the main model can keep the room.
        extrema(gate, width, height, close, std::greater_equal<uint8_t>{});
        extrema(gate, width, height, close, std::less_equal<uint8_t>{});
    }
    const int feather = pixels(shape.feather);
    if (feather > 0) {
        auto softened = gate;
        separable(softened, width, height, feather, blur_line);
        std::transform(gate.begin(), gate.end(), softened.begin(), gate.begin(),
                       [](uint8_t original, uint8_t blurred) { return std::min(original, blurred); });
    }
    return gate;
}

ModelConfig gate_model_config(const ModelConfig &main)
{
    ModelConfig config;
    config.path = main.gate_path;
    config.device = main.device;
    config.threads = main.threads;
    config.rvm_max_size = kGateLongEdge;
    config.rvm_downsample = 0;
    config.rvm_foreground = false;
    return config;
}

void apply_person_gate(Model &gate, const Frame &frame, Mask &mask, const GateShape &shape)
{
    const auto started = monotonic_ns();
    const uint32_t source_width = frame.source_width ? frame.source_width : frame.width;
    const uint32_t source_height = frame.source_height ? frame.source_height : frame.height;
    auto small = resize_frame(frame, gate.input_size(source_width, source_height));
    small.source_width = source_width; small.source_height = source_height;
    const auto person = gate.run(small);
    mask.gate = person_gate(person.pixels, person.width, person.height, shape);
    mask.gate_width = person.width; mask.gate_height = person.height;
    mask.gate_ms = double(monotonic_ns() - started) / 1e6;
    mask.inference_ms += mask.gate_ms;
}
}
