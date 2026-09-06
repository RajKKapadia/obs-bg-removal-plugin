#include "model.hpp"
#include "../tools/png-image.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
double difference(const rmbg::Mask &a, const rmbg::Mask &b)
{
    require(a.width == b.width && a.height == b.height, "Compare masks with matching shapes");
    double sum = 0;
    for (size_t i = 0; i < a.pixels.size(); ++i) sum += std::abs(int(a.pixels[i]) - int(b.pixels[i]));
    return sum / a.pixels.size();
}
}
int main(int argc, char **argv)
{
    if (argc != 4) { std::cerr << "Usage: rvm-integration MODEL.onnx PORTRAIT.png cpu|cuda\n"; return 2; }
    try {
        const auto image = image_io::read(argv[2]);
        const std::string device = argv[3];
        require(device == "cpu" || device == "cuda", "Select cpu or cuda");
        rmbg::ModelConfig config{argv[1], device == "cpu" ? rmbg::Device::CPU : rmbg::Device::CUDA, 2};
        config.rvm_max_size = 640;
        rmbg::Model model(config);
        require(model.kind() == rmbg::ModelKind::RVM, "Detect the RVM signature");
        auto frame = [&](uint32_t width, uint32_t height, uint64_t timestamp, uint64_t generation = 1) {
            const auto size = model.input_size(width, height);
            auto resized = image_io::resize(image, size.width, size.height);
            rmbg::Frame f; f.width = size.width; f.height = size.height; f.source_width = width; f.source_height = height;
            f.timestamp_ns = timestamp; f.generation = generation; f.rgba = std::move(resized.rgba);
            for (size_t i = 0; i < f.rgba.size(); i += 4)
                for (size_t c = 0; c < 3; ++c) f.rgba[i + c] = uint8_t((unsigned(f.rgba[i + c]) * f.rgba[i + 3] + 127) / 255);
            return f;
        };
        const auto first = model.run(frame(1280, 720, 1000000000));
        const auto second = model.run(frame(1280, 720, 1033333333));
        require(first.direct_alpha && first.recurrent_frames == 1 && second.recurrent_frames == 2, "Recycle temporal state across frames");
        require(second.recurrent_on_gpu == (device == "cuda"), "Keep state on the selected device");
        const auto temporal_change = difference(first, second);
        require(temporal_change > 0, "Recycled state must affect the portrait prediction");
        const auto reset = model.run(frame(1280, 720, 3000000000));
        require(reset.recurrent_frames == 1 && difference(first, reset) < 0.05, "A pause must reproduce the initial-state prediction");
        model.run(frame(1280, 720, 3033333333));
        const auto source_resize = model.run(frame(1920, 1080, 3066666666));
        require(source_resize.recurrent_frames == 1 && difference(first, source_resize) < 0.05,
                "Reset for source changes even when capture dimensions stay 640x360");
        const auto portrait = model.run(frame(720, 1280, 3100000000));
        require(portrait.width == 360 && portrait.height == 640 && portrait.recurrent_frames == 1,
                "Reallocate recurrent tensors for portrait input");
        const auto generation = model.run(frame(720, 1280, 3133333333, 2));
        require(generation.recurrent_frames == 1 && difference(portrait, generation) < 0.05, "New generation starts with zero state");
        model.run(frame(720, 1280, 3166666666, 2));
        const auto rewind = model.run(frame(720, 1280, 3150000000, 2));
        require(rewind.recurrent_frames == 1 && difference(portrait, rewind) < 0.05, "A timestamp discontinuity resets state");
        std::cout << model.label() << ": temporal state, GPU residency, resize, generation and pause checks passed; temporal alpha change="
                  << temporal_change << "/255\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
