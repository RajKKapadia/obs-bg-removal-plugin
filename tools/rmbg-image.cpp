#include "model.hpp"
#include "png-image.hpp"
#include <filesystem>
#include <iostream>
#include <numeric>

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: rmbg-image MODEL.onnx INPUT.png OUTPUT.png [--device cpu|cuda|auto] [--iterations N] [--threads N] [--rvm-max-size N] [--rvm-downsample RATIO]\n";
        return 2;
    }
    try {
        rmbg::ModelConfig config; config.path = argv[1]; int iterations = 3;
        for (int i = 4; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Missing option value");
            const std::string option = argv[i], value = argv[i + 1];
            if (option == "--device") {
                if (value != "cpu" && value != "cuda" && value != "auto") throw std::runtime_error("Invalid device");
                config.device = value == "cpu" ? rmbg::Device::CPU : value == "cuda" ? rmbg::Device::CUDA : rmbg::Device::Auto;
            } else if (option == "--iterations") iterations = std::stoi(value);
            else if (option == "--threads") config.threads = std::stoi(value);
            else if (option == "--rvm-max-size") config.rvm_max_size = std::stoul(value);
            else if (option == "--rvm-downsample") config.rvm_downsample = std::stof(value);
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (iterations < 1 || iterations > 1000) throw std::runtime_error("Iterations must be 1..1000");
        const auto source = image_io::read(argv[2]);
        rmbg::Model model(config);
        const auto size = model.input_size(source.width, source.height);
        auto resized = image_io::resize(source, size.width, size.height);
        rmbg::Frame frame; frame.width = resized.width; frame.height = resized.height;
        frame.source_width = source.width; frame.source_height = source.height;
        frame.rgba = std::move(resized.rgba);
        for (size_t i = 0; i < frame.rgba.size(); i += 4)
            for (size_t c = 0; c < 3; ++c) frame.rgba[i + c] = uint8_t((unsigned(frame.rgba[i + c]) * frame.rgba[i + 3] + 127) / 255);
        rmbg::Mask mask;
        std::vector<double> times;
        for (int i = 0; i < iterations; ++i) {
            frame.timestamp_ns = rmbg::monotonic_ns(); mask = model.run(frame); times.push_back(mask.inference_ms);
            std::cout << "Iteration " << i + 1 << ": " << mask.inference_ms << " ms\n" << std::flush;
        }
        auto output = source, preview = source;
        for (uint32_t y = 0; y < source.height; ++y) {
            for (uint32_t x = 0; x < source.width; ++x) {
                const auto alpha = image_io::sample(mask.pixels.data(), mask.width, mask.height, 1, 0,
                    (x + 0.5f) * mask.width / source.width - 0.5f, (y + 0.5f) * mask.height / source.height - 0.5f);
                const size_t i = (size_t(y) * source.width + x) * 4;
                output.rgba[i + 3] = uint8_t((unsigned(alpha) * source.rgba[i + 3] + 127) / 255);
                preview.rgba[i] = preview.rgba[i + 1] = preview.rgba[i + 2] = alpha; preview.rgba[i + 3] = 255;
            }
        }
        image_io::write(argv[3], output);
        const std::filesystem::path path(argv[3]);
        image_io::write((path.parent_path() / (path.stem().string() + "-mask.png")).string(), preview);
        const size_t start = times.size() > 1 ? 1 : 0;
        const double mean = std::accumulate(times.begin() + start, times.end(), 0.0) / (times.size() - start);
        std::cout << "Backend: " << model.backend() << "\nMean processing (warmup excluded when available): " << mean
                  << " ms; model pipeline ceiling: " << 1000 / mean << " FPS (not OBS output FPS)\n";
        std::cout << "Model: " << model.label() << "; input: " << size.width << 'x' << size.height
                  << "; recurrent frames: " << mask.recurrent_frames << "; recurrent state on GPU: " << mask.recurrent_on_gpu << '\n';
        if (!model.warning().empty()) std::cerr << "Warning: " << model.warning() << '\n';
    } catch (const std::exception &e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
