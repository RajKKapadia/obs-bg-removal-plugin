#include "model.hpp"
#include "frame-sequence.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: rmbg-sequence MODEL.onnx FRAMES_DIR OUTPUT_DIR [--fps 30 --device cuda|cpu|auto --rvm-max-size 1920 --rvm-downsample 0 --rvm-foreground 0|1 --warmup 10]\n";
        return 2;
    }
    try {
        rmbg::ModelConfig config; config.path = argv[1]; config.rvm_max_size = 1920;
        unsigned fps = 30, warmup = 10;
        for (int i = 4; i < argc; i += 2) {
            if (i + 1 == argc) throw std::runtime_error("Missing argument value");
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--fps") fps = std::stoul(value);
            else if (key == "--warmup") warmup = std::stoul(value);
            else if (key == "--rvm-max-size") config.rvm_max_size = std::stoul(value);
            else if (key == "--rvm-downsample") config.rvm_downsample = std::stof(value);
            else if (key == "--rvm-foreground" && (value == "0" || value == "1")) config.rvm_foreground = value == "1";
            else if (key == "--device" && (value == "cpu" || value == "cuda" || value == "auto"))
                config.device = value == "cpu" ? rmbg::Device::CPU : value == "cuda" ? rmbg::Device::CUDA : rmbg::Device::Auto;
            else throw std::runtime_error("Invalid option: " + key);
        }
        if (!fps || fps > 60) throw std::runtime_error("FPS must be 1..60");
        const auto paths = image_io::sequence_paths(argv[2]);
        if (warmup >= paths.size()) throw std::runtime_error("Sequence must contain more frames than warmup");
        const std::filesystem::path output = argv[3];
        if (std::filesystem::exists(output) && !std::filesystem::is_empty(output))
            throw std::runtime_error("Use a new or empty output directory");
        for (const auto *dir : {"alpha", "cutout", "comparison"}) std::filesystem::create_directories(output / dir);
        std::ofstream csv(output / "timings.csv");
        csv << "frame,source_time_ms,processing_ms,recurrent_frames,input_width,input_height,downsample_ratio,warmup\n";
        rmbg::Model model(config);
        if (config.rvm_foreground && model.kind() != rmbg::ModelKind::RVM) throw std::runtime_error("Foreground colors require RVM");
        std::vector<double> measured;
        for (size_t index = 0; index < paths.size(); ++index) {
            auto source = image_io::read(paths[index].string());
            const auto size = model.input_size(source.width, source.height);
            auto input = image_io::resize(source, size.width, size.height);
            rmbg::Frame frame; frame.width = size.width; frame.height = size.height;
            frame.source_width = source.width; frame.source_height = source.height;
            frame.timestamp_ns = 1000000000ULL + index * 1000000000ULL / fps;
            frame.capture_id = index + 1; frame.generation = 1; frame.rgba = std::move(input.rgba);
            for (size_t i = 0; i < frame.rgba.size(); i += 4)
                for (size_t c = 0; c < 3; ++c) frame.rgba[i + c] = uint8_t((unsigned(frame.rgba[i + c]) * frame.rgba[i + 3] + 127) / 255);
            const auto mask = model.run(frame);
            if (index >= warmup) measured.push_back(mask.inference_ms);
            csv << index + 1 << ',' << index * 1000.0 / fps << ',' << mask.inference_ms << ',' << mask.recurrent_frames << ','
                << mask.width << ',' << mask.height << ',' << mask.downsample_ratio << ',' << (index < warmup) << '\n';
            auto cutout = source, alpha = source;
            for (uint32_t y = 0; y < source.height; ++y) for (uint32_t x = 0; x < source.width; ++x) {
                const float mx = (x + .5f) * mask.width / source.width - .5f, my = (y + .5f) * mask.height / source.height - .5f;
                const auto coverage = image_io::sample(mask.pixels.data(), mask.width, mask.height, 1, 0, mx, my);
                const size_t i = (size_t(y) * source.width + x) * 4;
                cutout.rgba[i + 3] = uint8_t((unsigned(coverage) * source.rgba[i + 3] + 127) / 255);
                for (size_t c = 0; c < 3; ++c) {
                    alpha.rgba[i + c] = coverage;
                    if (!mask.foreground_rgba.empty()) cutout.rgba[i + c] = image_io::sample(mask.foreground_rgba.data(), mask.width, mask.height, 4, c, mx, my);
                }
                alpha.rgba[i + 3] = 255;
            }
            std::ostringstream name; name << std::setw(6) << std::setfill('0') << index + 1 << ".png";
            image_io::write((output / "alpha" / name.str()).string(), alpha);
            image_io::write((output / "cutout" / name.str()).string(), cutout);
            // Three opaque full-resolution composites (black, white, blue), then
            // resize for a compact comparison strip without transparent-RGB bleed.
            const uint32_t pw = std::min(640u, source.width), ph = uint32_t(std::lround(double(source.height) * pw / source.width));
            image_io::Image strip{pw * 3, ph, std::vector<uint8_t>(size_t(pw) * 3 * ph * 4)};
            const std::array<std::array<int, 3>, 3> backgrounds{{{0, 0, 0}, {255, 255, 255}, {32, 100, 180}}};
            for (size_t panel = 0; panel < 3; ++panel) {
                auto composite = cutout;
                for (size_t i = 0; i < composite.rgba.size(); i += 4) {
                    const unsigned a = composite.rgba[i + 3];
                    for (size_t c = 0; c < 3; ++c) composite.rgba[i + c] = uint8_t((cutout.rgba[i + c] * a + backgrounds[panel][c] * (255 - a) + 127) / 255);
                    composite.rgba[i + 3] = 255;
                }
                const auto small = image_io::resize(composite, pw, ph);
                for (uint32_t y = 0; y < ph; ++y) std::copy_n(small.rgba.data() + size_t(y) * pw * 4, size_t(pw) * 4,
                    strip.rgba.data() + (size_t(y) * strip.width + panel * pw) * 4);
            }
            image_io::write((output / "comparison" / name.str()).string(), strip);
            if ((index + 1) % 30 == 0) std::cout << "Processed " << index + 1 << '/' << paths.size() << '\n' << std::flush;
        }
        const double mean = std::accumulate(measured.begin(), measured.end(), 0.0) / measured.size();
        std::sort(measured.begin(), measured.end());
        const double p95 = measured[size_t(std::ceil(.95 * measured.size())) - 1];
        std::ofstream report(output / "summary.txt");
        report << "Model file: " << config.path << "\nBackend: " << model.backend() << "\nPrecision: " << model.precision()
               << "\nSource FPS: " << fps << "\nFrames: " << paths.size() << "\nWarmup frames excluded: " << warmup
               << "\nForeground colors: " << config.rvm_foreground << "\nProcessing mean ms: " << mean << "\nProcessing p95 ms: " << p95
               << "\nOffline sequential inference only; excludes PNG I/O, resizing, OBS, camera, display, and encoding latency.\n";
        if (!model.warning().empty()) report << "Warning: " << model.warning() << '\n';
        std::cout << "Mean/p95 processing: " << mean << '/' << p95 << " ms; results: " << output << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
