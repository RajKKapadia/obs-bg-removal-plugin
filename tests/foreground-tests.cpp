#include "model.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    if (argc != 2) { std::cerr << "Usage: rmbg-foreground-tests FIXTURES_DIR\n"; return 2; }
    try {
        rmbg::Frame frame;
        frame.width = frame.source_width = 64; frame.height = frame.source_height = 64;
        frame.timestamp_ns = 1000000000; frame.capture_id = 73; frame.generation = 2;
        frame.rgba.resize(64 * 64 * 4);
        for (size_t i = 0; i < 64 * 64; ++i) {
            frame.rgba[i * 4] = i < 2048 ? 128 : 0;
            frame.rgba[i * 4 + 2] = i < 2048 ? 0 : 128;
            frame.rgba[i * 4 + 3] = 128;
        }
        for (const auto *name : {"fp32", "fp16", "nan", "shape", "type"}) {
            rmbg::ModelConfig config;
            config.path = (std::filesystem::path(argv[1]) / (std::string(name) + ".onnx")).string();
            config.device = rmbg::Device::CPU; config.rvm_foreground = true;
            bool rejected = false;
            try {
                rmbg::Model model(config);
                const auto mask = model.run(frame);
                if (mask.capture_id != 73 || mask.generation != 2 || mask.pixels.front() != 255 || mask.pixels.back() != 0 ||
                    mask.foreground_rgba.size() != frame.rgba.size() || mask.foreground_rgba[0] != 0 || mask.foreground_rgba[1] != 255)
                    throw std::logic_error("Incorrect deterministic foreground/alpha output");
                config.rvm_foreground = false;
                rmbg::Model baseline(config);
                const auto original = baseline.run(frame);
                if (!original.foreground_rgba.empty() || original.pixels != mask.pixels) throw std::logic_error("Disabled foreground changes alpha");
            } catch (const std::logic_error &) { throw; }
              catch (const std::exception &) { rejected = true; }
            const bool invalid = std::string(name) != "fp32" && std::string(name) != "fp16";
            if (rejected != invalid) throw std::runtime_error(std::string("Wrong rejection behavior: ") + name);
            std::cout << name << ": " << (rejected ? "invalid output rejected" : "color, alpha, identity and disabled path passed") << '\n';
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
