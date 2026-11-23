#include "worker.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
template<class Predicate> void await(Predicate predicate)
{
    const auto deadline = rmbg::monotonic_ns() + 5000000000ULL;
    while (!predicate()) {
        require(rmbg::monotonic_ns() < deadline, "V-RMBG worker timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) { std::cerr << "Usage: vrmbg3-integration FIXTURES_DIR [cpu|cuda]\n"; return 2; }
    try {
        const bool cuda = argc == 3 && std::string(argv[2]) == "cuda";
        if (argc == 3 && !cuda && std::string(argv[2]) != "cpu") throw std::runtime_error("Invalid device");
        rmbg::ModelConfig config; config.device = cuda ? rmbg::Device::CUDA : rmbg::Device::CPU;
        rmbg::Frame frame;
        frame.width = frame.height = 1024; frame.source_width = 1920; frame.source_height = 1080;
        frame.timestamp_ns = 1000000000ULL; frame.generation = 2; frame.capture_id = 37;
        frame.rgba.resize(1024 * 1024 * 4);
        for (size_t i = 0; i < frame.rgba.size(); i += 4) {
            frame.rgba[i] = (i / 4 % 1024 < 512) ? 128 : 0;
            frame.rgba[i + 1] = 32; frame.rgba[i + 2] = 64; frame.rgba[i + 3] = 128;
        }
        for (const auto *name : {"fp32", "fp16"}) {
            config.path = (std::filesystem::path(argv[1]) / (std::string(name) + ".onnx")).string();
            config.rvm_foreground = true; // A saved RVM-only setting must be inert.
            rmbg::Model model(config);
            require(model.kind() == rmbg::ModelKind::VRMBG3 && model.width() == 1024, "Recognize the versioned V-RMBG export");
            const auto size = model.input_size(1920, 1080);
            require(size.width == 1024 && size.height == 1024, "Use fixed inference dimensions, independent of RVM size settings");
            auto f = frame;
            auto previous = model.run(f);
            require(previous.direct_alpha && previous.recurrent_frames == 1 && previous.foreground_rgba.empty(),
                    "Start with zero feedback, preserve alpha, and ignore RVM foreground colors");
            require(previous.capture_id == 37 && previous.generation == 2 && previous.source_width == 1920,
                    "Preserve capture identity and full source dimensions");
            require(previous.recurrent_on_gpu == cuda, "Keep feedback on the selected device");
            const auto first_pixels = previous.pixels;
            const bool half = std::string(name) == "fp16";
            float feedback_left = 0, feedback_right = 0;
            auto quantize = [&](float value) { return half ? Ort::Float16_t(value).ToFloat() : value; };
            for (unsigned index = 0; index < 20; ++index) {
                if (index) {
                    f.timestamp_ns += 33333333;
                    previous = model.run(f);
                }
                const float offset = quantize(0.0007f);
                auto expected = [&](float rgb, float &feedback) {
                    const float normalized = quantize((rgb - 0.485f) / 0.229f);
                    const float logits = quantize(quantize(quantize(normalized + feedback) * 0.125f) + offset);
                    const float alpha = quantize(1.0f / (1.0f + std::exp(-logits)));
                    feedback = quantize(normalized * alpha);
                    return int(std::lround(alpha * 255));
                };
                require(std::abs(int(previous.pixels.front()) - expected(1, feedback_left)) <= 1 &&
                        std::abs(int(previous.pixels.back()) - expected(0, feedback_right)) <= 1,
                        "Match normalized RGB and full-precision raw-alpha temporal feedback");
                require(previous.recurrent_frames == index + 1, "Count processed frames, including skipped source timestamps");
            }
            require(previous.pixels != first_pixels, "The graph must observe temporal feedback");
            // Source dimensions can change even though capture stays 1024 square.
            ++f.source_width; ++f.timestamp_ns;
            auto reset = model.run(f);
            require(reset.recurrent_frames == 1 && reset.pixels == first_pixels, "Reset on source resize");
            reset = model.run(f);
            require(reset.recurrent_frames == 1 && reset.pixels == first_pixels, "Reset on duplicate timestamps");
            --f.timestamp_ns; reset = model.run(f);
            require(reset.recurrent_frames == 1 && reset.pixels == first_pixels, "Reset on timestamp reversal");
            f.timestamp_ns += 1000000000ULL; reset = model.run(f);
            require(reset.recurrent_frames == 1 && reset.pixels == first_pixels, "Reset on a one-second pause");
            ++f.timestamp_ns; ++f.generation; reset = model.run(f);
            require(reset.recurrent_frames == 1 && reset.pixels == first_pixels, "Reset on a new generation");
            std::cout << name << ": temporal recurrence, resets, direct alpha and capture identity passed on " << model.backend() << '\n';
        }
        for (const auto *name : {"nan", "missing-metadata", "wrong-contract", "wrong-size", "wrong-type"}) {
            config.path = (std::filesystem::path(argv[1]) / (std::string(name) + ".onnx")).string();
            bool rejected = false;
            try { rmbg::Model model(config); model.run(frame); }
            catch (const std::exception &) { rejected = true; }
            require(rejected, "Reject invalid graph contract or non-finite alpha");
        }
        // Worker smoothing must affect display bytes only, not the feedback graph.
        config.path = (std::filesystem::path(argv[1]) / "fp32.onnx").string();
        rmbg::Worker worker;
        worker.configure(config); worker.set_smoothing(0.8f);
        await([&] { return worker.status().ready || worker.status().message.rfind("Error:", 0) == 0; });
        require(worker.status().ready, "Load fixture through the real worker");
        rmbg::Model reference(config);
        auto f = frame; f.generation = worker.status().generation;
        uint64_t completed = worker.status().completed;
        std::vector<uint8_t> displayed;
        for (unsigned index = 0; index < 4; ++index) {
            f.timestamp_ns += 33333333;
            const auto raw = reference.run(f);
            require(worker.submit(f), "Submit fixture to worker");
            await([&] { return worker.status().completed > completed || !worker.status().ready; });
            require(worker.status().ready, "Worker stays healthy");
            completed = worker.status().completed;
            require(worker.latest()->recurrent_frames == raw.recurrent_frames, "Worker keeps temporal state active");
            auto expected = raw.pixels;
            if (index) {
                const float weight = std::pow(0.8f, float((33333333.0 / 1e9) * 30));
                for (size_t pixel = 0; pixel < expected.size(); ++pixel)
                    expected[pixel] = uint8_t(std::lround(displayed[pixel] * weight + expected[pixel] * (1 - weight)));
            }
            require(worker.latest()->pixels == expected, "Display smoothing must use raw model recurrence independently of smoothed bytes");
            displayed = std::move(expected);
            if (index == 3) {
                f.timestamp_ns += 1000000000ULL;
                const auto restarted = reference.run(f);
                f.generation = worker.status().generation;
                require(worker.submit(f), "Submit reset frame");
                await([&] { return worker.status().completed > completed; });
                require(worker.latest()->recurrent_frames == 1 && worker.latest()->pixels == restarted.pixels,
                        "Worker must bypass smoothing after a temporal reset");
            }
        }
        // Matching captures bypass extra smoothing while preserving recurrence.
        f.timestamp_ns += 33333333;
        f.capture_token = std::make_shared<rmbg::CaptureToken>();
        const auto matched = reference.run(f);
        completed = worker.status().completed;
        require(worker.submit(f), "Submit matched frame");
        await([&] { return worker.status().completed > completed || !worker.status().ready; });
        require(worker.status().ready && worker.latest()->pixels == matched.pixels,
                "Matched frame must retain direct alpha without extra smoothing");
        std::cout << "Invalid exports rejected; real worker reset/smoothing checks passed\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
