#include "worker.hpp"
#include "capture-schedule.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
int main()
{
    try {
        // Non-divisor limits must not collapse to the next slower video cadence.
        for (const unsigned fps : {15u, 20u, 30u, 60u}) {
            rmbg::CaptureSchedule schedule;
            unsigned captures = 0;
            for (uint64_t tick = 0; tick < 300; ++tick)
                captures += schedule.due(1000000000ULL + tick * 1000000000ULL / 30, fps);
            require(captures == std::min(fps, 30u) * 10, "Respect mask rate at 30 video FPS without phase drift");
        }
        rmbg::CaptureSchedule schedule;
        require(schedule.due(1000000000ULL, 1) && schedule.due(1033333333ULL, 30), "Rate increases apply on the next video tick");
        require(schedule.due(10000000000ULL, 30) && !schedule.due(10000000000ULL, 30), "Resume without a catch-up burst");
        require(schedule.due(1000000000ULL, 30), "Reset scheduling if the video clock moves backwards");
        rmbg::Frame frame; frame.width = 2; frame.height = 1;
        frame.rgba = {255, 0, 128, 255, 64, 32, 0, 128};
        std::vector<float> tensor;
        rmbg::prepare_rgb(frame, tensor);
        require(tensor.size() == 6 && tensor[0] == 0.5f && tensor[2] == -0.5f, "RGB must be planar, scaled by 255, centered by 0.5");
        require(std::abs(tensor[1]) < 0.0001 && std::abs(tensor[3] + 0.25f) < 0.0001, "Premultiplied RGB must be recovered");
        rmbg::prepare_rgb(frame, tensor, rmbg::ModelKind::RVM);
        require(tensor[0] == 1.0f && tensor[2] == 0.0f && std::abs(tensor[1] - 0.5f) < 0.0001,
                "RVM needs uncentered RGB in 0..1 with premultiplied RGB recovered");
        // Check fused conversion against the previous FP32-then-FP16 path,
        // including transparent and invalid premultiplied channel values.
        frame.width = 256; frame.height = 256; frame.rgba.resize(256 * 256 * 4);
        for (size_t i = 0; i < 256 * 256; ++i) {
            frame.rgba[i * 4] = uint8_t(i); frame.rgba[i * 4 + 1] = uint8_t(255 - i);
            frame.rgba[i * 4 + 2] = uint8_t(i / 7); frame.rgba[i * 4 + 3] = uint8_t(i / 256);
        }
        for (const auto kind : {rmbg::ModelKind::RMBG, rmbg::ModelKind::RVM}) {
            rmbg::prepare_rgb(frame, tensor, kind);
            std::vector<Ort::Float16_t> half;
            rmbg::prepare_rgb(frame, half, kind);
            for (size_t i = 0; i < tensor.size(); ++i)
                require(half[i].val == Ort::Float16_t(tensor[i]).val, "Fused FP16 input must be bit-identical");
        }
        std::vector<Ort::Float16_t> all_half;
        std::vector<float> all_float;
        for (unsigned bits = 0; bits < 65536; ++bits) {
            Ort::Float16_t half; half.val = uint16_t(bits);
            if (std::isfinite(half.ToFloat())) { all_half.push_back(half); all_float.push_back(half.ToFloat()); }
        }
        require(rmbg::alpha_mask(all_half.data(), all_half.size()) == rmbg::alpha_mask(all_float.data(), all_float.size()),
                "Fused alpha conversion must preserve every finite FP16 value exactly");
        frame.rgba.pop_back();
        bool rejected = false;
        try { rmbg::prepare_rgb(frame, tensor); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Reject truncated input before accessing pixels");
        const float values[] = {-2, 0, 2};
        const auto mask = rmbg::normalize_mask(values, 3);
        require(mask[0] == 0 && mask[1] == 128 && mask[2] == 255, "Match reference mask normalization");
        const float ones[] = {1, 1}, zeros[] = {0, 0};
        require(rmbg::normalize_mask(ones, 2)[0] == 255 && rmbg::normalize_mask(zeros, 2)[0] == 0, "Uniform masks must remain finite and meaningful");
        const float nan[] = {0, std::numeric_limits<float>::quiet_NaN()};
        rejected = false;
        try { rmbg::normalize_mask(nan, 2); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Reject non-finite model output");
        const float alpha[] = {-0.1f, 0.2f, 0.4f, 0.6f, 1.1f};
        const auto matte = rmbg::alpha_mask(alpha, 5);
        require(matte == std::vector<uint8_t>({0, 51, 102, 153, 255}), "Alpha must be clamped without stretching its range");
        rejected = false;
        try { rmbg::alpha_mask(nan, 2); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "Reject non-finite alpha before conversion to bytes");
        const auto landscape = rmbg::capture_size(rmbg::ModelKind::RVM, 1920, 1080);
        const auto portrait = rmbg::capture_size(rmbg::ModelKind::RVM, 1080, 1920);
        const auto small = rmbg::capture_size(rmbg::ModelKind::RVM, 640, 480);
        const auto square = rmbg::capture_size(rmbg::ModelKind::RMBG, 1920, 1080);
        require(landscape.width == 1280 && landscape.height == 720 && portrait.width == 720 && portrait.height == 1280,
                "RVM capture must preserve landscape and portrait aspect ratios");
        require(small.width == 640 && small.height == 480 && square.width == 1024 && square.height == 1024,
                "Do not upscale small RVM inputs or change RMBG capture dimensions");
        rmbg::Frame previous; previous.width = 640; previous.height = 360;
        previous.source_width = 1280; previous.source_height = 720; previous.generation = 1; previous.timestamp_ns = 1000000000;
        auto current = previous; current.timestamp_ns += 33333333;
        require(rmbg::continues_sequence(previous, current), "Adjacent video frames must reuse temporal memory");
        current.timestamp_ns = previous.timestamp_ns;
        require(!rmbg::continues_sequence(previous, current), "Duplicate timestamps must reset temporal memory");
        current.timestamp_ns += 1000000000;
        require(!rmbg::continues_sequence(previous, current), "A long pause must reset temporal memory");
        current.timestamp_ns = previous.timestamp_ns + 1; ++current.source_width;
        require(!rmbg::continues_sequence(previous, current), "Source resizing must reset state even if capture size is unchanged");
        current.source_width = previous.source_width; ++current.generation;
        require(!rmbg::continues_sequence(previous, current), "A new configuration generation must reset state");
        current.generation = previous.generation; current.width = 360; current.height = 640;
        require(!rmbg::continues_sequence(previous, current), "Equal pixel count must not hide orientation changes");
        rmbg::Worker worker;
        worker.configure({"/nonexistent/rmbg.onnx", rmbg::Device::CPU, 2});
        const auto deadline = rmbg::monotonic_ns() + 2000000000ULL;
        while (worker.status().message.rfind("Error:", 0) != 0 && rmbg::monotonic_ns() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(!worker.status().ready && !worker.latest() && worker.status().message.rfind("Error:", 0) == 0,
                "Missing model must yield an observable error and no mask");
        require(!worker.submit({}), "Unready worker must reject frames");
        const auto generation = worker.status().generation;
        worker.configure({"/nonexistent/other.onnx", rmbg::Device::CPU, 2});
        require(worker.status().generation == generation + 1 && !worker.latest(), "Changing models invalidates old results");
        std::cout << "All core checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
