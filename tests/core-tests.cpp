#include "gate.hpp"
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
        rmbg::RollingMetric metric;
        metric.reset(1000000000ULL);
        for (uint64_t i = 1; i <= 100; ++i) metric.add(1000000000ULL + i * 10000000ULL, double(i));
        auto summary = metric.summary(2000000000ULL);
        require(summary.count == 100 && summary.mean == 50.5 && summary.p95 == 95 && summary.rate == 100,
                "Rolling diagnostics use elapsed window time and nearest-rank p95");
        require(metric.summary(8000000000ULL).count == 0, "Idle metrics expire without new samples");
        for (uint64_t i = 0; i < 5000; ++i) metric.add(9000000000ULL + i, 1);
        require(metric.stored() == rmbg::RollingMetric::capacity, "Statistics storage must stay bounded under excessive event rates");
        metric.add(1000000000ULL, 2);
        require(metric.stored() == 1 && metric.summary(1000000001ULL).mean == 2, "Reset on clock reversal");
        const float colors[] = {1, 0.5f, 0, 0.25f, -1, 2};
        require(rmbg::foreground_rgba(colors, 2) == std::vector<uint8_t>({255, 0, 0, 255, 128, 64, 255, 255}),
                "Foreground colors convert planar RGB to clamped straight RGBA");
        std::vector<Ort::Float16_t> half_colors;
        for (float value : colors) half_colors.emplace_back(value);
        require(rmbg::foreground_rgba(half_colors.data(), 2) == rmbg::foreground_rgba(colors, 2), "FP16 foreground conversion matches FP32");
        const float bad_colors[] = {0, 1, std::numeric_limits<float>::infinity()};
        bool bad_foreground = false;
        try { rmbg::foreground_rgba(bad_colors, 1); } catch (const std::runtime_error &) { bad_foreground = true; }
        require(bad_foreground, "Reject non-finite foreground output");
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
        rmbg::prepare_rgb(frame, tensor, rmbg::ModelKind::VRMBG3);
        require(std::abs(tensor[0] - (1.0f - 0.485f) / 0.229f) < 0.0001f &&
                std::abs(tensor[3] - (0.25f - 0.456f) / 0.224f) < 0.0001f,
                "V-RMBG uses reference channel normalization after recovering straight RGB");
        // Check fused conversion against the previous FP32-then-FP16 path,
        // including transparent and invalid premultiplied channel values.
        frame.width = 256; frame.height = 256; frame.rgba.resize(256 * 256 * 4);
        for (size_t i = 0; i < 256 * 256; ++i) {
            frame.rgba[i * 4] = uint8_t(i); frame.rgba[i * 4 + 1] = uint8_t(255 - i);
            frame.rgba[i * 4 + 2] = uint8_t(i / 7); frame.rgba[i * 4 + 3] = uint8_t(i / 256);
        }
        for (const auto kind : {rmbg::ModelKind::RMBG, rmbg::ModelKind::RVM, rmbg::ModelKind::VRMBG3}) {
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
        {
            // A narrow hole can be filled without growing the outside edge.
            std::vector<uint8_t> person(20 * 10, 0);
            for (int y = 3; y <= 7; ++y)
                for (int x = 5; x <= 15; ++x) person[y * 20 + x] = 255;
            person[5 * 20 + 10] = 0;
            rmbg::GateShape shape{0.5f, 10, 0}; // 10% of 10 rows = 1 px
            auto closed = rmbg::person_gate(person, 20, 10, shape);
            require(closed[5 * 20 + 10] == 255 && closed[5 * 20 + 4] == 0 && closed[2 * 20 + 10] == 0,
                    "Gate closes small gaps without adding an outer band");
            shape.close = 0;
            person[5 * 20 + 5] = 128;
            const auto soft_edge = rmbg::person_gate(person, 20, 10, shape);
            require(soft_edge[5 * 20 + 5] == 128 && soft_edge[5 * 20 + 10] == 0,
                    "Default midpoint preserves fractional alpha and zero radius leaves holes alone");
            shape.threshold = 0.9f;
            const auto tighter = rmbg::person_gate(person, 20, 10, shape);
            require(tighter[5 * 20 + 5] < soft_edge[5 * 20 + 5] && tighter[5 * 20 + 6] == 255,
                    "Higher midpoint tightens fractional coverage and keeps opaque pixels opaque");
            shape.threshold = 0.5f; shape.feather = 10;
            auto soft = rmbg::person_gate(person, 20, 10, shape);
            require(soft[3 * 20 + 7] > 0 && soft[3 * 20 + 7] < 255 && soft[2 * 20 + 7] == 0 && soft[5 * 20 + 7] == 255,
                    "Gate feathering softens inward without adding background coverage");
            std::vector<uint8_t> full(20 * 10, 255);
            require(rmbg::person_gate(full, 20, 10, shape) == full, "Feathering keeps a solid gate solid at the image edges");
            bool bad_gate = false;
            try { rmbg::person_gate(full, 20, 9, shape); } catch (const std::invalid_argument &) { bad_gate = true; }
            require(bad_gate, "Gate rejects mismatched dimensions");
            // Resampling keeps metadata, drops the capture token, and averages areas.
            rmbg::Frame square; square.width = square.height = 4; square.source_width = 16; square.source_height = 9;
            square.timestamp_ns = 42; square.generation = 7; square.capture_token = std::make_shared<rmbg::CaptureToken>();
            square.rgba.assign(4 * 4 * 4, 0);
            for (uint32_t y = 0; y < 4; ++y) for (uint32_t x = 2; x < 4; ++x) for (int c = 0; c < 4; ++c) square.rgba[(y * 4 + x) * 4 + c] = 255;
            const auto half = rmbg::resize_frame(square, {2, 2});
            require(half.width == 2 && half.height == 2 && half.source_width == 16 && half.timestamp_ns == 42 && half.generation == 7 && !half.capture_token,
                    "Resampled gate frames keep timing and source metadata without pinning video");
            require(half.rgba[0] == 0 && half.rgba[4] == 255 && half.rgba[8] == 0 && half.rgba[12] == 255, "Area resampling preserves hard columns");
            const auto wide = rmbg::resize_frame(square, {3, 1});
            require(wide.rgba[0] == 0 && wide.rgba[4 * 1] == 128 && wide.rgba[4 * 2] == 255, "Fractional area weights blend straddling pixels");
            const auto up = rmbg::resize_frame(square, {8, 8});
            require(up.width == 8 && up.rgba[0] == 0 && up.rgba[7 * 4] == 255, "Upsampling interpolates between source centers");
            rmbg::ModelConfig main; main.path = "rmbg.onnx"; main.gate_path = "rvm.onnx"; main.device = rmbg::Device::CUDA; main.threads = 6;
            main.rvm_max_size = 1920; main.rvm_downsample = 0.5f; main.rvm_foreground = true;
            const auto gate_config = rmbg::gate_model_config(main);
            require(gate_config.path == "rvm.onnx" && gate_config.device == rmbg::Device::CUDA && gate_config.threads == 6 &&
                    gate_config.rvm_max_size == rmbg::kGateLongEdge && gate_config.rvm_downsample == 0 && !gate_config.rvm_foreground &&
                    gate_config.gate_path.empty(), "Gate session inherits device and threads only");
            rmbg::ModelConfig other = main; other.gate_path.clear();
            require(!(other == main), "Changing the person gate model reloads the worker");
        }
        std::cout << "All core checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
