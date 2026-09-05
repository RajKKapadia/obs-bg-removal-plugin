#include "model.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace rmbg {
uint64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void prepare_rgb(const Frame &frame, std::vector<float> &tensor)
{
    const size_t n = size_t(frame.width) * frame.height;
    if (!n || frame.rgba.size() != n * 4)
        throw std::invalid_argument("Invalid RGBA frame dimensions or buffer size");
    tensor.resize(n * 3);
    for (size_t i = 0; i < n; ++i) {
        // OBS provides premultiplied RGBA. Recover RGB before model normalization.
        const float alpha = frame.rgba[i * 4 + 3] / 255.0f;
        for (size_t c = 0; c < 3; ++c) {
            const float rgb = alpha > 0 ? std::min(1.0f, frame.rgba[i * 4 + c] / (255.0f * alpha)) : 0;
            tensor[c * n + i] = rgb - 0.5f;
        }
    }
}

std::vector<uint8_t> normalize_mask(const float *data, size_t size)
{
    if (!data || !size)
        throw std::invalid_argument("Empty model mask");
    float low = data[0], high = data[0];
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i]))
            throw std::runtime_error("Model produced a non-finite mask");
        low = std::min(low, data[i]);
        high = std::max(high, data[i]);
    }
    const float range = high - low;
    std::vector<uint8_t> pixels(size);
    for (size_t i = 0; i < size; ++i) {
        // Match BRIA's reference min/max normalization, guarding uniform masks.
        const float value = range > 1e-6f ? (data[i] - low) / range : std::clamp(data[i], 0.0f, 1.0f);
        pixels[i] = static_cast<uint8_t>(std::lround(value * 255));
    }
    return pixels;
}

Model::Model(const ModelConfig &config)
{
    env_.DisableTelemetryEvents();
    if (config.path.empty() || !std::filesystem::is_regular_file(config.path))
        throw std::runtime_error("Choose an existing RMBG-1.4 FP32 ONNX model file");
    auto options = [&] {
        Ort::SessionOptions o;
        o.SetIntraOpNumThreads(std::clamp(config.threads, 1, 32));
        o.SetInterOpNumThreads(1);
        o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return o;
    };
    if (config.device != Device::CPU) {
        try {
            const auto providers = Ort::GetAvailableProviders();
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
                throw std::runtime_error("This ONNX Runtime build does not include CUDA");
            auto o = options();
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            cuda.gpu_mem_limit = size_t(2) * 1024 * 1024 * 1024;
            cuda.do_copy_in_default_stream = 1;
            o.AppendExecutionProvider_CUDA(cuda);
            session_ = Ort::Session(env_, std::filesystem::path(config.path).c_str(), o);
            backend_ = "CUDA";
        } catch (const std::exception &e) {
            if (config.device == Device::CUDA)
                throw std::runtime_error(std::string("CUDA initialization failed: ") + e.what());
            warning_ = std::string("CUDA unavailable; using CPU. ") + e.what();
        }
    }
    if (!session_) {
        auto o = options();
        session_ = Ort::Session(env_, std::filesystem::path(config.path).c_str(), o);
        backend_ = "CPU";
    }
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() < 1)
        throw std::runtime_error("Expected RMBG-1.4 with one RGB input and at least one mask output");
    auto input_info = session_.GetInputTypeInfo(0);
    auto info = input_info.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("This version requires FP32 input; select model.onnx (not model_fp16.onnx)");
    if (shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) || shape[1] != 3 ||
        (shape[2] != 1024 && shape[2] != -1) || (shape[3] != 1024 && shape[3] != -1)) {
        std::string description;
        for (const auto dim : shape) description += std::to_string(dim) + " ";
        throw std::runtime_error("Expected RMBG-1.4 NCHW RGB input compatible with [1,3,1024,1024]; got " + description);
    }
    width_ = height_ = 1024;
    auto output_info = session_.GetOutputTypeInfo(0);
    if (output_info.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Expected FP32 output mask");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}

Mask Model::run(const Frame &frame)
{
    if (frame.width != width_ || frame.height != height_)
        throw std::invalid_argument("Frame must be resized to the model's input dimensions");
    const auto started = monotonic_ns();
    prepare_rgb(frame, input_);
    std::array<int64_t, 4> shape{1, 3, height_, width_};
    auto tensor = Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), shape.data(), shape.size());
    const char *inputs[] = {input_name_.c_str()};
    const char *outputs[] = {output_name_.c_str()};
    auto result = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    auto result_info = result[0].GetTensorTypeAndShapeInfo();
    const auto output_shape = result_info.GetShape();
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 1 ||
        output_shape[2] != height_ || output_shape[3] != width_)
        throw std::runtime_error("Unexpected model mask dimensions");
    Mask mask;
    mask.width = width_; mask.height = height_;
    mask.source_width = frame.source_width; mask.source_height = frame.source_height;
    mask.timestamp_ns = frame.timestamp_ns; mask.generation = frame.generation;
    mask.pixels = normalize_mask(result[0].GetTensorData<float>(), size_t(width_) * height_);
    mask.inference_ms = (monotonic_ns() - started) / 1e6;
    return mask;
}
}
