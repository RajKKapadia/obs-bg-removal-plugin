#include "model.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <stdexcept>

namespace rmbg {
uint64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

ImageSize capture_size(ModelKind kind, uint32_t width, uint32_t height, uint32_t max_size)
{
    if (!width || !height) throw std::invalid_argument("Source dimensions must be nonzero");
    if (kind == ModelKind::RMBG) return {1024, 1024};
    const double scale = std::min(1.0, double(std::clamp(max_size, 320u, 1920u)) / std::max(width, height));
    return {std::max(1u, uint32_t(std::lround(width * scale))), std::max(1u, uint32_t(std::lround(height * scale)))};
}

bool continues_sequence(const Frame &previous, const Frame &current)
{
    return previous.width == current.width && previous.height == current.height &&
           previous.source_width == current.source_width && previous.source_height == current.source_height &&
           previous.generation == current.generation && current.timestamp_ns > previous.timestamp_ns &&
           current.timestamp_ns - previous.timestamp_ns < 1000000000ULL;
}

template<class T> void prepare_rgb_impl(const Frame &frame, std::vector<T> &tensor, ModelKind kind)
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
            tensor[c * n + i] = T(rgb - (kind == ModelKind::RMBG ? 0.5f : 0.0f));
        }
    }
}
void prepare_rgb(const Frame &frame, std::vector<float> &tensor, ModelKind kind) { prepare_rgb_impl(frame, tensor, kind); }
void prepare_rgb(const Frame &frame, std::vector<Ort::Float16_t> &tensor, ModelKind kind) { prepare_rgb_impl(frame, tensor, kind); }

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

template<class T> std::vector<uint8_t> alpha_mask_impl(const T *data, size_t size)
{
    if (!data || !size) throw std::invalid_argument("Empty alpha matte");
    std::vector<uint8_t> pixels(size);
    for (size_t i = 0; i < size; ++i) {
        const float value = float(data[i]);
        if (!std::isfinite(value)) throw std::runtime_error("Model produced a non-finite alpha matte");
        pixels[i] = uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
    }
    return pixels;
}
std::vector<uint8_t> alpha_mask(const float *data, size_t size) { return alpha_mask_impl(data, size); }
std::vector<uint8_t> alpha_mask(const Ort::Float16_t *data, size_t size) { return alpha_mask_impl(data, size); }

template<class T> std::vector<uint8_t> foreground_rgba_impl(const T *data, size_t pixels)
{
    if (!data || !pixels) throw std::invalid_argument("Empty foreground image");
    std::vector<uint8_t> rgba(pixels * 4, 255);
    for (size_t i = 0; i < pixels; ++i) for (size_t c = 0; c < 3; ++c) {
        const float value = float(data[c * pixels + i]);
        if (!std::isfinite(value)) throw std::runtime_error("Model produced non-finite foreground colors");
        rgba[i * 4 + c] = uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
    }
    return rgba;
}
std::vector<uint8_t> foreground_rgba(const float *data, size_t pixels) { return foreground_rgba_impl(data, pixels); }
std::vector<uint8_t> foreground_rgba(const Ort::Float16_t *data, size_t pixels) { return foreground_rgba_impl(data, pixels); }

namespace {
struct TensorSpec { ONNXTensorElementDataType type; std::vector<int64_t> shape; };
std::map<std::string, TensorSpec> tensor_specs(const Ort::Session &session, bool inputs)
{
    std::map<std::string, TensorSpec> specs;
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < count; ++i) {
        const auto name = inputs ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        const auto info = inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        if (info.GetONNXType() != ONNX_TYPE_TENSOR) throw std::runtime_error("Expected tensor inputs and outputs");
        const auto tensor = info.GetTensorTypeAndShapeInfo();
        specs.emplace(name.get(), TensorSpec{tensor.GetElementType(), tensor.GetShape()});
    }
    return specs;
}
bool floating(ONNXTensorElementDataType type)
{
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}
void require_image(const TensorSpec &tensor, int64_t channels, const char *name)
{
    if (!floating(tensor.type) || tensor.shape.size() != 4 ||
        (tensor.shape[0] != 1 && tensor.shape[0] != -1) || tensor.shape[1] != channels)
        throw std::runtime_error(std::string("Invalid ") + name + ": expected floating-point NCHW image tensor");
}
}

Model::Model(const ModelConfig &config) : config_(config)
{
    env_.DisableTelemetryEvents();
    if (config.path.empty() || !std::filesystem::is_regular_file(config.path))
        throw std::runtime_error("Choose an existing RMBG-1.4 or RVM ONNX model file");
    if (config.rvm_max_size < 320 || config.rvm_max_size > 1920 || !std::isfinite(config.rvm_downsample) ||
        config.rvm_downsample < 0 || config.rvm_downsample > 1 || (config.rvm_downsample > 0 && config.rvm_downsample < 0.1f))
        throw std::invalid_argument("RVM input limit must be 320..1920; downsample ratio must be 0 (auto) or 0.1..1");
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
    const auto inputs = tensor_specs(session_, true), outputs = tensor_specs(session_, false);
    if (inputs.size() == 6 && inputs.count("src") && inputs.count("downsample_ratio") && outputs.count("pha")) {
        kind_ = ModelKind::RVM;
        input_name_ = "src"; output_name_ = "pha";
        const auto &src = inputs.at("src");
        require_image(src, 3, "RVM src");
        require_image(outputs.at("pha"), 1, "RVM pha");
        if (config.rvm_foreground) {
            if (!outputs.count("fgr")) throw std::runtime_error("RVM model has no foreground color output");
            require_image(outputs.at("fgr"), 3, "RVM fgr");
            if (outputs.at("fgr").type != src.type) throw std::runtime_error("RVM foreground precision must match src");
        }
        if (src.shape[2] != -1 || src.shape[3] != -1 || outputs.at("pha").type != src.type)
            throw std::runtime_error("Use the official RVM ONNX export with dynamic spatial dimensions");
        const auto &ratio = inputs.at("downsample_ratio");
        if (ratio.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || ratio.shape != std::vector<int64_t>{1})
            throw std::runtime_error("RVM downsample_ratio must be a single FP32 value");
        for (int i = 1; i <= 4; ++i) {
            const auto in = "r" + std::to_string(i) + "i", out = "r" + std::to_string(i) + "o";
            if (!inputs.count(in) || !outputs.count(out) || inputs.at(in).type != src.type ||
                outputs.at(out).type != src.type || inputs.at(in).shape.size() != 4 || outputs.at(out).shape.size() != 4)
                throw std::runtime_error("RVM requires four matching recurrent state inputs and outputs");
        }
        label_ = "RVM";
    } else if (inputs.size() == 1 && !outputs.empty()) {
        input_name_ = inputs.begin()->first;
        Ort::AllocatorWithDefaultOptions allocator;
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
        const auto &src = inputs.begin()->second;
        require_image(src, 3, "RMBG input");
        // RMBG exports can leave output dimensions symbolic. Validate the
        // actual [1,1,H,W] shape after inference, as in the original adapter.
        if (!floating(outputs.at(output_name_).type)) throw std::runtime_error("Expected floating-point RMBG output");
        if ((src.shape[2] != 1024 && src.shape[2] != -1) || (src.shape[3] != 1024 && src.shape[3] != -1))
            throw std::runtime_error("RMBG-1.4 input must be compatible with [1,3,1024,1024]");
        width_ = height_ = 1024;
        label_ = "RMBG-1.4";
    } else throw std::runtime_error("Unsupported model signature; select RMBG-1.4 or RVM ONNX");
    input_type_ = inputs.at(input_name_).type;
    if (kind_ == ModelKind::RVM)
        label_ += input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? " FP16" : " FP32";
}

Ort::Value Model::input_tensor(const Frame &frame)
{
    const std::array<int64_t, 4> shape{1, 3, frame.height, frame.width};
    if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Write the final FP16 tensor directly, without a full FP32 intermediate.
        prepare_rgb(frame, input_half_, kind_);
        return Ort::Value::CreateTensor<Ort::Float16_t>(memory_, input_half_.data(), input_half_.size(), shape.data(), shape.size());
    }
    prepare_rgb(frame, input_, kind_);
    return Ort::Value::CreateTensor<float>(memory_, input_.data(), input_.size(), shape.data(), shape.size());
}

void Model::reset_recurrence()
{
    const std::array<int64_t, 4> shape{1, 1, 1, 1};
    for (size_t i = 0; i < recurrent_.size(); ++i) {
        if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            recurrent_[i] = Ort::Value::CreateTensor<Ort::Float16_t>(memory_, &initial_half_[i], 1, shape.data(), shape.size());
        else recurrent_[i] = Ort::Value::CreateTensor<float>(memory_, &initial_float_[i], 1, shape.data(), shape.size());
    }
    recurrent_frames_ = 0;
}

Ort::Value Model::run_rvm(const Frame &frame, const Ort::Value &input, Ort::Value &foreground)
{
    if (!recurrent_frames_ || !continues_sequence(previous_frame_, frame)) reset_recurrence();
    float ratio = config_.rvm_downsample > 0 ? config_.rvm_downsample : std::min(1.0f, 480.0f / std::max(frame.width, frame.height));
    const std::array<int64_t, 1> ratio_shape{1};
    auto ratio_tensor = Ort::Value::CreateTensor<float>(memory_, &ratio, 1, ratio_shape.data(), ratio_shape.size());
    Ort::IoBinding binding(session_);
    binding.BindInput("src", input);
    binding.BindInput("downsample_ratio", ratio_tensor);
    binding.BindOutput("pha", memory_);
    // Keep the recurrent state tensors on CUDA between frames. Only alpha is
    // copied back for the OBS texture upload; CPU inference uses CPU state.
    auto state_memory = backend_ == "CUDA" ? Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
                                            : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < recurrent_.size(); ++i) {
        binding.BindInput(("r" + std::to_string(i + 1) + "i").c_str(), recurrent_[i]);
        binding.BindOutput(("r" + std::to_string(i + 1) + "o").c_str(), state_memory);
    }
    if (config_.rvm_foreground) binding.BindOutput("fgr", memory_);
    binding.SynchronizeInputs();
    session_.Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    auto outputs = binding.GetOutputValues();
    if (outputs.size() != (config_.rvm_foreground ? 6u : 5u))
        throw std::runtime_error("RVM did not return the requested outputs");
    for (size_t i = 0; i < recurrent_.size(); ++i) recurrent_[i] = std::move(outputs[i + 1]);
    if (config_.rvm_foreground) foreground = std::move(outputs[5]);
    ++recurrent_frames_;
    previous_frame_.width = frame.width; previous_frame_.height = frame.height;
    previous_frame_.source_width = frame.source_width; previous_frame_.source_height = frame.source_height;
    previous_frame_.timestamp_ns = frame.timestamp_ns; previous_frame_.generation = frame.generation;
    return std::move(outputs[0]);
}

Mask Model::run(const Frame &frame)
{
    const auto expected = input_size(frame.source_width ? frame.source_width : frame.width,
                                     frame.source_height ? frame.source_height : frame.height);
    if (frame.width != expected.width || frame.height != expected.height ||
        (kind_ == ModelKind::RVM && std::min(frame.width, frame.height) < 16))
        throw std::invalid_argument("Frame dimensions do not match the model capture size (RVM requires at least 16 pixels per side)");
    const auto started = monotonic_ns();
    auto tensor = input_tensor(frame);
    Ort::Value result{nullptr}, foreground{nullptr};
    if (kind_ == ModelKind::RVM) result = run_rvm(frame, tensor, foreground);
    else {
        const char *inputs[] = {input_name_.c_str()}, *outputs[] = {output_name_.c_str()};
        auto values = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        result = std::move(values[0]);
    }
    const auto result_info = result.GetTensorTypeAndShapeInfo();
    const auto output_shape = result_info.GetShape();
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 1 ||
        output_shape[2] != frame.height || output_shape[3] != frame.width)
        throw std::runtime_error("Unexpected model mask dimensions");
    Mask mask;
    mask.width = frame.width; mask.height = frame.height;
    mask.source_width = frame.source_width; mask.source_height = frame.source_height;
    mask.timestamp_ns = frame.timestamp_ns; mask.generation = frame.generation;
    mask.capture_id = frame.capture_id;
    const size_t count = size_t(frame.width) * frame.height;
    mask.direct_alpha = kind_ == ModelKind::RVM;
    std::vector<float> expanded;
    const float *alpha;
    if (result_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto *half = result.GetTensorData<Ort::Float16_t>();
        if (mask.direct_alpha) {
            // RVM alpha needs one conversion pass, not an FP32 expansion followed by quantization.
            mask.pixels = alpha_mask(half, count);
            alpha = nullptr;
        } else {
            expanded.resize(count);
            std::transform(half, half + count, expanded.begin(), [](Ort::Float16_t value) { return value.ToFloat(); });
            alpha = expanded.data();
        }
    } else if (result_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) alpha = result.GetTensorData<float>();
    else throw std::runtime_error("Expected floating-point mask output");
    if (alpha) mask.pixels = mask.direct_alpha ? alpha_mask(alpha, count) : normalize_mask(alpha, count);
    if (foreground) {
        const auto info = foreground.GetTensorTypeAndShapeInfo();
        if (info.GetShape() != std::vector<int64_t>{1, 3, frame.height, frame.width} || info.GetElementType() != input_type_)
            throw std::runtime_error("Unexpected RVM foreground dimensions or precision");
        mask.foreground_rgba = input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            ? foreground_rgba(foreground.GetTensorData<Ort::Float16_t>(), count)
            : foreground_rgba(foreground.GetTensorData<float>(), count);
    }
    if (kind_ == ModelKind::RVM)
        mask.downsample_ratio = config_.rvm_downsample > 0 ? config_.rvm_downsample : std::min(1.0f, 480.0f / std::max(frame.width, frame.height));
    mask.recurrent_frames = recurrent_frames_;
    mask.recurrent_on_gpu = kind_ == ModelKind::RVM && recurrent_[0].GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
    mask.inference_ms = (monotonic_ns() - started) / 1e6;
    return mask;
}
}
