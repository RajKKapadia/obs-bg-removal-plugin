#pragma once
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rmbg {
enum class Device { Auto, CPU, CUDA };
enum class ModelKind { RMBG, RVM };
struct ImageSize { uint32_t width, height; };
ImageSize capture_size(ModelKind kind, uint32_t source_width, uint32_t source_height, uint32_t max_size = 1280);
struct ModelConfig {
    std::string path;
    Device device = Device::Auto;
    int threads = 2;
    uint32_t rvm_max_size = 1280;
    float rvm_downsample = 0; // 0 chooses an internal long edge of approximately 480 pixels.
    bool operator==(const ModelConfig &other) const {
        return path == other.path && device == other.device && threads == other.threads &&
               rvm_max_size == other.rvm_max_size && rvm_downsample == other.rvm_downsample;
    }
};
struct Frame {
    std::vector<uint8_t> rgba;
    uint32_t width = 0, height = 0;
    uint32_t source_width = 0, source_height = 0;
    uint64_t timestamp_ns = 0, generation = 0;
};
struct Mask {
    std::vector<uint8_t> pixels;
    uint32_t width = 0, height = 0;
    uint32_t source_width = 0, source_height = 0;
    uint64_t timestamp_ns = 0, generation = 0;
    double inference_ms = 0;
    bool direct_alpha = false;
    uint64_t recurrent_frames = 0;
    bool recurrent_on_gpu = false;
};
bool continues_sequence(const Frame &previous, const Frame &current);
void prepare_rgb(const Frame &frame, std::vector<float> &tensor, ModelKind kind = ModelKind::RMBG);
std::vector<uint8_t> normalize_mask(const float *data, size_t size);
std::vector<uint8_t> alpha_mask(const float *data, size_t size);
class Model {
public:
    explicit Model(const ModelConfig &config);
    Mask run(const Frame &frame);
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string &backend() const { return backend_; }
    const std::string &warning() const { return warning_; }
    const std::string &label() const { return label_; }
    ModelKind kind() const { return kind_; }
    uint32_t capture_limit() const { return config_.rvm_max_size; }
    ImageSize input_size(uint32_t width, uint32_t height) const { return capture_size(kind_, width, height, capture_limit()); }
private:
    Ort::Value input_tensor(const Frame &frame);
    Ort::Value run_rvm(const Frame &frame, const Ort::Value &input);
    void reset_recurrence();
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "obs-rmbg"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string input_name_, output_name_, backend_, warning_, label_;
    ModelConfig config_;
    ModelKind kind_ = ModelKind::RMBG;
    ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    uint32_t width_ = 0, height_ = 0;
    std::vector<float> input_;
    std::vector<Ort::Float16_t> input_half_;
    std::array<Ort::Value, 4> recurrent_{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
    std::array<float, 4> initial_float_{};
    std::array<Ort::Float16_t, 4> initial_half_{};
    Frame previous_frame_; // Only metadata is retained; rgba stays empty.
    uint64_t recurrent_frames_ = 0;
};
uint64_t monotonic_ns();
}
