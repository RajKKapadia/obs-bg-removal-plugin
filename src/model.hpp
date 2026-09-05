#pragma once
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <string>
#include <vector>

namespace rmbg {
enum class Device { Auto, CPU, CUDA };
struct ModelConfig {
    std::string path;
    Device device = Device::Auto;
    int threads = 2;
    bool operator==(const ModelConfig &other) const {
        return path == other.path && device == other.device && threads == other.threads;
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
};
void prepare_rgb(const Frame &frame, std::vector<float> &tensor);
std::vector<uint8_t> normalize_mask(const float *data, size_t size);
class Model {
public:
    explicit Model(const ModelConfig &config);
    Mask run(const Frame &frame);
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::string &backend() const { return backend_; }
    const std::string &warning() const { return warning_; }
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "obs-rmbg"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::string input_name_, output_name_, backend_, warning_;
    uint32_t width_ = 0, height_ = 0;
    std::vector<float> input_;
};
uint64_t monotonic_ns();
}
