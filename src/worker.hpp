#pragma once
#include "model.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace rmbg {
struct WorkerStatus {
    bool ready = false, busy = false;
    uint32_t width = 0, height = 0;
    ModelKind kind = ModelKind::RMBG;
    uint32_t capture_limit = 1280;
    uint64_t generation = 0, completed = 0;
    double inference_ms = 0;
    uint64_t recurrent_frames = 0;
    bool recurrent_on_gpu = false;
    std::string message = "Waiting for model";
};
class Worker {
public:
    Worker();
    ~Worker();
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;
    void configure(ModelConfig config, bool force = false);
    void set_smoothing(float value);
    std::vector<uint8_t> acquire_rgba(size_t bytes);
    bool submit(Frame frame);
    WorkerStatus status() const;
    std::shared_ptr<const Mask> latest() const;
private:
    void run();
    void recycle_rgba(std::vector<uint8_t> &buffer); // mutex_ held; never recycles active inference.
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false, reload_ = false, configured_ = false;
    float smoothing_ = 0.0f;
    ModelConfig requested_;
    WorkerStatus status_;
    std::optional<Frame> pending_;
    std::shared_ptr<const Mask> mask_;
    std::array<std::vector<uint8_t>, 3> free_rgba_;
    std::thread thread_;
};
}
