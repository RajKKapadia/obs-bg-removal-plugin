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
    uint64_t generation = 0, completed = 0;
    double inference_ms = 0;
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
    bool submit(Frame frame);
    WorkerStatus status() const;
    std::shared_ptr<const Mask> latest() const;
private:
    void run();
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false, reload_ = false, configured_ = false;
    float smoothing_ = 0.15f;
    ModelConfig requested_;
    WorkerStatus status_;
    std::optional<Frame> pending_;
    std::shared_ptr<const Mask> mask_;
    std::thread thread_;
};
}
