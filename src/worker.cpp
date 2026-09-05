#include "worker.hpp"
#include <algorithm>
#include <cmath>

namespace rmbg {
Worker::Worker() : thread_([this] { run(); }) {}
Worker::~Worker()
{
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; pending_.reset(); }
    cv_.notify_one();
    thread_.join();
}
void Worker::configure(ModelConfig config, bool force)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (configured_ && !force && requested_ == config) return;
        configured_ = true;
        requested_ = std::move(config);
        ++status_.generation;
        status_.ready = false;
        status_.message = "Loading model; original source visible";
        status_.width = status_.height = 0;
        mask_.reset(); pending_.reset(); reload_ = true;
    }
    cv_.notify_one();
}
void Worker::set_smoothing(float value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    smoothing_ = std::clamp(value, 0.0f, 0.95f);
}
bool Worker::submit(Frame frame)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // At most one in-flight frame. Never build a queue of outdated video.
        if (stopping_ || !status_.ready || status_.busy || pending_ || frame.generation != status_.generation)
            return false;
        pending_ = std::move(frame);
        status_.busy = true;
    }
    cv_.notify_one();
    return true;
}
WorkerStatus Worker::status() const
{
    std::lock_guard<std::mutex> lock(mutex_); return status_;
}
std::shared_ptr<const Mask> Worker::latest() const
{
    std::lock_guard<std::mutex> lock(mutex_); return mask_;
}
void Worker::run()
{
    std::unique_ptr<Model> model;
    while (true) {
        ModelConfig config;
        uint64_t generation;
        bool load_model;
        float smoothing;
        std::optional<Frame> frame;
        std::shared_ptr<const Mask> previous;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || reload_ || pending_; });
            if (stopping_) return;
            load_model = reload_; reload_ = false;
            config = requested_; generation = status_.generation;
            smoothing = smoothing_;
            if (!load_model) { frame = std::move(pending_); pending_.reset(); }
            previous = mask_;
        }
        try {
            if (load_model) {
                // Release old GPU allocations before opening the replacement session.
                model.reset();
                model = std::make_unique<Model>(config);
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation != status_.generation) continue;
                status_.ready = true; status_.busy = false;
                status_.width = model->width(); status_.height = model->height();
                status_.message = "Ready: " + model->backend();
                if (!model->warning().empty()) status_.message += ". " + model->warning();
                continue;
            }
            if (!frame || !model) continue;
            auto mask = std::make_shared<Mask>(model->run(*frame));
            // Smoothing is expressed at 30 Hz so slow inference doesn't add seconds of lag.
            if (previous && previous->generation == generation && previous->pixels.size() == mask->pixels.size() &&
                previous->source_width == mask->source_width && previous->source_height == mask->source_height &&
                mask->timestamp_ns > previous->timestamp_ns && mask->timestamp_ns - previous->timestamp_ns < 1000000000ULL) {
                const double elapsed = (mask->timestamp_ns - previous->timestamp_ns) / 1e9;
                const float weight = std::pow(smoothing, float(elapsed * 30));
                for (size_t i = 0; i < mask->pixels.size(); ++i)
                    mask->pixels[i] = uint8_t(std::lround(previous->pixels[i] * weight + mask->pixels[i] * (1 - weight)));
            }
            std::lock_guard<std::mutex> lock(mutex_);
            if (generation != status_.generation) continue;
            status_.busy = false;
            status_.inference_ms = mask->inference_ms;
            ++status_.completed;
            mask_ = std::move(mask);
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (generation != status_.generation) continue;
            status_.ready = false; status_.busy = false;
            status_.message = std::string("Error: ") + e.what() + ". Original source visible; retry after fixing the problem.";
            mask_.reset(); pending_.reset();
        }
    }
}
}
