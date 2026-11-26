#include "worker.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>

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
        status_.backend.clear(); status_.precision.clear(); status_.model_file.clear(); status_.downsample_ratio = 0;
        const auto now = monotonic_ns();
        processing_.reset(now); queue_wait_.reset(now); replacements_.reset(now);
        status_.recurrent_frames = 0; status_.recurrent_on_gpu = false;
        status_.gate_label.clear(); status_.gate_ms = 0;
        mask_.reset(); pending_.reset(); reload_ = true;
    }
    cv_.notify_one();
}
void Worker::set_smoothing(float value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    smoothing_ = std::clamp(value, 0.0f, 0.95f);
}
void Worker::set_gate_shape(GateShape shape)
{
    std::lock_guard<std::mutex> lock(mutex_);
    gate_shape_ = shape;
}
std::vector<uint8_t> Worker::acquire_rgba(size_t bytes)
{
    std::vector<uint8_t> buffer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &free : free_rgba_) if (!free.empty()) { buffer.swap(free); break; }
    }
    // Reused buffers already have this size: avoid allocation and zero-filling
    // every capture before the readback overwrites all of the bytes.
    buffer.resize(bytes);
    return buffer;
}
void Worker::recycle_rgba(std::vector<uint8_t> &buffer)
{
    for (auto &free : free_rgba_) if (free.empty()) { buffer.swap(free); return; }
}
bool Worker::submit(Frame frame)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Inference may run while capture fills this single replaceable slot.
        // An overloaded worker always takes the newest waiting frame next.
        if (stopping_ || !status_.ready || frame.generation != status_.generation)
            return false;
        const auto now = monotonic_ns();
        frame.submitted_ns = now;
        if (pending_) {
            recycle_rgba(pending_->rgba);
            ++status_.replaced_frames;
            replacements_.add(now);
        }
        pending_ = std::move(frame);
        status_.busy = true;
    }
    cv_.notify_one();
    return true;
}
WorkerStatus Worker::status(bool diagnostics) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = status_;
    if (diagnostics) {
        const auto now = monotonic_ns();
        result.processing = processing_.summary(now);
        result.queue_wait = queue_wait_.summary(now);
        result.replacements = replacements_.summary(now);
    }
    return result;
}
std::shared_ptr<const Mask> Worker::latest() const
{
    std::lock_guard<std::mutex> lock(mutex_); return mask_;
}
void Worker::run()
{
    std::unique_ptr<Model> model, gate;
    while (true) {
        ModelConfig config;
        uint64_t generation;
        bool load_model;
        float smoothing;
        GateShape gate_shape;
        std::optional<Frame> frame;
        std::shared_ptr<const Mask> previous;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || reload_ || pending_; });
            if (stopping_) return;
            load_model = reload_; reload_ = false;
            config = requested_; generation = status_.generation;
            smoothing = smoothing_; gate_shape = gate_shape_;
            if (!load_model) { frame = std::move(pending_); pending_.reset(); }
            previous = mask_;
        }
        try {
            if (load_model) {
                // Release old GPU allocations before opening the replacement session.
                model.reset(); gate.reset();
                model = std::make_unique<Model>(config);
                std::string gate_label;
                if (!config.gate_path.empty() && model->kind() == ModelKind::RVM) {
                    gate_label = "unused (main model is already RVM)";
                } else if (!config.gate_path.empty()) {
                    try {
                        gate = std::make_unique<Model>(gate_model_config(config));
                    } catch (const std::exception &e) {
                        throw std::runtime_error(std::string("Person gate: ") + e.what());
                    }
                    if (gate->kind() != ModelKind::RVM)
                        throw std::runtime_error("Person gate needs an RVM model file (for example rvm_mobilenetv3_fp16.onnx)");
                    gate_label = gate->label() + " on " + gate->backend();
                }
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation != status_.generation) continue;
                status_.ready = true; status_.busy = false;
                status_.width = model->width(); status_.height = model->height();
                status_.kind = model->kind(); status_.capture_limit = model->capture_limit();
                status_.message = "Ready: " + model->backend() + "; model: " + model->label();
                status_.backend = model->backend(); status_.precision = model->precision();
                status_.model_file = std::filesystem::path(config.path).filename().string();
                status_.downsample_ratio = 0;
                const auto now = monotonic_ns();
                processing_.reset(now); queue_wait_.reset(now); replacements_.reset(now);
                if (!model->warning().empty()) status_.message += ". " + model->warning();
                status_.gate_label = gate_label;
                if (gate && !gate->warning().empty()) status_.message += ". Person gate: " + gate->warning();
                continue;
            }
            if (!frame || !model) continue;
            const double queue_ms = double(monotonic_ns() - frame->submitted_ns) / 1e6;
            auto mask = std::make_shared<Mask>(model->run(*frame));
            if (gate) apply_person_gate(*gate, *frame, *mask, gate_shape);
            mask->capture_id = frame->capture_id;
            mask->capture_token = frame->capture_token;
            // Smoothing is expressed at 30 Hz so slow inference doesn't add seconds of lag.
            if (!mask->capture_token && smoothing > 0 && (!mask->direct_alpha || mask->recurrent_frames > 1) && previous && previous->generation == generation &&
                previous->width == mask->width && previous->height == mask->height && previous->pixels.size() == mask->pixels.size() &&
                previous->source_width == mask->source_width && previous->source_height == mask->source_height &&
                mask->timestamp_ns > previous->timestamp_ns && mask->timestamp_ns - previous->timestamp_ns < 1000000000ULL) {
                const double elapsed = (mask->timestamp_ns - previous->timestamp_ns) / 1e9;
                const float weight = std::pow(smoothing, float(elapsed * 30));
                for (size_t i = 0; i < mask->pixels.size(); ++i)
                    mask->pixels[i] = uint8_t(std::lround(previous->pixels[i] * weight + mask->pixels[i] * (1 - weight)));
            }
            std::lock_guard<std::mutex> lock(mutex_);
            recycle_rgba(frame->rgba);
            if (generation != status_.generation) continue;
            status_.busy = pending_.has_value();
            status_.inference_ms = mask->inference_ms;
            status_.downsample_ratio = mask->downsample_ratio;
            const auto completed_at = monotonic_ns();
            processing_.add(completed_at, mask->inference_ms); queue_wait_.add(completed_at, queue_ms);
            status_.width = mask->width; status_.height = mask->height;
            status_.recurrent_frames = mask->recurrent_frames; status_.recurrent_on_gpu = mask->recurrent_on_gpu;
            status_.gate_ms = mask->gate_ms;
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
