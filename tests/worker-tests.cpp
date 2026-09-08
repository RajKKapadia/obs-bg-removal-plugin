#include "worker.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {
struct InferenceGate {
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t started = 0, released = 0;
    bool release_all = false;
} gate;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class Predicate> void await(Predicate predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, "Worker operation timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
void await_started(uint64_t timestamp)
{
    await([&] { std::lock_guard<std::mutex> lock(gate.mutex); return gate.started == timestamp; });
}
void release(uint64_t timestamp)
{
    { std::lock_guard<std::mutex> lock(gate.mutex); gate.released = timestamp; }
    gate.cv.notify_all();
}
// Unblock inference before the worker joins, including assertion failures.
struct ReleaseOnExit {
    ~ReleaseOnExit()
    {
        { std::lock_guard<std::mutex> lock(gate.mutex); gate.release_all = true; }
        gate.cv.notify_all();
    }
};
}

// This target links the real worker to a gated model double, allowing exact
// concurrency checks without GPU timing, model downloads, or inference races.
namespace rmbg {
uint64_t monotonic_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
Model::Model(const ModelConfig &) { width_ = height_ = 1; backend_ = "test"; }
Mask Model::run(const Frame &frame)
{
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.started = frame.timestamp_ns;
    gate.cv.notify_all();
    gate.cv.wait(lock, [&] { return gate.release_all || gate.released >= frame.timestamp_ns; });
    Mask mask;
    mask.pixels = {frame.rgba[0]}; mask.width = mask.height = 1;
    mask.timestamp_ns = frame.timestamp_ns; mask.generation = frame.generation;
    mask.source_width = frame.source_width; mask.source_height = frame.source_height;
    return mask;
}
}

int main()
{
    try {
        rmbg::Worker worker;
        ReleaseOnExit unblock;
        worker.configure({"first", rmbg::Device::CPU, 1});
        await([&] { return worker.status().ready; });
        const auto generation = worker.status().generation;
        auto frame = [&](uint64_t timestamp) {
            rmbg::Frame f; f.timestamp_ns = timestamp; f.generation = generation;
            f.source_width = f.source_height = f.width = f.height = 1;
            f.rgba = {255, 255, 255, 255}; return f;
        };
        auto first = frame(1), dropped = frame(2), newest = frame(3);
        first.capture_token = std::make_shared<rmbg::CaptureToken>();
        dropped.capture_token = std::make_shared<rmbg::CaptureToken>();
        newest.capture_token = std::make_shared<rmbg::CaptureToken>();
        std::weak_ptr<const rmbg::CaptureToken> first_token = first.capture_token, dropped_token = dropped.capture_token;
        std::weak_ptr<const rmbg::CaptureToken> newest_token = newest.capture_token;
        const auto *active_bytes = newest.rgba.data();
        const auto *dropped_bytes = dropped.rgba.data();
        require(worker.submit(std::move(first)), "Ready worker must accept a frame");
        await_started(1);
        require(worker.submit(std::move(dropped)) && worker.submit(std::move(newest)), "Capture must overlap active inference");
        require(dropped_token.expired() && !first_token.expired() && !newest_token.expired(),
                "Dropped pending video is released while active and newest video remain pinned");
        auto recycled = worker.acquire_rgba(4);
        require(recycled.data() == dropped_bytes && recycled.data() != active_bytes,
                "Reuse dropped capture storage without aliasing queued or active inference");
        recycled[0] = 0;
        require(worker.status().busy && !worker.latest(), "Inference remains active while frames wait");
        auto obsolete = frame(4); obsolete.generation = generation - 1;
        require(!worker.submit(std::move(obsolete)), "Reject old-generation captures without replacing pending work");
        release(1);
        await_started(3);
        require(worker.latest() && worker.latest()->timestamp_ns == 1 && worker.status().busy,
                "Publish finished work and immediately process the newest waiting frame, skipping frame 2");
        auto displayed = worker.latest();
        require(displayed->capture_token == first_token.lock(), "Publish a mask with its exact captured-video identity");
        release(3);
        await([&] { return !worker.status().busy; });
        require(worker.latest()->timestamp_ns == 3 && worker.status().completed == 2,
                "Only the in-flight and newest pending frames should be processed");
        const auto diagnostic = worker.status(true);
        require(diagnostic.replaced_frames == 1 && diagnostic.replacements.count == 1 && diagnostic.processing.count == 2 &&
                diagnostic.queue_wait.count == 2 && diagnostic.queue_wait.mean >= 0,
                "Diagnostics count actual replacements and completions without counting rejected captures");
        require(worker.latest()->pixels[0] == 255, "Reusing capture storage must not corrupt in-flight pixels");
        require(!first_token.expired(), "A displayed pair pins its video after the worker publishes a replacement");
        displayed.reset();
        await([&] { return first_token.expired(); });

        require(worker.submit(frame(4)), "Accept work after the previous batch");
        await_started(4);
        require(worker.submit(frame(5)), "Accept pending work before reconfiguration");
        worker.configure({"second", rmbg::Device::CPU, 1});
        require(!worker.latest() && !worker.submit(frame(6)), "Model change invalidates output and blocks submissions during reload");
        release(4);
        await([&] { return worker.status().ready; });
        require(!worker.latest() && worker.status().completed == 2 && !worker.status().busy,
                "Discard old in-flight results and pending frames when reconfiguring");
        auto fresh = frame(7); fresh.generation = worker.status().generation;
        require(worker.submit(std::move(fresh)), "Replacement model accepts its own generation");
        await_started(7); release(7);
        await([&] { return !worker.status().busy; });
        require(worker.latest()->generation == generation + 1 && worker.latest()->timestamp_ns == 7,
                "Publish only replacement-model results after reload");
        std::cout << "Worker concurrency checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
