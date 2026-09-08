#pragma once
#include "png-image.hpp"
#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace image_io {
inline std::vector<std::filesystem::path> sequence_paths(const std::filesystem::path &directory)
{
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".png") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error("Sequence directory contains no PNG frames");
    return paths;
}
// Three decoded frames, one decode in progress, and the caller's displayed frame.
// Disk I/O and PNG decoding never run in the OBS render callback. Slow decoding
// repeats the previous source image; the harness reports those underruns.
class SequenceReplay {
public:
    explicit SequenceReplay(const std::filesystem::path &directory) : paths_(sequence_paths(directory))
    {
        auto first = std::make_shared<Image>(read(paths_.front().string()));
        width_ = first->width; height_ = first->height;
        if (uint64_t(width_) * height_ > 3840ULL * 2160) throw std::runtime_error("Replay supports up to 4K frames");
        cache_[0] = first;
        thread_ = std::thread([this] { decode(); });
    }
    ~SequenceReplay()
    {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        cv_.notify_one(); thread_.join();
    }
    std::shared_ptr<const Image> frame(uint64_t index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        requested_ = index;
        for (auto i = cache_.begin(); i != cache_.end() && i->first < index;) i = cache_.erase(i);
        cv_.notify_one();
        auto found = cache_.find(index);
        return found == cache_.end() ? nullptr : found->second;
    }
    std::string error() const { std::lock_guard<std::mutex> lock(mutex_); return error_; }
    size_t cached() const { std::lock_guard<std::mutex> lock(mutex_); return cache_.size(); }
    size_t size() const { return paths_.size(); }
private:
    void decode()
    {
        try {
            for (;;) {
                uint64_t index;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stopping_ || cache_.size() < 3; });
                    if (stopping_) return;
                    index = std::max(next_, requested_); next_ = index + 1;
                }
                auto image = std::make_shared<Image>(read(paths_[index % paths_.size()].string()));
                if (image->width != width_ || image->height != height_) throw std::runtime_error("Replay frame dimensions changed");
                std::lock_guard<std::mutex> lock(mutex_);
                if (index >= requested_) cache_[index] = std::move(image);
            }
        } catch (const std::exception &e) { std::lock_guard<std::mutex> lock(mutex_); error_ = e.what(); }
    }
    std::vector<std::filesystem::path> paths_;
    uint32_t width_ = 0, height_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<uint64_t, std::shared_ptr<const Image>> cache_;
    uint64_t requested_ = 0, next_ = 1;
    bool stopping_ = false;
    std::string error_;
    std::thread thread_;
};
}
