#pragma once
#include <cstdint>

namespace rmbg {
class CaptureSchedule {
public:
    bool due(uint64_t video_tick, unsigned max_fps)
    {
        if (max_fps != max_fps_ || video_tick < last_tick_) next_ = 0;
        max_fps_ = max_fps;
        last_tick_ = video_tick;
        if (video_tick < next_) return false;
        const uint64_t interval = 1000000000ULL / max_fps;
        // Advance along the video clock, preserving phase across rounded video
        // ticks. Skip missed intervals after a pause instead of catching up.
        next_ = next_ ? next_ + ((video_tick - next_) / interval + 1) * interval : video_tick + interval;
        return true;
    }
private:
    uint64_t next_ = 0, last_tick_ = 0;
    unsigned max_fps_ = 0;
};
}
