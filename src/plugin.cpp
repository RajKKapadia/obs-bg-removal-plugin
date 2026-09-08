#include "worker.hpp"
#include "capture-schedule.hpp"
#include <obs-module.h>
#include <graphics/vec4.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-rmbg", "en-US")
MODULE_EXPORT const char *obs_module_description(void) { return "Local RMBG-1.4 and RVM background removal with ONNX Runtime"; }
MODULE_EXPORT const char *obs_module_author(void) { return "obs-rmbg contributors"; }

namespace {
struct Settings {
    int max_fps = 15, stale_ms = 2000;
    float threshold = 0.5f, softness = 0.5f;
    bool preview = false, immediate_readback = true, match_video = false, foreground = false;
};
struct RenderStats {
    uint64_t masked_frames = 0;
    double mask_age_ms = 0, mask_age_sum_ms = 0;
    uint64_t readbacks = 0, matched_frames = 0, cache_misses = 0;
    double readback_ms = 0, readback_sum_ms = 0, readback_max_ms = 0, video_delay_ms = 0;
    uint64_t cached_frames = 0;
    uint64_t distinct_pairs = 0, repeated_presentations = 0;
    rmbg::MetricSummary age, pairs, repeats;
};
class Filter {
public:
    explicit Filter(obs_source_t *source) : source_(source)
    {
        obs_enter_graphics();
        char *path = obs_module_file("rmbg.effect");
        char *errors = nullptr;
        if (path) effect_ = gs_effect_create_from_file(path, &errors);
        const std::string error = errors ? errors : "Missing rmbg.effect";
        bfree(errors); bfree(path);
        obs_leave_graphics();
        if (!effect_) throw std::runtime_error(error);
    }
    ~Filter()
    {
        // The worker owns no graphics objects and joins on its own destruction.
        obs_enter_graphics();
        gs_stagesurface_destroy(stage_);
        gs_texrender_destroy(capture_);
        gs_texture_destroy(mask_texture_);
        gs_texture_destroy(foreground_texture_);
        for (auto &frame : video_cache_) gs_texrender_destroy(frame.texture);
        gs_effect_destroy(effect_);
        obs_leave_graphics();
    }
    void update(obs_data_t *data, bool retry = false)
    {
        rmbg::ModelConfig model;
        model.path = obs_data_get_string(data, "model_path");
        const std::string device = obs_data_get_string(data, "device");
        model.device = device == "cpu" ? rmbg::Device::CPU : device == "cuda" ? rmbg::Device::CUDA : rmbg::Device::Auto;
        model.threads = int(std::clamp<int64_t>(obs_data_get_int(data, "threads"), 1, 32));
        model.rvm_max_size = uint32_t(std::clamp<int64_t>(obs_data_get_int(data, "rvm_max_size"), 320, 1920));
        model.rvm_downsample = float(obs_data_get_double(data, "rvm_downsample"));
        Settings settings;
        settings.max_fps = int(std::clamp<int64_t>(obs_data_get_int(data, "max_fps"), 1, 60));
        settings.stale_ms = int(std::clamp<int64_t>(obs_data_get_int(data, "stale_ms"), 250, 10000));
        settings.threshold = float(std::clamp(obs_data_get_double(data, "threshold"), 0.0, 1.0));
        settings.softness = float(std::clamp(obs_data_get_double(data, "softness"), 0.001, 0.5));
        settings.preview = obs_data_get_bool(data, "preview");
        settings.immediate_readback = obs_data_get_bool(data, "immediate_readback");
        settings.match_video = obs_data_get_bool(data, "match_video");
        settings.foreground = settings.match_video && obs_data_get_bool(data, "rvm_foreground");
        model.rvm_foreground = settings.foreground;
        { std::lock_guard<std::mutex> lock(mutex_); settings_ = settings; }
        worker_.set_smoothing(float(obs_data_get_double(data, "smoothing")));
        worker_.configure(std::move(model), retry);
    }
    obs_source_t *source() const { return source_; }
    rmbg::WorkerStatus worker_status() const { return worker_.status(true); }
    RenderStats render_stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = render_stats_;
        const auto now = rmbg::monotonic_ns();
        result.age = age_.summary(now); result.pairs = pairs_.summary(now); result.repeats = repeats_.summary(now);
        return result;
    }
    std::string status() const
    {
        auto s = worker_.status(true);
        std::ostringstream text;
        text << s.message;
        if (s.completed) text << ". Last processing: " << std::fixed << std::setprecision(1) << s.inference_ms
                              << " ms; masks completed: " << s.completed;
        if (s.recurrent_frames) text << "; recurrent frames: " << s.recurrent_frames
                                    << " (state on " << (s.recurrent_on_gpu ? "GPU" : "CPU") << ")";
        auto mask = worker_.latest();
        std::lock_guard<std::mutex> lock(mutex_);
        if (render_stats_.masked_frames) text << "; mask age: " << render_stats_.mask_age_ms << " ms";
        if (render_stats_.readbacks) text << "; readback: " << render_stats_.readback_ms << " ms";
        if (settings_.match_video) text << "; matched video (delay: " << render_stats_.video_delay_ms
                                       << " ms; extra smoothing disabled)";
        if (!render_error_.empty()) text << ". " << render_error_;
        if (mask && rmbg::monotonic_ns() - mask->timestamp_ns > uint64_t(settings_.stale_ms) * 1000000)
            text << ". Mask stale; original source visible";
        const auto now = rmbg::monotonic_ns();
        const auto age = age_.summary(now), pairs = pairs_.summary(now), repeats = repeats_.summary(now);
        text << ". Last 5 s: " << s.processing.rate << " masks/s; " << pairs.rate << " distinct pairs/s; "
             << repeats.count << " repeated pairs; " << s.replacements.count << " replaced waiting frames"
             << "; processing mean/p95: " << s.processing.mean << '/' << s.processing.p95
             << " ms; queue mean/p95: " << s.queue_wait.mean << '/' << s.queue_wait.p95
             << " ms; age mean/p95: " << age.mean << '/' << age.p95 << " ms"
             << "; input: " << s.width << 'x' << s.height << "; " << s.precision << "; " << s.backend;
        if (s.kind == rmbg::ModelKind::RVM) text << "; downsample: " << std::setprecision(3) << s.downsample_ratio;
        text << "; file: " << s.model_file;
        if (settings_.foreground && s.kind == rmbg::ModelKind::RVM) text << "; experimental RVM foreground colors";
        return text.str();
    }
    void error(const std::string &message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (render_error_ != message && !message.empty())
            blog(LOG_WARNING, "[obs-rmbg] %s", message.c_str());
        render_error_ = message;
    }
    void render()
    {
        obs_source_t *target = obs_filter_get_target(source_);
        if (!target || !obs_filter_get_parent(source_)) return;
        const uint32_t width = obs_source_get_base_width(target), height = obs_source_get_base_height(target);
        constexpr gs_color_space preferred[] = {GS_CS_SRGB};
        if (obs_source_get_color_space(target, 1, preferred) != GS_CS_SRGB) {
            error("HDR source unsupported: convert to SDR before RMBG. Original source visible");
            obs_source_skip_video_filter(source_); return;
        }
        if (!width || !height) { obs_source_skip_video_filter(source_); return; }
        Settings settings;
        { std::lock_guard<std::mutex> lock(mutex_); settings = settings_; }
        auto state = worker_.status();
        const uint64_t now = rmbg::monotonic_ns();
        const uint64_t tick = obs_get_video_frame_time();
        if (tick != last_tick_) {
            last_tick_ = tick;
            if (stats_generation_ != state.generation || stats_matched_ != settings.match_video) {
                std::lock_guard<std::mutex> lock(mutex_);
                age_.reset(now); pairs_.reset(now); repeats_.reset(now);
                stats_generation_ = state.generation; stats_matched_ = settings.match_video;
                last_presented_id_ = 0;
            }
            if (!settings.foreground && foreground_texture_) {
                gs_texture_destroy(foreground_texture_); foreground_texture_ = nullptr;
            }
            readback_frame(state, width, height);
            if (!settings.match_video) {
                // Tokens own identity only, so releasing these GPU resources is safe
                // even while old inference completes. Re-enabling waits for a new pair.
                for (auto &frame : video_cache_) {
                    gs_texrender_destroy(frame.texture); frame = {};
                }
                std::lock_guard<std::mutex> lock(mutex_);
                render_stats_.cached_frames = 0;
                render_stats_.video_delay_ms = 0;
            }
            state = worker_.status();
            if (state.ready && capture_schedule_.due(tick, unsigned(settings.max_fps))) {
                capture_frame(state, width, height, now, settings.match_video);
                // Start inference this tick instead of imposing a whole video-frame wait.
                // Mapping can block briefly on the GPU; the deferred option remains available.
                if (settings.immediate_readback) readback_frame(state, width, height);
            }
        }
        const auto mask = worker_.latest();
        if (!mask || mask->generation != state.generation || mask->source_width != width || mask->source_height != height ||
            now - mask->timestamp_ns > uint64_t(settings.stale_ms) * 1000000) {
            uploaded_mask_.reset();
            obs_source_skip_video_filter(source_); return;
        }
        gs_texture_t *matching_video = nullptr;
        if (settings.match_video && mask->capture_token) {
            for (const auto &frame : video_cache_)
                if (frame.token == mask->capture_token) matching_video = gs_texrender_get_texture(frame.texture);
        }
        if (settings.match_video && !matching_video) {
            uploaded_mask_.reset();
            obs_source_skip_video_filter(source_); return;
        }
        const bool use_foreground = matching_video && settings.foreground && mask->direct_alpha;
        if (use_foreground && mask->foreground_rgba.size() != size_t(mask->width) * mask->height * 4) {
            uploaded_mask_.reset();
            error("Missing or invalid RVM foreground colors; original source visible");
            obs_source_skip_video_filter(source_); return;
        }
        if (mask != uploaded_mask_) {
            if (mask_texture_ && (gs_texture_get_width(mask_texture_) != mask->width || gs_texture_get_height(mask_texture_) != mask->height)) {
                gs_texture_destroy(mask_texture_); mask_texture_ = nullptr;
            }
            if (!mask_texture_) mask_texture_ = gs_texture_create(mask->width, mask->height, GS_R8, 1, nullptr, GS_DYNAMIC);
            if (!mask_texture_) throw std::runtime_error("Could not allocate GPU mask texture");
            gs_texture_set_image(mask_texture_, mask->pixels.data(), mask->width, false);
            if (foreground_texture_ && (!use_foreground || gs_texture_get_width(foreground_texture_) != mask->width ||
                                       gs_texture_get_height(foreground_texture_) != mask->height)) {
                gs_texture_destroy(foreground_texture_); foreground_texture_ = nullptr;
            }
            if (use_foreground) {
                if (!foreground_texture_) foreground_texture_ = gs_texture_create(mask->width, mask->height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
                if (!foreground_texture_) throw std::runtime_error("Could not allocate foreground color texture");
                gs_texture_set_image(foreground_texture_, mask->foreground_rgba.data(), mask->width * 4, false);
            }
            uploaded_mask_ = mask;
        }
        if (!matching_video && !obs_source_process_filter_begin(source_, GS_RGBA, OBS_NO_DIRECT_RENDERING)) return;
        gs_effect_set_texture(gs_effect_get_param_by_name(effect_, "person_mask"), mask_texture_);
        gs_effect_set_float(gs_effect_get_param_by_name(effect_, "threshold"), settings.threshold);
        gs_effect_set_float(gs_effect_get_param_by_name(effect_, "softness"), settings.softness);
        gs_effect_set_bool(gs_effect_get_param_by_name(effect_, "preview_mask"), settings.preview);
        gs_effect_set_bool(gs_effect_get_param_by_name(effect_, "direct_alpha"), mask->direct_alpha);
        gs_effect_set_bool(gs_effect_get_param_by_name(effect_, "use_foreground"), use_foreground);
        gs_effect_set_texture(gs_effect_get_param_by_name(effect_, "foreground_colors"), use_foreground ? foreground_texture_ : mask_texture_);
        if (matching_video) {
            const bool linear = gs_set_linear_srgb(false);
            const bool srgb = gs_framebuffer_srgb_enabled();
            gs_enable_framebuffer_srgb(false);
            gs_effect_set_texture(gs_effect_get_param_by_name(effect_, "image"), matching_video);
            // Cached captures and the effect output already contain premultiplied
            // RGB. A surrounding source/scene may leave SRCALPHA blending active;
            // using it here would multiply soft edges by alpha a second time.
            gs_blend_state_push();
            gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
            while (gs_effect_loop(effect_, "Draw")) gs_draw_sprite(matching_video, 0, width, height);
            gs_blend_state_pop();
            gs_enable_framebuffer_srgb(srgb);
            gs_set_linear_srgb(linear);
        } else obs_source_process_filter_end(source_, effect_, width, height);
        if (tick != last_stats_tick_) {
            last_stats_tick_ = tick;
            std::lock_guard<std::mutex> lock(mutex_);
            ++render_stats_.masked_frames;
            render_stats_.mask_age_ms = double(rmbg::monotonic_ns() - mask->timestamp_ns) / 1e6;
            render_stats_.video_delay_ms = matching_video ? render_stats_.mask_age_ms : 0;
            render_stats_.matched_frames += matching_video != nullptr;
            render_stats_.mask_age_sum_ms += render_stats_.mask_age_ms;
            const auto presented_at = rmbg::monotonic_ns();
            age_.add(presented_at, render_stats_.mask_age_ms);
            if (matching_video) {
                if (mask->capture_id != last_presented_id_) {
                    ++render_stats_.distinct_pairs; pairs_.add(presented_at);
                } else {
                    ++render_stats_.repeated_presentations; repeats_.add(presented_at);
                }
                last_presented_id_ = mask->capture_id;
            }
        }
    }
private:
    struct CachedVideo {
        gs_texrender_t *texture = nullptr;
        std::shared_ptr<const rmbg::CaptureToken> token;
    };
    CachedVideo *acquire_video()
    {
        for (auto &frame : video_cache_) {
            // The stage, active worker, waiting frame, and completed masks pin their
            // slots independently. Never overwrite a frame still used by any of them.
            if (!frame.token || frame.token.use_count() == 1) {
                frame.token = std::make_shared<rmbg::CaptureToken>();
                if (!frame.texture) frame.texture = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
                if (!frame.texture) throw std::runtime_error("Could not allocate matching video texture");
                return &frame;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        ++render_stats_.cache_misses;
        return nullptr; // Drop this capture instead of growing the queue or breaking a pair.
    }
    void readback_frame(const rmbg::WorkerStatus &state, uint32_t width, uint32_t height)
    {
        if (!readback_pending_) return;
        readback_pending_ = false;
        if (!state.ready || staged_frame_.generation != state.generation ||
            staged_frame_.source_width != width || staged_frame_.source_height != height ||
            rmbg::monotonic_ns() - staged_frame_.timestamp_ns >= 250000000ULL) {
            staged_frame_ = {}; return;
        }
        const auto started = rmbg::monotonic_ns();
        uint8_t *bytes = nullptr; uint32_t stride = 0;
        if (!gs_stagesurface_map(stage_, &bytes, &stride)) {
            staged_frame_ = {};
            error("GPU readback failed; original source used until a fresh mask is available"); return;
        }
        // Allocation happens before mapping. One bulk copy suffices for tight rows.
        const size_t row = size_t(staged_frame_.width) * 4;
        if (stride == row) std::memcpy(staged_frame_.rgba.data(), bytes, row * staged_frame_.height);
        else for (uint32_t y = 0; y < staged_frame_.height; ++y)
            std::memcpy(staged_frame_.rgba.data() + y * row, bytes + size_t(y) * stride, row);
        gs_stagesurface_unmap(stage_);
        worker_.submit(std::move(staged_frame_));
        staged_frame_ = {};
        const double elapsed = double(rmbg::monotonic_ns() - started) / 1e6;
        std::lock_guard<std::mutex> lock(mutex_);
        ++render_stats_.readbacks;
        render_stats_.readback_ms = elapsed;
        render_stats_.readback_sum_ms += elapsed;
        render_stats_.readback_max_ms = std::max(render_stats_.readback_max_ms, elapsed);
    }
    void render_capture(gs_texrender_t *destination, uint32_t width, uint32_t height,
                        uint32_t source_width, uint32_t source_height, gs_texture_t *cached = nullptr)
    {
        gs_texrender_reset(destination);
        if (!gs_texrender_begin_with_color_space(destination, width, height, GS_CS_SRGB))
            throw std::runtime_error("Could not begin source capture");
        const bool linear = gs_set_linear_srgb(false);
        const bool srgb = gs_framebuffer_srgb_enabled();
        gs_enable_framebuffer_srgb(false);
        gs_blend_state_push();
        gs_blend_function_separate(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA, GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
        vec4 clear{};
        gs_clear(GS_CLEAR_COLOR, &clear, 0, 0);
        gs_ortho(0, float(source_width), 0, float(source_height), -100, 100);
        if (cached) {
            // Downsample the exact retained frame entirely on the GPU. Its alpha
            // is already premultiplied, so copying must not multiply it a second time.
            gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
            auto *copy = obs_get_base_effect(OBS_EFFECT_DEFAULT);
            gs_effect_set_texture(gs_effect_get_param_by_name(copy, "image"), cached);
            while (gs_effect_loop(copy, "Draw")) gs_draw_sprite(cached, 0, source_width, source_height);
        } else obs_source_skip_video_filter(source_);
        gs_blend_state_pop();
        gs_enable_framebuffer_srgb(srgb);
        gs_set_linear_srgb(linear);
        gs_texrender_end(destination);
    }
    void capture_frame(const rmbg::WorkerStatus &state, uint32_t width, uint32_t height, uint64_t now, bool match_video)
    {
        const auto size = rmbg::capture_size(state.kind, width, height, state.capture_limit);
        auto *video = match_video ? acquire_video() : nullptr;
        if (match_video && !video) return;
        if (stage_ && (gs_stagesurface_get_width(stage_) != size.width || gs_stagesurface_get_height(stage_) != size.height)) {
            gs_stagesurface_destroy(stage_); stage_ = nullptr;
        }
        if (!stage_) stage_ = gs_stagesurface_create(size.width, size.height, GS_RGBA);
        if (!stage_) throw std::runtime_error("Could not allocate GPU readback buffer");
        staged_frame_.width = size.width; staged_frame_.height = size.height;
        staged_frame_.source_width = width; staged_frame_.source_height = height;
        staged_frame_.generation = state.generation; staged_frame_.timestamp_ns = now;
        staged_frame_.capture_id = ++next_capture_id_;
        staged_frame_.capture_token = video ? video->token : nullptr;
        staged_frame_.rgba = worker_.acquire_rgba(size_t(size.width) * size.height * 4);
        gs_texture_t *texture = nullptr;
        if (video) {
            render_capture(video->texture, width, height, width, height);
            texture = gs_texrender_get_texture(video->texture);
        }
        if (!texture || size.width != width || size.height != height) {
            if (!capture_) capture_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
            if (!capture_) throw std::runtime_error("Could not allocate GPU capture buffer");
            render_capture(capture_, size.width, size.height, width, height, texture);
            texture = gs_texrender_get_texture(capture_);
        }
        gs_stage_texture(stage_, texture);
        readback_pending_ = true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            render_stats_.cached_frames = std::count_if(video_cache_.begin(), video_cache_.end(),
                [](const CachedVideo &entry) { return entry.texture != nullptr; });
        }
        error("");
    }
    obs_source_t *source_;
    rmbg::Worker worker_;
    mutable std::mutex mutex_;
    Settings settings_;
    RenderStats render_stats_;
    rmbg::RollingMetric age_, pairs_, repeats_;
    std::string render_error_;
    gs_effect_t *effect_ = nullptr;
    gs_texrender_t *capture_ = nullptr;
    std::array<CachedVideo, 6> video_cache_{};
    gs_stagesurf_t *stage_ = nullptr;
    gs_texture_t *mask_texture_ = nullptr;
    gs_texture_t *foreground_texture_ = nullptr;
    std::shared_ptr<const rmbg::Mask> uploaded_mask_;
    rmbg::Frame staged_frame_;
    bool readback_pending_ = false;
    rmbg::CaptureSchedule capture_schedule_;
    uint64_t last_tick_ = 0;
    uint64_t last_stats_tick_ = 0;
    uint64_t next_capture_id_ = 0, last_presented_id_ = 0, stats_generation_ = 0;
    bool stats_matched_ = false;
};

void defaults(obs_data_t *data)
{
    char *path = obs_module_file("models/rmbg-1.4.onnx");
    obs_data_set_default_string(data, "model_path", path ? path : ""); bfree(path);
    obs_data_set_default_string(data, "device", "auto");
    obs_data_set_default_int(data, "threads", 2);
    obs_data_set_default_int(data, "rvm_max_size", 1280);
    obs_data_set_default_double(data, "rvm_downsample", 0);
    obs_data_set_default_int(data, "max_fps", 15);
    obs_data_set_default_int(data, "stale_ms", 2000);
    obs_data_set_default_double(data, "smoothing", 0.0);
    obs_data_set_default_double(data, "threshold", 0.5);
    obs_data_set_default_double(data, "softness", 0.5);
    obs_data_set_default_bool(data, "preview", false);
    obs_data_set_default_bool(data, "immediate_readback", true);
    obs_data_set_default_bool(data, "match_video", false);
    obs_data_set_default_bool(data, "rvm_foreground", false);
}
void status_proc(void *data, calldata_t *params)
{
    try {
        auto *filter = static_cast<Filter *>(data);
        const auto state = filter->worker_status();
        calldata_set_string(params, "status", filter->status().c_str());
        calldata_set_int(params, "completed", state.completed);
        calldata_set_float(params, "inference_ms", state.inference_ms);
        calldata_set_bool(params, "ready", state.ready);
        calldata_set_int(params, "recurrent_frames", state.recurrent_frames);
        calldata_set_bool(params, "recurrent_on_gpu", state.recurrent_on_gpu);
        calldata_set_bool(params, "is_rvm", state.kind == rmbg::ModelKind::RVM);
        calldata_set_int(params, "mask_width", state.width); calldata_set_int(params, "mask_height", state.height);
        const auto render = filter->render_stats();
        calldata_set_int(params, "readbacks", render.readbacks);
        calldata_set_float(params, "readback_sum_ms", render.readback_sum_ms);
        calldata_set_float(params, "readback_max_ms", render.readback_max_ms);
        calldata_set_int(params, "matched_frames", render.matched_frames);
        calldata_set_int(params, "cached_frames", render.cached_frames);
        calldata_set_int(params, "cache_misses", render.cache_misses);
        calldata_set_float(params, "video_delay_ms", render.video_delay_ms);
        calldata_set_int(params, "masked_frames", render.masked_frames);
        calldata_set_float(params, "mask_age_ms", render.mask_age_ms);
        calldata_set_float(params, "mask_age_sum_ms", render.mask_age_sum_ms);
        calldata_set_float(params, "mask_rate", state.processing.rate);
        calldata_set_float(params, "pair_rate", render.pairs.rate);
        calldata_set_int(params, "distinct_pairs", render.distinct_pairs);
        calldata_set_int(params, "repeated_presentations", render.repeated_presentations);
        calldata_set_int(params, "repeated_window", render.repeats.count);
        calldata_set_int(params, "replaced_frames", state.replaced_frames);
        calldata_set_int(params, "replaced_window", state.replacements.count);
        calldata_set_float(params, "processing_mean_ms", state.processing.mean);
        calldata_set_float(params, "processing_p95_ms", state.processing.p95);
        calldata_set_float(params, "queue_mean_ms", state.queue_wait.mean);
        calldata_set_float(params, "queue_p95_ms", state.queue_wait.p95);
        calldata_set_float(params, "age_mean_ms", render.age.mean);
        calldata_set_float(params, "age_p95_ms", render.age.p95);
        calldata_set_float(params, "downsample_ratio", state.downsample_ratio);
        calldata_set_string(params, "precision", state.precision.c_str());
        calldata_set_string(params, "backend", state.backend.c_str());
        calldata_set_string(params, "model_file", state.model_file.c_str());
    } catch (...) { calldata_set_string(params, "status", "Unable to read filter status"); }
}
void *create(obs_data_t *data, obs_source_t *source)
{
    try {
        auto filter = std::make_unique<Filter>(source);
        filter->update(data);
        proc_handler_add(obs_source_get_proc_handler(source),
            "void rmbg_status(out string status, out int completed, out float inference_ms, out bool ready, out int masked_frames, out float mask_age_ms, out float mask_age_sum_ms, out int recurrent_frames, out bool recurrent_on_gpu, out bool is_rvm, out int mask_width, out int mask_height, out int readbacks, out float readback_sum_ms, out float readback_max_ms, out int matched_frames, out int cached_frames, out int cache_misses, out float video_delay_ms, out float mask_rate, out float pair_rate, out int distinct_pairs, out int repeated_presentations, out int repeated_window, out int replaced_frames, out int replaced_window, out float processing_mean_ms, out float processing_p95_ms, out float queue_mean_ms, out float queue_p95_ms, out float age_mean_ms, out float age_p95_ms, out float downsample_ratio, out string precision, out string backend, out string model_file)", status_proc, filter.get());
        return filter.release();
    } catch (const std::exception &e) { blog(LOG_ERROR, "[obs-rmbg] Creation failed: %s", e.what()); return nullptr; }
}
void update(void *data, obs_data_t *settings)
{
    try { static_cast<Filter *>(data)->update(settings); }
    catch (const std::exception &e) { blog(LOG_ERROR, "[obs-rmbg] Update failed: %s", e.what()); }
}
void render(void *data, gs_effect_t *)
{
    auto *filter = static_cast<Filter *>(data);
    try { filter->render(); }
    catch (const std::exception &e) { filter->error(e.what()); obs_source_skip_video_filter(filter->source()); }
}
bool refresh(obs_properties_t *props, obs_property_t *, void *data)
{
    auto *filter = static_cast<Filter *>(data);
    if (!filter) return false;
    try {
        const auto state = filter->worker_status();
        if (!state.ready && state.message.rfind("Error:", 0) == 0) {
            obs_data_t *settings = obs_source_get_settings(filter->source());
            filter->update(settings, true); obs_data_release(settings);
        }
        obs_property_set_description(obs_properties_get(props, "status"), filter->status().c_str());
        obs_property_set_enabled(obs_properties_get(props, "quality_preset"), state.ready && state.kind == rmbg::ModelKind::RVM);
        obs_data_t *settings = obs_source_get_settings(filter->source());
        obs_property_set_enabled(obs_properties_get(props, "rvm_foreground"), state.ready && state.kind == rmbg::ModelKind::RVM && obs_data_get_bool(settings, "match_video"));
        obs_data_release(settings);
    } catch (const std::exception &e) { filter->error(e.what()); }
    return true;
}
bool apply_quality(obs_properties_t *props, obs_property_t *, void *data)
{
    auto *filter = static_cast<Filter *>(data);
    if (!filter) return false;
    const auto state = filter->worker_status();
    if (!state.ready || state.kind != rmbg::ModelKind::RVM) return false;
    obs_data_t *settings = obs_source_get_settings(filter->source());
    obs_data_set_int(settings, "max_fps", 30);
    obs_data_set_bool(settings, "match_video", true);
    obs_data_set_bool(settings, "immediate_readback", true);
    obs_data_set_double(settings, "smoothing", 0);
    obs_data_set_int(settings, "rvm_max_size", 1920);
    obs_data_set_double(settings, "rvm_downsample", 0);
    obs_data_set_double(settings, "threshold", 0.5);
    obs_data_set_double(settings, "softness", 0.5);
    obs_data_set_int(settings, "stale_ms", 2000);
    obs_data_set_bool(settings, "rvm_foreground", false);
    obs_source_update(filter->source(), settings);
    obs_properties_apply_settings(props, settings);
    obs_data_release(settings);
    return true;
}
bool matching_changed(void *data, obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
    auto *filter = static_cast<Filter *>(data);
    const auto state = filter ? filter->worker_status() : rmbg::WorkerStatus{};
    obs_property_set_enabled(obs_properties_get(props, "rvm_foreground"), state.ready && state.kind == rmbg::ModelKind::RVM && obs_data_get_bool(settings, "match_video"));
    return true;
}
obs_properties_t *properties(void *data)
{
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX model (*.onnx)", nullptr);
    auto *device = obs_properties_add_list(props, "device", obs_module_text("Device"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(device, obs_module_text("DeviceAuto"), "auto");
    obs_property_list_add_string(device, obs_module_text("DeviceCPU"), "cpu");
    obs_property_list_add_string(device, obs_module_text("DeviceCUDA"), "cuda");
    obs_properties_add_int_slider(props, "threads", obs_module_text("Threads"), 1, 32, 1);
    obs_properties_add_int_slider(props, "max_fps", obs_module_text("MaxFPS"), 1, 60, 1);
    obs_properties_add_bool(props, "immediate_readback", obs_module_text("ImmediateReadback"));
    auto *matching = obs_properties_add_bool(props, "match_video", obs_module_text("MatchVideo"));
    obs_property_set_modified_callback2(matching, matching_changed, data);
    obs_properties_add_float_slider(props, "smoothing", obs_module_text("Smoothing"), 0, 0.95, 0.05);
    obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0, 1, 0.01);
    obs_properties_add_float_slider(props, "softness", obs_module_text("Softness"), 0.001, 0.5, 0.01);
    auto *rvm = obs_properties_create();
    auto *size = obs_properties_add_list(rvm, "rvm_max_size", obs_module_text("RVMMaxSize"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(size, "640", 640);
    obs_property_list_add_int(size, "1280", 1280);
    obs_property_list_add_int(size, "1920", 1920);
    // Use a list so every value is valid, including the separate automatic mode.
    auto *ratio = obs_properties_add_list(rvm, "rvm_downsample", obs_module_text("RVMDownsample"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_FLOAT);
    obs_property_list_add_float(ratio, obs_module_text("RVMAuto"), 0);
    for (const double value : {0.125, 0.25, 0.375, 0.5, 0.6, 0.75, 1.0}) {
        std::ostringstream label; label << value;
        obs_property_list_add_float(ratio, label.str().c_str(), value);
    }
    obs_properties_add_group(props, "rvm_options", obs_module_text("RVMOptions"), OBS_GROUP_NORMAL, rvm);
    obs_properties_add_bool(rvm, "rvm_foreground", obs_module_text("RVMForeground"));
    auto *quality = obs_properties_add_button2(rvm, "quality_preset", obs_module_text("QualityPreset"), apply_quality, data);
    auto *current_filter = static_cast<Filter *>(data);
    const auto current_state = current_filter ? current_filter->worker_status() : rmbg::WorkerStatus{};
    obs_property_set_enabled(quality, current_state.ready && current_state.kind == rmbg::ModelKind::RVM);
    obs_property_set_enabled(obs_properties_get(rvm, "rvm_foreground"), false);
    obs_properties_add_int_slider(props, "stale_ms", obs_module_text("StaleTimeout"), 250, 10000, 250);
    obs_properties_add_bool(props, "preview", obs_module_text("Preview"));
    auto *filter = static_cast<Filter *>(data);
    const std::string status = filter ? filter->status() : "Choose the downloaded model to begin";
    obs_properties_add_text(props, "status", status.c_str(), OBS_TEXT_INFO);
    obs_properties_add_button2(props, "refresh", obs_module_text("Refresh"), refresh, data);
    obs_properties_add_text(props, "usage_note", obs_module_text("UsageNote"), OBS_TEXT_INFO);
    return props;
}
}

bool obs_module_load(void)
{
    obs_source_info info{};
    info.id = "obs_rmbg_filter";
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_VIDEO;
    info.get_name = [](void *) { return obs_module_text("FilterName"); };
    info.create = create;
    info.destroy = [](void *data) { delete static_cast<Filter *>(data); };
    info.update = update;
    info.get_defaults = defaults;
    info.get_properties = properties;
    info.video_render = render;
    obs_register_source(&info);
    blog(LOG_INFO, "[obs-rmbg] Loaded v0.4.0 (ONNX Runtime %s)", Ort::GetVersionString().c_str());
    return true;
}
