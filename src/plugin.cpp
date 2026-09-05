#include "worker.hpp"
#include <obs-module.h>
#include <graphics/vec4.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-rmbg", "en-US")
MODULE_EXPORT const char *obs_module_description(void) { return "Local RMBG-1.4 background removal with ONNX Runtime"; }
MODULE_EXPORT const char *obs_module_author(void) { return "obs-rmbg contributors"; }

namespace {
struct Settings {
    int max_fps = 15, stale_ms = 2000;
    float threshold = 0.5f, softness = 0.5f;
    bool preview = false;
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
        Settings settings;
        settings.max_fps = int(std::clamp<int64_t>(obs_data_get_int(data, "max_fps"), 1, 60));
        settings.stale_ms = int(std::clamp<int64_t>(obs_data_get_int(data, "stale_ms"), 250, 10000));
        settings.threshold = float(std::clamp(obs_data_get_double(data, "threshold"), 0.0, 1.0));
        settings.softness = float(std::clamp(obs_data_get_double(data, "softness"), 0.001, 0.5));
        settings.preview = obs_data_get_bool(data, "preview");
        { std::lock_guard<std::mutex> lock(mutex_); settings_ = settings; }
        worker_.set_smoothing(float(obs_data_get_double(data, "smoothing")));
        worker_.configure(std::move(model), retry);
    }
    obs_source_t *source() const { return source_; }
    rmbg::WorkerStatus worker_status() const { return worker_.status(); }
    std::string status() const
    {
        auto s = worker_.status();
        std::ostringstream text;
        text << s.message;
        if (s.completed) text << ". Last processing: " << std::fixed << std::setprecision(1) << s.inference_ms
                              << " ms; masks completed: " << s.completed;
        auto mask = worker_.latest();
        std::lock_guard<std::mutex> lock(mutex_);
        if (!render_error_.empty()) text << ". " << render_error_;
        if (mask && rmbg::monotonic_ns() - mask->timestamp_ns > uint64_t(settings_.stale_ms) * 1000000)
            text << ". Mask stale; original source visible";
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
            // Map a transfer issued on an earlier video tick, not immediately after staging.
            if (readback_pending_) {
                readback_pending_ = false;
                if (state.ready && staged_frame_.generation == state.generation &&
                    staged_frame_.source_width == width && staged_frame_.source_height == height &&
                    now - staged_frame_.timestamp_ns < 250000000ULL) {
                    uint8_t *bytes = nullptr; uint32_t stride = 0;
                    if (gs_stagesurface_map(stage_, &bytes, &stride)) {
                        // Allocate before mapping below; memcpy cannot throw while GPU memory is mapped.
                        for (uint32_t y = 0; y < state.height; ++y)
                            std::memcpy(staged_frame_.rgba.data() + size_t(y) * state.width * 4,
                                        bytes + size_t(y) * stride, size_t(state.width) * 4);
                        gs_stagesurface_unmap(stage_);
                        worker_.submit(std::move(staged_frame_));
                    } else error("GPU readback failed; original source used until a fresh mask is available");
                }
            }
            state = worker_.status();
            if (state.ready && !state.busy && now >= next_capture_) {
                capture_frame(state, width, height, now);
                next_capture_ = now + 1000000000ULL / uint64_t(settings.max_fps);
            }
        }
        const auto mask = worker_.latest();
        if (!mask || mask->generation != state.generation || mask->source_width != width || mask->source_height != height ||
            now - mask->timestamp_ns > uint64_t(settings.stale_ms) * 1000000) {
            obs_source_skip_video_filter(source_); return;
        }
        if (mask != uploaded_mask_) {
            if (!mask_texture_) mask_texture_ = gs_texture_create(mask->width, mask->height, GS_R8, 1, nullptr, GS_DYNAMIC);
            if (!mask_texture_) throw std::runtime_error("Could not allocate GPU mask texture");
            gs_texture_set_image(mask_texture_, mask->pixels.data(), mask->width, false);
            uploaded_mask_ = mask;
        }
        if (!obs_source_process_filter_begin(source_, GS_RGBA, OBS_NO_DIRECT_RENDERING)) return;
        gs_effect_set_texture(gs_effect_get_param_by_name(effect_, "person_mask"), mask_texture_);
        gs_effect_set_float(gs_effect_get_param_by_name(effect_, "threshold"), settings.threshold);
        gs_effect_set_float(gs_effect_get_param_by_name(effect_, "softness"), settings.softness);
        gs_effect_set_bool(gs_effect_get_param_by_name(effect_, "preview_mask"), settings.preview);
        obs_source_process_filter_end(source_, effect_, width, height);
    }
private:
    void capture_frame(const rmbg::WorkerStatus &state, uint32_t width, uint32_t height, uint64_t now)
    {
        if (!capture_) capture_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
        if (!stage_) stage_ = gs_stagesurface_create(state.width, state.height, GS_RGBA);
        if (!capture_ || !stage_) throw std::runtime_error("Could not allocate GPU capture buffers");
        // All supported model inputs are 1024x1024; source aspect ratio is restored on output.
        staged_frame_.width = state.width; staged_frame_.height = state.height;
        staged_frame_.source_width = width; staged_frame_.source_height = height;
        staged_frame_.generation = state.generation; staged_frame_.timestamp_ns = now;
        staged_frame_.rgba.resize(size_t(state.width) * state.height * 4);
        gs_texrender_reset(capture_);
        if (!gs_texrender_begin_with_color_space(capture_, state.width, state.height, GS_CS_SRGB))
            throw std::runtime_error("Could not begin source capture");
        const bool linear = gs_set_linear_srgb(false);
        const bool srgb = gs_framebuffer_srgb_enabled();
        gs_enable_framebuffer_srgb(false);
        gs_blend_state_push();
        gs_blend_function_separate(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA, GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
        vec4 clear{};
        gs_clear(GS_CLEAR_COLOR, &clear, 0, 0);
        gs_ortho(0, float(width), 0, float(height), -100, 100);
        obs_source_skip_video_filter(source_);
        gs_blend_state_pop();
        gs_enable_framebuffer_srgb(srgb);
        gs_set_linear_srgb(linear);
        gs_texrender_end(capture_);
        gs_stage_texture(stage_, gs_texrender_get_texture(capture_));
        readback_pending_ = true;
        error("");
    }
    obs_source_t *source_;
    rmbg::Worker worker_;
    mutable std::mutex mutex_;
    Settings settings_;
    std::string render_error_;
    gs_effect_t *effect_ = nullptr;
    gs_texrender_t *capture_ = nullptr;
    gs_stagesurf_t *stage_ = nullptr;
    gs_texture_t *mask_texture_ = nullptr;
    std::shared_ptr<const rmbg::Mask> uploaded_mask_;
    rmbg::Frame staged_frame_;
    bool readback_pending_ = false;
    uint64_t last_tick_ = 0, next_capture_ = 0;
};

void defaults(obs_data_t *data)
{
    char *path = obs_module_file("models/rmbg-1.4.onnx");
    obs_data_set_default_string(data, "model_path", path ? path : ""); bfree(path);
    obs_data_set_default_string(data, "device", "auto");
    obs_data_set_default_int(data, "threads", 2);
    obs_data_set_default_int(data, "max_fps", 15);
    obs_data_set_default_int(data, "stale_ms", 2000);
    obs_data_set_default_double(data, "smoothing", 0.15);
    obs_data_set_default_double(data, "threshold", 0.5);
    obs_data_set_default_double(data, "softness", 0.5);
    obs_data_set_default_bool(data, "preview", false);
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
    } catch (...) { calldata_set_string(params, "status", "Unable to read filter status"); }
}
void *create(obs_data_t *data, obs_source_t *source)
{
    try {
        auto filter = std::make_unique<Filter>(source);
        filter->update(data);
        proc_handler_add(obs_source_get_proc_handler(source),
            "void rmbg_status(out string status, out int completed, out float inference_ms, out bool ready)", status_proc, filter.get());
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
    } catch (const std::exception &e) { filter->error(e.what()); }
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
    obs_properties_add_float_slider(props, "smoothing", obs_module_text("Smoothing"), 0, 0.95, 0.05);
    obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0, 1, 0.01);
    obs_properties_add_float_slider(props, "softness", obs_module_text("Softness"), 0.001, 0.5, 0.01);
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
    blog(LOG_INFO, "[obs-rmbg] Loaded v0.1.0 (ONNX Runtime %s)", Ort::GetVersionString().c_str());
    return true;
}
