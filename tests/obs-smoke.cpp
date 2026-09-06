#include "../tools/png-image.hpp"
#include <obs.h>
#include <obs-nix-platform.h>
#include <chrono>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
struct ImageSource {
    image_io::Image image;
    gs_texture_t *texture = nullptr;
    std::atomic<uint32_t> width{0}, height{0};
    bool motion = false;
    uint64_t last_tick = 0, motion_frame = 0;
};
void *create_image(obs_data_t *settings, obs_source_t *)
{
    try {
        auto *s = new ImageSource{image_io::read(obs_data_get_string(settings, "path")), nullptr};
        s->width = s->image.width; s->height = s->image.height;
        s->motion = std::getenv("RMBG_TEST_MOTION") != nullptr;
        if (s->motion && (s->image.width < 64 || s->image.height < 64)) {
            delete s;
            throw std::runtime_error("Motion test requires at least 64 pixels per side");
        }
        obs_enter_graphics(); const uint8_t *ptr = s->image.rgba.data();
        s->texture = gs_texture_create(s->image.width, s->image.height, GS_RGBA, 1, &ptr, s->motion ? GS_DYNAMIC : 0);
        obs_leave_graphics(); return s;
    } catch (...) { return nullptr; }
}
struct Capture {
    std::mutex mutex;
    image_io::Image image{0, 0, {}};
    size_t frames = 0, motion_frames = 0, mismatch_pixels = 0, opaque_pixels = 0;
    bool measure_motion = false;
};
void receive(void *data, video_data *frame)
{
    auto &c = *static_cast<Capture *>(data); std::lock_guard<std::mutex> lock(c.mutex);
    for (uint32_t y = 0; y < c.image.height; ++y) {
        auto *dest = c.image.rgba.data() + size_t(y) * c.image.width * 4;
        const auto *src = frame->data[0] + size_t(y) * frame->linesize[0];
        for (uint32_t x = 0; x < c.image.width; ++x) {
            dest[x * 4] = src[x * 4 + 2]; dest[x * 4 + 1] = src[x * 4 + 1];
            dest[x * 4 + 2] = src[x * 4]; dest[x * 4 + 3] = src[x * 4 + 3];
        }
    }
    ++c.frames;
    if (c.measure_motion) {
        size_t opaque = 0, transparent = 0, mismatches = 0;
        for (size_t i = 0; i < c.image.rgba.size(); i += 4) {
            transparent += c.image.rgba[i + 3] < 20;
            if (c.image.rgba[i + 3] > 240) {
                ++opaque;
                mismatches += c.image.rgba[i] < 200;
            }
        }
        if (transparent > 500 && opaque > 500) {
            ++c.motion_frames; c.mismatch_pixels += mismatches; c.opaque_pixels += opaque;
        }
    }
}
}
int main(int argc, char **argv)
{
    if (argc != 7 && argc != 10 && argc != 11) {
        std::cerr << "Usage: obs-rmbg-smoke PLUGIN.so DATA_DIR MODEL.onnx INPUT.png OUTPUT.png cpu|cuda|auto [MAX_FPS SMOOTHING MEASURE_SECONDS [SWITCH_MODEL.onnx]]\n"; return 2;
    }
    const int measure_seconds = argc >= 10 ? std::stoi(argv[9]) : 0;
    if (argc >= 10 && (std::stoi(argv[7]) < 1 || std::stoi(argv[7]) > 60 ||
                      std::stod(argv[8]) < 0 || std::stod(argv[8]) > 0.95 || measure_seconds < 1 || measure_seconds > 60)) return 2;
    obs_set_nix_platform(OBS_NIX_PLATFORM_X11_EGL);
    if (!obs_startup("en-US", nullptr, nullptr)) return 3;
    obs_video_info video{};
    video.graphics_module = "libobs-opengl"; video.fps_num = 30; video.fps_den = 1;
    video.base_width = video.output_width = 640; video.base_height = video.output_height = 360;
    video.output_format = VIDEO_FORMAT_BGRA; video.colorspace = VIDEO_CS_709; video.range = VIDEO_RANGE_FULL;
    video.gpu_conversion = false; video.scale_type = OBS_SCALE_BILINEAR;
    if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) { obs_shutdown(); return 4; }
    obs_module_t *module = nullptr;
    if (obs_open_module(&module, argv[1], argv[2]) != MODULE_SUCCESS || !obs_init_module(module)) { obs_shutdown(); return 5; }
    obs_source_info info{};
    info.id = "rmbg_test_image"; info.type = OBS_SOURCE_TYPE_INPUT; info.output_flags = OBS_SOURCE_VIDEO;
    info.get_name = [](void *) { return "RMBG test image"; }; info.create = create_image;
    info.destroy = [](void *data) { auto *s = static_cast<ImageSource *>(data); obs_enter_graphics(); gs_texture_destroy(s->texture); obs_leave_graphics(); delete s; };
    info.get_width = [](void *data) { return static_cast<ImageSource *>(data)->width.load(); };
    info.get_height = [](void *data) { return static_cast<ImageSource *>(data)->height.load(); };
    info.update = [](void *data, obs_data_t *settings) {
        auto *s = static_cast<ImageSource *>(data);
        if (obs_data_has_user_value(settings, "width")) s->width = uint32_t(obs_data_get_int(settings, "width"));
        if (obs_data_has_user_value(settings, "height")) s->height = uint32_t(obs_data_get_int(settings, "height"));
    };
    info.video_render = [](void *data, gs_effect_t *effect) {
        auto *s = static_cast<ImageSource *>(data);
        const auto tick = obs_get_video_frame_time();
        if (s->motion && tick != s->last_tick) {
            s->last_tick = tick;
            const auto left = (++s->motion_frame * 23) % (s->image.width - s->image.width / 3);
            for (uint32_t y = 0; y < s->image.height; ++y) for (uint32_t x = 0; x < s->image.width; ++x) {
                const size_t i = (size_t(y) * s->image.width + x) * 4;
                const bool foreground = x >= left && x < left + s->image.width / 3;
                s->image.rgba[i] = foreground ? 255 : 0;
                s->image.rgba[i + 1] = foreground ? 80 : 0;
                s->image.rgba[i + 2] = foreground ? 0 : 255;
                s->image.rgba[i + 3] = 255;
            }
            gs_texture_set_image(s->texture, s->image.rgba.data(), s->image.width * 4, false);
        }
        gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), s->texture);
        gs_draw_sprite(s->texture, 0, s->width.load(), s->height.load());
    };
    obs_register_source(&info);
    obs_data_t *data = obs_data_create(); obs_data_set_string(data, "path", argv[4]);
    auto *source = obs_source_create_private("rmbg_test_image", "Test input", data); obs_data_release(data);
    data = obs_data_create(); obs_data_set_string(data, "model_path", argv[3]); obs_data_set_string(data, "device", argv[6]);
    obs_data_set_int(data, "stale_ms", 10000);
    if (const auto *readback = std::getenv("RMBG_TEST_READBACK"))
        obs_data_set_bool(data, "immediate_readback", std::string(readback) != "deferred");
    const bool match_video = std::getenv("RMBG_TEST_MATCH_VIDEO") != nullptr;
    obs_data_set_bool(data, "match_video", match_video);
    if (argc >= 10) {
        obs_data_set_int(data, "max_fps", std::stoi(argv[7]));
        obs_data_set_double(data, "smoothing", std::stod(argv[8]));
    }
    auto *filter = obs_source_create_private("obs_rmbg_filter", "Test removal", data); obs_data_release(data);
    if (!source || !filter) { if (source) obs_source_release(source); if (filter) obs_source_release(filter); obs_shutdown(); return 6; }
    obs_source_filter_add(source, filter);
    // A scene scales the input to the output canvas, exercising the normal filter chain.
    auto *scene = obs_scene_create_private("RMBG test scene");
    auto *item = obs_scene_add(scene, source);
    const auto original_width = obs_source_get_width(source), original_height = obs_source_get_height(source);
    vec2 scale; scale.x = 640.0f / obs_source_get_width(source); scale.y = 360.0f / obs_source_get_height(source);
    obs_sceneitem_set_scale(item, &scale);
    Capture captured; captured.image = {640, 360, std::vector<uint8_t>(640 * 360 * 4)};
    video_scale_info conversion{}; conversion.format = VIDEO_FORMAT_BGRA; conversion.width = 640; conversion.height = 360;
    conversion.colorspace = VIDEO_CS_709; conversion.range = VIDEO_RANGE_FULL;
    obs_add_raw_video_callback(&conversion, receive, &captured);
    obs_set_output_source(0, obs_scene_get_source(scene));
    bool success = false;
    image_io::Image filtered{0, 0, {}};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        calldata_t params{};
        proc_handler_call(obs_source_get_proc_handler(filter), "rmbg_status", &params);
        if (calldata_int(&params, "completed") >= 3) { std::cout << calldata_string(&params, "status") << '\n'; success = true; }
        const char *status = calldata_string(&params, "status");
        const bool error = status && std::string(status).rfind("Error:", 0) == 0;
        if (error) std::cerr << status << '\n';
        calldata_free(&params);
        if (success || error) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (success && measure_seconds) {
        { std::lock_guard<std::mutex> lock(captured.mutex); captured.measure_motion = std::getenv("RMBG_TEST_MOTION") != nullptr; }
        calldata_t before{}, after{};
        proc_handler_call(obs_source_get_proc_handler(filter), "rmbg_status", &before);
        const auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(measure_seconds));
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        proc_handler_call(obs_source_get_proc_handler(filter), "rmbg_status", &after);
        const auto masks = calldata_int(&after, "completed") - calldata_int(&before, "completed");
        const auto frames = calldata_int(&after, "masked_frames") - calldata_int(&before, "masked_frames");
        const double total_age = calldata_float(&after, "mask_age_sum_ms") - calldata_float(&before, "mask_age_sum_ms");
        std::cout << "Benchmark: max_fps=" << argv[7] << "; smoothing=" << argv[8]
                  << "; mask_updates_per_second=" << masks / elapsed << "; masked_video_fps=" << frames / elapsed
                  << "; mean_displayed_mask_age_ms=" << (frames ? total_age / frames : 0) << '\n';
        std::cout << calldata_string(&after, "status") << '\n';
        const auto readbacks = calldata_int(&after, "readbacks") - calldata_int(&before, "readbacks");
        const auto matched = calldata_int(&after, "matched_frames") - calldata_int(&before, "matched_frames");
        std::cout << "Readback mean ms=" << (readbacks ? (calldata_float(&after, "readback_sum_ms") -
                  calldata_float(&before, "readback_sum_ms")) / readbacks : 0)
                  << "; max ms=" << calldata_float(&after, "readback_max_ms") << "; matched frames=" << matched
                  << "; cached frames=" << calldata_int(&after, "cached_frames")
                  << "; cache misses=" << calldata_int(&after, "cache_misses") << '\n';
        success = masks > 0 && frames > 0 && calldata_bool(&after, "ready");
        if (match_video) success = success && matched == frames && calldata_int(&after, "cached_frames") <= 6;
        if (std::getenv("RMBG_TEST_MOTION")) {
            std::lock_guard<std::mutex> lock(captured.mutex);
            captured.measure_motion = false;
            const double mismatch = captured.opaque_pixels ? double(captured.mismatch_pixels) / captured.opaque_pixels : 1;
            std::cout << "Motion frames=" << captured.motion_frames << "; mismatched foreground fraction=" << mismatch << '\n';
            // Live mode is a negative control: it must visibly mismatch the moving
            // foreground. Matched mode must preserve red foreground on every frame.
            success = success && captured.motion_frames >= 20 && (match_video ? mismatch < 0.001 : mismatch > 0.01);
        }
        calldata_free(&before); calldata_free(&after);
    }
    if (success) {
        std::lock_guard<std::mutex> lock(captured.mutex);
        image_io::write(argv[5], captured.image);
        filtered = captured.image;
        size_t transparent = 0, opaque = 0;
        for (size_t i = 3; i < captured.image.rgba.size(); i += 4) { transparent += captured.image.rgba[i] < 20; opaque += captured.image.rgba[i] > 235; }
        std::cout << "Captured " << captured.frames << " frames; transparent=" << transparent << "; opaque=" << opaque << '\n';
        success = transparent > 500 && opaque > 500;
    }
    if (success && argc == 11) {
        auto snapshot = [&] {
            calldata_t params{};
            proc_handler_call(obs_source_get_proc_handler(filter), "rmbg_status", &params);
            return params;
        };
        auto wait_for = [&](int64_t completed, bool rvm, uint32_t width, uint32_t height) {
            const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (std::chrono::steady_clock::now() < end) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                auto params = snapshot();
                const bool good = calldata_bool(&params, "ready") && calldata_bool(&params, "is_rvm") == rvm &&
                    calldata_int(&params, "completed") >= completed + 3 && calldata_int(&params, "mask_width") == width &&
                    calldata_int(&params, "mask_height") == height && (!rvm || calldata_int(&params, "recurrent_frames") >= 2);
                calldata_free(&params);
                if (good) return true;
            }
            auto params = snapshot();
            std::cerr << "Timed out waiting for model: " << calldata_string(&params, "status")
                      << "; mask=" << calldata_int(&params, "mask_width") << 'x' << calldata_int(&params, "mask_height")
                      << "; completed=" << calldata_int(&params, "completed") << '\n';
            calldata_free(&params);
            return false;
        };
        auto before = snapshot();
        const bool initial_rvm = calldata_bool(&before, "is_rvm");
        calldata_free(&before);
        for (const auto dimensions : {std::pair<uint32_t, uint32_t>{640, 360}, {360, 640}, {original_width, original_height}}) {
            before = snapshot(); const auto completed = calldata_int(&before, "completed"); calldata_free(&before);
            data = obs_data_create(); obs_data_set_int(data, "width", dimensions.first); obs_data_set_int(data, "height", dimensions.second);
            obs_source_update(source, data); obs_data_release(data);
            const double factor = std::min(1.0, 1280.0 / std::max(dimensions.first, dimensions.second));
            const auto width = initial_rvm ? uint32_t(std::lround(dimensions.first * factor)) : 1024u;
            const auto height = initial_rvm ? uint32_t(std::lround(dimensions.second * factor)) : 1024u;
            const bool resized = wait_for(completed, initial_rvm, width, height);
            std::cout << "Source resize " << dimensions.first << 'x' << dimensions.second << ": " << (resized ? "passed" : "FAILED") << '\n';
            success = success && resized;
        }
        // The switch fixture is the opposite model family; switch back as well.
        for (const auto model : {std::pair<const char *, bool>{argv[10], !initial_rvm}, {argv[3], initial_rvm}}) {
            before = snapshot(); const auto completed = calldata_int(&before, "completed"); calldata_free(&before);
            data = obs_data_create(); obs_data_set_string(data, "model_path", model.first);
            obs_source_update(filter, data); obs_data_release(data);
            const double factor = std::min(1.0, 1280.0 / std::max(original_width, original_height));
            const auto width = model.second ? uint32_t(std::lround(original_width * factor)) : 1024u;
            const auto height = model.second ? uint32_t(std::lround(original_height * factor)) : 1024u;
            const bool switched = wait_for(completed, model.second, width, height);
            std::cout << "Switch to " << (model.second ? "RVM" : "RMBG") << ": " << (switched ? "passed" : "FAILED") << '\n';
            success = success && switched;
        }
    }
    if (success && std::getenv("RMBG_TEST_TOGGLE")) {
        // Exercise both switches while captures and inference are active. The
        // synthetic moving foreground makes an accidental stale pairing visible.
        for (const bool immediate : {false, true}) for (const bool matched : {false, true}) {
            data = obs_data_create();
            obs_data_set_bool(data, "immediate_readback", immediate);
            obs_data_set_bool(data, "match_video", matched);
            obs_data_set_double(data, "smoothing", matched ? 0.9 : 0);
            obs_source_update(filter, data); obs_data_release(data);
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            {
                std::lock_guard<std::mutex> lock(captured.mutex);
                captured.motion_frames = captured.mismatch_pixels = captured.opaque_pixels = 0;
                captured.measure_motion = true;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::lock_guard<std::mutex> lock(captured.mutex);
            captured.measure_motion = false;
            const double mismatch = captured.opaque_pixels ? double(captured.mismatch_pixels) / captured.opaque_pixels : 1;
            std::cout << "Toggle immediate=" << immediate << "; matched=" << matched << "; motion frames="
                      << captured.motion_frames << "; mismatched fraction=" << mismatch << '\n';
            success = success && captured.motion_frames >= 20 && (matched ? mismatch < 0.001 : mismatch > 0.01);
        }
    }
    // Exercise missing-model fail-open behavior without restarting OBS.
    data = obs_data_create(); obs_data_set_string(data, "model_path", "/nonexistent/rmbg.onnx");
    obs_source_update(filter, data); obs_data_release(data);
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    {
        std::lock_guard<std::mutex> lock(captured.mutex);
        size_t transparent = 0;
        for (size_t i = 3; i < captured.image.rgba.size(); i += 4) transparent += captured.image.rgba[i] < 250;
        std::cout << "Missing-model passthrough: nonopaque pixels=" << transparent << '\n';
        success = success && transparent == 0;
        image_io::write(std::string(argv[5]) + ".passthrough.png", captured.image);
        if (!filtered.rgba.empty() && !std::getenv("RMBG_TEST_MOTION")) {
            uint64_t difference = 0, samples = 0;
            for (size_t i = 0; i < filtered.rgba.size(); i += 4) {
                if (filtered.rgba[i + 3] <= 250) continue;
                for (size_t channel = 0; channel < 3; ++channel) {
                    difference += std::abs(int(filtered.rgba[i + channel]) - int(captured.image.rgba[i + channel]));
                    ++samples;
                }
            }
            const double error = samples ? double(difference) / samples : 255;
            std::cout << "Opaque foreground RGB mean difference from passthrough=" << error << '\n';
            success = success && error < 4;
        }
    }
    obs_remove_raw_video_callback(receive, &captured);
    obs_set_output_source(0, nullptr);
    obs_source_filter_remove(source, filter); obs_source_release(filter);
    obs_sceneitem_remove(item);
    obs_scene_release(scene); obs_source_release(source);
    obs_wait_for_destroy_queue();
    obs_shutdown(); return success ? 0 : 8;
}
