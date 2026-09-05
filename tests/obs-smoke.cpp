#include "../tools/png-image.hpp"
#include <obs.h>
#include <obs-nix-platform.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
struct ImageSource { image_io::Image image; gs_texture_t *texture = nullptr; };
void *create_image(obs_data_t *settings, obs_source_t *)
{
    try {
        auto *s = new ImageSource{image_io::read(obs_data_get_string(settings, "path")), nullptr};
        obs_enter_graphics(); const uint8_t *ptr = s->image.rgba.data();
        s->texture = gs_texture_create(s->image.width, s->image.height, GS_RGBA, 1, &ptr, 0);
        obs_leave_graphics(); return s;
    } catch (...) { return nullptr; }
}
struct Capture { std::mutex mutex; image_io::Image image{0, 0, {}}; size_t frames = 0; };
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
}
}
int main(int argc, char **argv)
{
    if (argc != 7) {
        std::cerr << "Usage: obs-rmbg-smoke PLUGIN.so DATA_DIR MODEL.onnx INPUT.png OUTPUT.png cpu|cuda|auto\n"; return 2;
    }
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
    info.get_width = [](void *data) { return static_cast<ImageSource *>(data)->image.width; };
    info.get_height = [](void *data) { return static_cast<ImageSource *>(data)->image.height; };
    info.video_render = [](void *data, gs_effect_t *effect) {
        auto *s = static_cast<ImageSource *>(data); gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), s->texture);
        gs_draw_sprite(s->texture, 0, s->image.width, s->image.height);
    };
    obs_register_source(&info);
    obs_data_t *data = obs_data_create(); obs_data_set_string(data, "path", argv[4]);
    auto *source = obs_source_create_private("rmbg_test_image", "Test input", data); obs_data_release(data);
    data = obs_data_create(); obs_data_set_string(data, "model_path", argv[3]); obs_data_set_string(data, "device", argv[6]);
    obs_data_set_int(data, "stale_ms", 10000);
    auto *filter = obs_source_create_private("obs_rmbg_filter", "Test removal", data); obs_data_release(data);
    if (!source || !filter) { if (source) obs_source_release(source); if (filter) obs_source_release(filter); obs_shutdown(); return 6; }
    obs_source_filter_add(source, filter);
    // A scene scales the input to the output canvas, exercising the normal filter chain.
    auto *scene = obs_scene_create_private("RMBG test scene");
    auto *item = obs_scene_add(scene, source);
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
    if (success) {
        std::lock_guard<std::mutex> lock(captured.mutex);
        image_io::write(argv[5], captured.image);
        filtered = captured.image;
        size_t transparent = 0, opaque = 0;
        for (size_t i = 3; i < captured.image.rgba.size(); i += 4) { transparent += captured.image.rgba[i] < 20; opaque += captured.image.rgba[i] > 235; }
        std::cout << "Captured " << captured.frames << " frames; transparent=" << transparent << "; opaque=" << opaque << '\n';
        success = transparent > 500 && opaque > 500;
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
        if (!filtered.rgba.empty()) {
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
