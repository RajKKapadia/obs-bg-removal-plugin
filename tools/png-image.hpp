#pragma once
#include <png.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace image_io {
struct Image { uint32_t width, height; std::vector<uint8_t> rgba; };
inline Image read(const std::string &path)
{
    png_image png{}; png.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_file(&png, path.c_str()))
        throw std::runtime_error("PNG read failed: " + std::string(png.message));
    png.format = PNG_FORMAT_RGBA;
    if (!png.width || !png.height || png.width > 16384 || png.height > 16384) {
        png_image_free(&png); throw std::runtime_error("Unsupported image dimensions");
    }
    Image image{png.width, png.height, std::vector<uint8_t>(PNG_IMAGE_SIZE(png))};
    if (!png_image_finish_read(&png, nullptr, image.rgba.data(), 0, nullptr)) {
        std::string error = png.message; png_image_free(&png); throw std::runtime_error(error);
    }
    png_image_free(&png); return image;
}
inline void write(const std::string &path, const Image &image)
{
    png_image png{}; png.version = PNG_IMAGE_VERSION;
    png.width = image.width; png.height = image.height; png.format = PNG_FORMAT_RGBA;
    if (!png_image_write_to_file(&png, path.c_str(), 0, image.rgba.data(), 0, nullptr))
        throw std::runtime_error("PNG write failed: " + std::string(png.message));
}
inline uint8_t sample(const uint8_t *data, uint32_t w, uint32_t h, size_t channels, size_t c, float x, float y)
{
    x = std::clamp(x, 0.0f, float(w - 1)); y = std::clamp(y, 0.0f, float(h - 1));
    const auto x0 = uint32_t(x), y0 = uint32_t(y);
    const auto x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const float dx = x - x0, dy = y - y0;
    const float a = data[(size_t(y0) * w + x0) * channels + c] * (1 - dx) + data[(size_t(y0) * w + x1) * channels + c] * dx;
    const float b = data[(size_t(y1) * w + x0) * channels + c] * (1 - dx) + data[(size_t(y1) * w + x1) * channels + c] * dx;
    return uint8_t(std::lround(a * (1 - dy) + b * dy));
}
inline Image resize(const Image &image, uint32_t w, uint32_t h)
{
    Image result{w, h, std::vector<uint8_t>(size_t(w) * h * 4)};
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (size_t c = 0; c < 4; ++c)
                result.rgba[(size_t(y) * w + x) * 4 + c] = sample(image.rgba.data(), image.width, image.height, 4, c,
                    (x + 0.5f) * image.width / w - 0.5f, (y + 0.5f) * image.height / h - 0.5f);
    return result;
}
}
