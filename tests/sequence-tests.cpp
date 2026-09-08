#include "../tools/frame-sequence.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

namespace {
void require(bool test, const char *message) { if (!test) throw std::runtime_error(message); }
template<class Predicate> void wait_for(Predicate predicate)
{
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < until, "Decoder wait timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
}
int main()
{
    const auto path = std::filesystem::temp_directory_path() / ("rmbg-sequence-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(path);
        for (int i = 0; i < 4; ++i) {
            image_io::Image image{64, 64, std::vector<uint8_t>(64 * 64 * 4, uint8_t(i))};
            image_io::write((path / ("00000" + std::to_string(i) + ".png")).string(), image);
        }
        {
            image_io::SequenceReplay replay(path);
            require(replay.frame(0)->rgba[0] == 0, "First frame is available before rendering begins");
            wait_for([&] { return replay.cached() == 3; });
            for (const uint64_t index : {1, 2, 3, 4, 50, 51, 52}) {
                std::shared_ptr<const image_io::Image> frame;
                wait_for([&] { frame = replay.frame(index); return bool(frame); });
                require(frame->rgba[0] == index % 4, "Loop or dropped source frames must select the requested image");
                require(replay.cached() <= 3, "Decoder cache remains bounded");
            }
            require(replay.error().empty(), "Valid replay must not report an error");
        }
        image_io::write((path / "000001.png").string(), {32, 32, std::vector<uint8_t>(32 * 32 * 4)});
        {
            image_io::SequenceReplay replay(path);
            wait_for([&] { return !replay.error().empty(); });
            require(!replay.frame(1), "Dimension-changing sequence cannot supply mismatched pixels");
        }
        std::filesystem::remove_all(path);
        std::cout << "Sequence ordering, skip recovery, bounded cache, and dimension rejection passed\n";
    } catch (const std::exception &e) { std::filesystem::remove_all(path); std::cerr << e.what() << '\n'; return 1; }
}
