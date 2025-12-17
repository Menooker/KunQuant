#pragma once

#include "StateBuffer.hpp"
#include <fstream>
#include <vector>

namespace kun {
struct KUN_API MemoryInputStream final : public InputStreamBase {
    const char *data;
    size_t size;
    size_t pos;

    MemoryInputStream(const char *data, size_t size)
        : data(data), size(size), pos(0) {}

    bool read(void *buf, size_t len) override;
};
struct KUN_API MemoryOutputStream final : public OutputStreamBase {
    std::vector<char> buffer;

    bool write(const void *buf, size_t len) override;

    const char *getData() const { return buffer.data(); }
    size_t getSize() const { return buffer.size(); }
};
struct KUN_API FileInputStream final : public InputStreamBase {
    std::ifstream file;

    FileInputStream(const std::string &filename);

    bool read(void *buf, size_t len) override;
};
struct KUN_API FileOutputStream final : public OutputStreamBase {
    std::ofstream file;
    FileOutputStream(const std::string &filename);
    bool write(const void *buf, size_t len) override;
};
} // namespace kun
