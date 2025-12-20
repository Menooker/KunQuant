#include "IO.hpp"
#include <cstring>

namespace kun {
bool MemoryInputStream::read(void *buf, size_t len) {
    if (pos + len > size) {
        return false;
    }
    std::memcpy(buf, data + pos, len);
    pos += len;
    return true;
}

bool MemoryOutputStream::write(const void *buf, size_t len) {
    const char *cbuf = reinterpret_cast<const char *>(buf);
    buffer.insert(buffer.end(), cbuf, cbuf + len);
    return true;
}

bool MemoryRefOutputStream::write(const void *b, size_t len) {
    if (pos + len <= size) {
        std::memcpy(buf + pos, b, len);
    }
    pos += len;
    return true;
}

FileInputStream::FileInputStream(const std::string &filename)
    : file(filename, std::ios::binary) {}

bool FileInputStream::read(void *buf, size_t len) {
    if (!file.read(reinterpret_cast<char *>(buf), len)) {
        return false;
    }
    return true;
}

FileOutputStream::FileOutputStream(const std::string &filename)
    : file(filename, std::ios::binary) {}

bool FileOutputStream::write(const void *buf, size_t len) {
    if (!file.write(reinterpret_cast<const char *>(buf), len)) {
        return false;
    }
    return true;
}
} // namespace kun