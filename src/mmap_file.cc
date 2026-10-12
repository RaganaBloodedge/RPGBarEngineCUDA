#include "mmap_file.h"

#include <utility>

#include "common.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rpg {

MappedFile::~MappedFile() { Close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept {
  *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this == &other) return *this;
  Close();
  data_ = other.data_;
  size_ = other.size_;
  path_ = std::move(other.path_);
#if defined(_WIN32)
  file_ = other.file_;
  mapping_ = other.mapping_;
  other.file_ = nullptr;
  other.mapping_ = nullptr;
#endif
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

#if defined(_WIN32)

MappedFile MappedFile::Open(const std::string& path) {
  MappedFile m;
  m.path_ = path;

  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  RPG_CHECK(h != INVALID_HANDLE_VALUE, "打开文件失败：%s（错误码 %lu）", path.c_str(),
            GetLastError());

  LARGE_INTEGER sz;
  if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0) {
    CloseHandle(h);
    RPG_CHECK(false, "文件为空或无法取大小：%s", path.c_str());
  }

  HANDLE mp = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mp == nullptr) {
    const DWORD err = GetLastError();
    CloseHandle(h);
    RPG_CHECK(false, "创建文件映射失败：%s（错误码 %lu）", path.c_str(), err);
  }

  void* view = MapViewOfFile(mp, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD err = GetLastError();
    CloseHandle(mp);
    CloseHandle(h);
    RPG_CHECK(false, "映射视图失败：%s（错误码 %lu）", path.c_str(), err);
  }

  m.file_ = h;
  m.mapping_ = mp;
  m.data_ = static_cast<const uint8_t*>(view);
  m.size_ = static_cast<size_t>(sz.QuadPart);
  return m;
}

void MappedFile::Close() {
  if (data_ != nullptr) {
    UnmapViewOfFile(data_);
    data_ = nullptr;
  }
  if (mapping_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(mapping_));
    mapping_ = nullptr;
  }
  if (file_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(file_));
    file_ = nullptr;
  }
  size_ = 0;
}

#else

MappedFile MappedFile::Open(const std::string& path) {
  MappedFile m;
  m.path_ = path;

  const int fd = ::open(path.c_str(), O_RDONLY);
  RPG_CHECK(fd >= 0, "打开文件失败：%s", path.c_str());

  struct stat st;
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    RPG_CHECK(false, "文件为空或无法取大小：%s", path.c_str());
  }

  void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                   MAP_PRIVATE, fd, 0);
  // 映射建立后 fd 就可以关了：映射本身持有对文件的引用。
  const int saved = (p == MAP_FAILED) ? errno : 0;
  ::close(fd);
  RPG_CHECK(p != MAP_FAILED, "mmap 失败：%s（errno %d）", path.c_str(), saved);

  m.data_ = static_cast<const uint8_t*>(p);
  m.size_ = static_cast<size_t>(st.st_size);
  return m;
}

void MappedFile::Close() {
  if (data_ != nullptr) {
    ::munmap(const_cast<uint8_t*>(data_), size_);
    data_ = nullptr;
  }
  size_ = 0;
}

#endif

}  // namespace rpg
