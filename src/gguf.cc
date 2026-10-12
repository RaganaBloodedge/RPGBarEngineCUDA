#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <sstream>

#include "bytes.h"
#include "common.h"
#include "quant.h"

namespace rpg {
namespace {

// 元数据值类型（gguf_metadata_value_type）。
enum MetaType : uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5,
  kF32 = 6, kBool = 7, kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

Json ReadMetaValue(Reader& r, uint32_t type) {
  switch (type) {
    case kU8: return Json(static_cast<int64_t>(r.U8()));
    case kI8: return Json(static_cast<int64_t>(r.I8()));
    case kU16: return Json(static_cast<int64_t>(r.U16()));
    case kI16: return Json(static_cast<int64_t>(r.I16()));
    case kU32: return Json(static_cast<int64_t>(r.U32()));
    case kI32: return Json(static_cast<int64_t>(r.I32()));
    case kF32: return Json(static_cast<double>(r.F32()));
    case kBool: return Json(r.Bool());
    case kString: return Json(r.String());
    case kU64: return Json(static_cast<double>(r.U64()));
    case kI64: return Json(static_cast<double>(r.I64()));
    case kF64: return Json(r.F64());
    case kArray: {
      const uint32_t elem_type = r.U32();
      const uint64_t n = r.U64();
      // 防御：坏文件里 n 可能是天文数字。
      RPG_CHECK(n < (1ull << 32), "元数据数组长度异常：%llu",
                static_cast<unsigned long long>(n));
      Json arr = Json::Array();
      for (uint64_t i = 0; i < n; ++i) {
        arr.PushBack(ReadMetaValue(r, elem_type));
      }
      return arr;
    }
    default:
      RPG_CHECK(false, "未知的 GGUF 元数据类型：%u", type);
  }
  return Json();
}

std::string HumanBytes(int64_t n) {
  char buf[64];
  const double v = static_cast<double>(n);
  if (n >= (1ll << 30)) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB", v / 1073741824.0);
  } else if (n >= (1ll << 20)) {
    std::snprintf(buf, sizeof(buf), "%.1f MiB", v / 1048576.0);
  } else if (n >= 1024) {
    std::snprintf(buf, sizeof(buf), "%.1f KiB", v / 1024.0);
  } else {
    std::snprintf(buf, sizeof(buf), "%lld B", static_cast<long long>(n));
  }
  return buf;
}

}  // namespace

GgufFile GgufFile::Open(const std::string& path) {
  GgufFile g;
  g.file_ = MappedFile::Open(path);
  RPG_CHECK(g.file_.size() >= 24, "文件太小，不是 GGUF：%s", path.c_str());

  Reader r(g.file_.data(), g.file_.size());

  const uint32_t magic = r.U32();
  // 规范里写的字节序列是 'G' 'G' 'U' 'F'，小端读成 u32 就是 0x46554747。
  RPG_CHECK(magic == 0x46554747u, "魔数不对（读到 0x%08X），不是 GGUF 文件",
            magic);

  const uint32_t version = r.U32();
  RPG_CHECK(version >= 1 && version <= 3, "不支持的 GGUF 版本：%u", version);

  // v1 的两个计数是 uint32，v2/v3 是 uint64。
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;
  if (version == 1) {
    n_tensors = r.U32();
    n_kv = r.U32();
  } else {
    n_tensors = r.U64();
    n_kv = r.U64();
  }
  RPG_LOG_INFO("GGUF v%u：%llu 个张量，%llu 条元数据", version,
               static_cast<unsigned long long>(n_tensors),
               static_cast<unsigned long long>(n_kv));

  // ---- 元数据 ----
  Json meta = Json::Object();
  for (uint64_t i = 0; i < n_kv; ++i) {
    const std::string key = r.String();
    const uint32_t type = r.U32();
    meta.Set(key, ReadMetaValue(r, type));
  }
  g.meta_ = std::move(meta);

  g.arch_ = g.MetaStr("general.architecture");
  RPG_CHECK(!g.arch_.empty(), "GGUF 缺 general.architecture");
  g.alignment_ = static_cast<uint32_t>(g.MetaInt("general.alignment", 32));
  RPG_CHECK(g.alignment_ > 0 && (g.alignment_ & 7u) == 0,
            "对齐值非法：%u（必须是 8 的倍数）", g.alignment_);

  // ---- 张量表 ----
  g.tensors_.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t;
    t.name = r.String();
    const uint32_t n_dims = r.U32();
    RPG_CHECK(n_dims >= 1 && n_dims <= 4, "张量 %s 的维度数异常：%u",
              t.name.c_str(), n_dims);
    t.dims.resize(n_dims);
    int64_t elems = 1;
    for (uint32_t d = 0; d < n_dims; ++d) {
      const uint64_t dim = r.U64();
      RPG_CHECK(dim > 0 && dim < (1ull << 40), "张量 %s 的维度 %u 异常：%llu",
                t.name.c_str(), d, static_cast<unsigned long long>(dim));
      t.dims[d] = static_cast<int64_t>(dim);
      elems *= t.dims[d];
    }
    t.type = r.U32();
    t.offset = static_cast<int64_t>(r.U64());
    t.n_elements = elems;

    RPG_CHECK(TypeSupported(t.type),
              "张量 %s 的类型 %u（%s）暂不支持", t.name.c_str(), t.type,
              TypeName(t.type));
    t.n_bytes = TypeRowBytes(t.type, elems);

    RPG_CHECK(g.index_.emplace(t.name, static_cast<int>(g.tensors_.size()))
                  .second,
              "张量重名：%s", t.name.c_str());
    g.tensors_.push_back(std::move(t));
  }

  // ---- 张量数据起点（对齐到 alignment）----
  g.data_start_ = static_cast<int64_t>(AlignUp(r.pos(), g.alignment_));
  RPG_CHECK(g.data_start_ <= static_cast<int64_t>(g.file_.size()),
            "张量数据起点越过文件末尾");

  // 校验每个张量都落在 mmap 区内。坏文件在这里就拦下，别等到推理时崩。
  for (const auto& t : g.tensors_) {
    RPG_CHECK(g.data_start_ + t.offset + t.n_bytes <=
                  static_cast<int64_t>(g.file_.size()),
              "张量 %s 越界：需要到 %lld，文件只有 %lld", t.name.c_str(),
              static_cast<long long>(g.data_start_ + t.offset + t.n_bytes),
              static_cast<long long>(g.file_.size()));
  }

  return g;
}

int64_t GgufFile::MetaInt(const std::string& key, int64_t def) const {
  const Json& v = meta_[key];
  if (v.IsNumber()) return v.AsInt(def);
  if (v.IsBool()) return v.AsBool() ? 1 : 0;
  return def;
}

float GgufFile::MetaFloat(const std::string& key, float def) const {
  const Json& v = meta_[key];
  return v.IsNumber() ? v.AsFloat(def) : def;
}

bool GgufFile::MetaBool(const std::string& key, bool def) const {
  const Json& v = meta_[key];
  if (v.IsBool()) return v.AsBool();
  if (v.IsNumber()) return v.AsInt() != 0;
  return def;
}

std::string GgufFile::MetaStr(const std::string& key,
                              const std::string& def) const {
  const Json& v = meta_[key];
  return v.IsString() ? v.AsString() : def;
}

int64_t GgufFile::ArchInt(const std::string& suffix, int64_t def) const {
  return MetaInt(arch_ + "." + suffix, def);
}

float GgufFile::ArchFloat(const std::string& suffix, float def) const {
  return MetaFloat(arch_ + "." + suffix, def);
}

const GgufTensor* GgufFile::Find(const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &tensors_[static_cast<size_t>(it->second)];
}

const uint8_t* GgufFile::TensorBytes(const GgufTensor& t) const {
  return file_.data() + data_start_ + t.offset;
}

void GgufFile::DequantizeTo(const GgufTensor& t, float* dst) const {
  DequantizeRow(t.type, TensorBytes(t), t.n_elements, dst);
}

std::string GgufFile::Describe() const {
  std::ostringstream os;
  os << "GGUF: " << path() << "\n";
  os << "  架构: " << arch_ << "\n";
  const std::string name = MetaStr("general.name");
  if (!name.empty()) os << "  名称: " << name << "\n";
  os << "  张量: " << tensors_.size() << " 个，对齐 " << alignment_ << " 字节\n";
  os << "  大小: " << HumanBytes(static_cast<int64_t>(file_.size())) << "\n";

  // 类型直方图：一眼看出是不是 Q4_K_M（Q4_K + Q6_K 混合）。
  auto hist = TypeHistogram();
  os << "  量化: ";
  for (size_t i = 0; i < hist.size(); ++i) {
    if (i != 0) os << ", ";
    os << hist[i].first << "×" << hist[i].second;
  }
  os << "\n";
  return os.str();
}

std::vector<std::pair<std::string, int>> GgufFile::TypeHistogram() const {
  std::vector<std::pair<std::string, int>> out;
  for (const auto& t : tensors_) {
    const char* n = TypeName(t.type);
    bool found = false;
    for (auto& kv : out) {
      if (kv.first == n) {
        ++kv.second;
        found = true;
        break;
      }
    }
    if (!found) out.emplace_back(n, 1);
  }
  return out;
}

}  // namespace rpg
