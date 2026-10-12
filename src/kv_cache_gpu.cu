// kv_cache_gpu.cu —— 显存 KV Cache 的实现（见 kv_cache_gpu.h 的说明）

#include "kv_cache_gpu.h"

#ifdef RPG_HAS_CUDA

#include <cstdio>

#include <cuda_runtime.h>

#include "common.h"
#include "cuda_common.h"

namespace rpg {

GpuKvCache::GpuKvCache(const ModelMeta& c, int64_t max_seq)
    : n_layers_(c.n_layers),
      n_kv_heads_(c.n_kv_heads),
      head_dim_(c.head_dim),
      max_seq_(max_seq) {
  RPG_CHECK(n_layers_ > 0 && n_kv_heads_ > 0 && head_dim_ > 0,
            "模型配置不完整，建不了 KV Cache（层 %d / KV 头 %d / head_dim %d）",
            n_layers_, n_kv_heads_, head_dim_);
  RPG_CHECK(max_seq_ > 0, "max_seq 必须为正，给了 %lld",
            static_cast<long long>(max_seq_));

  const size_t per = static_cast<size_t>(n_layers_) *
                     static_cast<size_t>(n_kv_heads_) *
                     static_cast<size_t>(max_seq_) *
                     static_cast<size_t>(head_dim_) * sizeof(uint16_t);
  // 不像 CPU 版那样先清零：cudaMalloc 出来没有"读到未初始化内存"这一说能靠
  // 清零救回来（读到 0 也是错），而 448 MiB 的 memset 是一次真实的启动开销。
  // 可见性由 len_ 独家决定，attention 只遍历 [0, len)。
  k_.Alloc(per);
  v_.Alloc(per);
}

uint16_t* GpuKvCache::k_layer(int32_t layer) {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  return static_cast<uint16_t*>(k_.data()) + static_cast<int64_t>(layer) * LayerElems();
}

uint16_t* GpuKvCache::v_layer(int32_t layer) {
  RPG_CHECK(layer >= 0 && layer < n_layers_, "层号 %d 越界（共 %d 层）", layer,
            n_layers_);
  return static_cast<uint16_t*>(v_.data()) + static_cast<int64_t>(layer) * LayerElems();
}

const uint16_t* GpuKvCache::k_layer(int32_t layer) const {
  return const_cast<GpuKvCache*>(this)->k_layer(layer);
}

void GpuKvCache::SetLen(int64_t n) {
  RPG_CHECK(n >= 0 && n <= max_seq_, "长度 %lld 越界（容量 %lld）",
            static_cast<long long>(n), static_cast<long long>(max_seq_));
  len_ = n;
}

std::string GpuKvCache::Describe() const {
  char buf[192];
  const double mib = static_cast<double>(bytes()) / 1048576.0;
  std::snprintf(buf, sizeof(buf),
                "KV Cache(fp16) %d 层 × %d KV 头 × %lld × %d = %.1f MiB（当前 %lld）",
                n_layers_, n_kv_heads_, static_cast<long long>(max_seq_),
                head_dim_, mib, static_cast<long long>(len_));
  return std::string(buf);
}

}  // namespace rpg

#endif  // RPG_HAS_CUDA
