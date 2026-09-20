# sparse_conv 算子族

Ascend NPU (dav-3510/arch35) 稀疏卷积 kernel, 覆盖 LoGoNet/PV-RCNN 前向所需
子集, 语义与 MMCV/spconv 前向在支持的配置下对齐。
**非完整 CUDA spconv 等价实现** — 能力边界见下表。

## 能力边界

| 能力 | 支持 |
|------|------|
| SubMConv3d 3×3×3 | ✅ |
| Regular SparseConv3d (stride/pad/bias) | ✅ |
| k=(3,1,1) 等非对称核 (kvol≤27) | ✅ |
| 任意 kernel size / kvol>27 | ❌ 显式 CHECK |
| dilation > 1 | ❌ |
| groups > 1 / Inverse / transposed / MaxPool | ❌ |
| MMCV indice_pairs ABI / indice_key 完整语义 | ❌ (上层等价缓存) |
| 通用 stream-safe 生命周期 | ⚠️ keepAlive 工程方案 |

## 算子清单

| 算子 | 对标 | 说明 |
|------|------|------|
| `subm_rulebook` | SubMConv3d RuleBook 查询 | hash 表 + 27 邻居 (v2 UB 批量写) |
| `reg_rulebook` | SparseConv3d 输出索引 | stride/pad 通用 (v2 批量流水) |
| `sparse_conv_gemm` | indice_conv GEMM 段 | adv_api Matmul 托管 + bias |
| `sparse_subm` | SubMConv3d 组合入口 | rulebook+gather (GEMM 由 sparse_conv_gemm 完成) |

## 端到端 (LoGoNet KITTI 全量 3769 帧)

- 复合编排: 55ms/帧, 3D AP11@0.50 = 85.16/78.73/73.61
- 12 层稀疏卷积: 21020→13310μs (RuleBook 跨层缓存 + int32 hash + addmm + 奇偶分组)

## 使用约束

- subm_rulebook: hash 表 int32 由 host zeros+scatter (row+1, 0=空)
- sparse_conv_gemm: wt 传 (N,K) n-major (weight.t()), K/N 16 倍数
- reg_rulebook: stride=2 场景验证; 通用 stride 建议复合路径
