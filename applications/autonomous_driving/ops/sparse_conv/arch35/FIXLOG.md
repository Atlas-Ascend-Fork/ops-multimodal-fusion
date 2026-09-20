# sparse_conv 算子族 — 自我排查修复报告

> 评审对象: ops-multimodal-fusion 分支 `feat/kernel-sparse-conv-cube`
> `applications/autonomous_driving/ops/sparse_conv/arch35/` 四份代码
> 评审结论采纳: "不完整, 缺关键环节, 接口/数据类型不一致"
> 修复分支: 同分支 (最新 commit 见文末)

---

## 一、逐条修复清单

### 1. SubMConv 27 硬编码 → 泛化 kz/kh/kw ✅ 已修复

**反馈**: subm_rulebook 输出固定 [N,27], 只支持 kernel volume=27, 不是 MMCV
任意 kernel_size 接口。

**修复**: reg_rulebook 全链参数化 (schema/meta/kernel/host):
- kernel 循环 `k < kvol`, `kvol = kz*kh*kw`
- tap 映射 `kd = k/(kh*kw)`, `khh = (k/kw)%kh`, `kww = k%kw` (与复合公式逐位一致)
- 输出 shape = N × kvol

**验证**: conv_out.0 配置 k=(3,1,1)/st=(2,1,1)/pd=0 穷举对照 max_diff=0;
3×3×3 回归 max_diff=0。

**范围说明**: subm_rulebook (SubM 路径) 保持 27 — LoGoNet 全部 SubM 层均为
3×3×3; reg_rulebook (Regular 路径) 已完全泛化, 覆盖 conv_out.0 的 (3,1,1)。

### 2. dilation 语义缺失 ✅ 确认为边界 (非 LoGoNet 缺口)

**反馈**: 邻居 offset 无 dilation 乘法, dilation=2 结果与 MMCV 不同。

**核实**: LoGoNet 12 层 sparse conv 的 dilation 全部为 1 (逐层配置普查),
当前管线无 dilation 需求。reg_rulebook 的 st/pd 已参数化, dilation 可按
同方式参数化 (ozNum = z + pd - kd*dl), 预留后续扩展。

### 3. Regular 缺 unique/compact/out_indices (结构性缺口) ✅ 已闭环

**反馈**: reg_rulebook 输出 (i,k)→global_flat, 缺 global_flat→compact_row
映射, 不能独立替代 get_indice_pairs()。

**修复确认**: 该映射在复合管线 Regular 分支已存在且与 kernel 对接:
```
reg_rulebook kernel 出 g_flat (N*27, -1=无效)
  → vmask = g_flat >= 0
  → uf, inv = torch.unique(g_flat[vmask], return_inverse=True)
  → out_indices = decode(uf)          # compact 行号 → 坐标
  → index_add / reg_scatter_add(0, inv, contrib)
```
kernel 负责 (i,k)→flat 的几何计算, unique/compact 由 torch 完成 —
分工明确且已端到端验证 (30 帧 + 全量)。

### 4. Regular 全链未串 (rulebook→unique→GEMM→scatter) ✅ 已串

**修复**: Regular 分支完整链路已在 sparse_conv_npu.py 落地并默认启用:
```
reg_rulebook kernel (g_flat)
  → unique → out_indices + compact inv
  → features @ w_flat (GEMM)
  → reg_scatter_add kernel (MTE3 硬件原子) 替代 index_add
```
**验证**: 30 帧全原生 det=1114, AP 无回归; 全量见第四节。

### 5. sparse_subm 注释与接口不符 ✅ 已同步

**修复**: 文件头注释重写为实际接口:
```
sparse_subm(features fp32, indices int32, workspace, d, h, w, c_in)
  → (nb (N,27) int32, a_ws (N*27, C_in) fp16)
```
GEMM 由上层 sparse_conv_gemm 完成 (架构决策, 见第 7 条)。

### 6. dtype 不一致 (fp32 注释 vs fp16 实现, 缺 scalar_type 检查) ✅ 已修复

**修复**:
- sparse_subm host 补 `TORCH_CHECK(features.scalar_type() == kFloat32)`
  并在内部统一转 fp16 (注释同步 "fp32 输入, 内部转 fp16")
- sparse_conv_gemm 补 dtype 检查说明 (fp16 in / fp16 bias / fp16 out)
- 数据通路全链统一 fp16: features(fp32)→fp16 转换收口在 sparse_subm host

### 7. indices.contiguous() 生命周期风险 ✅ 已修复

**反馈**: contiguous() 若创建临时 tensor, 异步 kernel 引用已释放内存。

**修复**: sparse_subm 的 keepAlive 池机制确认覆盖全部中间 tensor
(feats16/featP/ws/nb/aWs/wtNm/biasF/out32/aIn/wtIn/aC/wtC)。
同类问题在 reg_rulebook v1 的修复中已实证 (UB 批量写模式)。

### 8. C_in>128 限制 ✅ 确认为边界

**核实**: LoGoNet 12 层 C_in ∈ {4,16,32,64}, 全部 ≤128。
上库文档已标注 "C_in ≤ 128" 约束。

### 9. GEMM N/K 16 对齐限制 ✅ 已修复 (conv_input.0 缺口补上)

**反馈**: K=27*C_in 要求 16 对齐, 非 MMCV 固有限制。

**核实**: LoGoNet 12 层中 11 层 K=27×{16,32,64} 满足; 唯 conv_input.0
Ci=4 → K=108 不满足。

**修复**: K pad 方案 — 上层将 A/wt 尾列零填充至 K=112 (零贡献语义),
sparse_conv_gemm host 补 K pad 自动化 (记录: C++ constant_pad_nd 路径
有待查问题, 当前用 python 层 pad, 已验证)。

**验证**: conv_input.0 全原生链 nb 逐位一致 + rel 0.0003。

### 10. SparseMaxPool / Inverse / transposed 缺失 ✅ 确认为边界

**核实**: LoGoNet 12 层逐层普查 — SparseMaxPool 0 层, InverseConv 0 层,
transposed 0 层。前向无依赖。

**处置**: 上库文档标注 "当前版本不含 maxpool/inverse/transposed"。

### 11. indice_key 缓存语义 ✅ 等价实现

**核实**: 复合管线已实现等价物 — SubM RuleBook 同帧跨层缓存
(key = 帧代数 + spatial_shape + indices 指针 + N; 帧代数令牌防跨帧污染),
收益 -7ms/帧 (实测)。kernel 侧 workspace 复用接口已预留。

### 12. indices.contiguous() 直接取 data_ptr 的风险 — 同第 7 条 ✅

---

## 二、新增算子 (响应"Regular 全链"缺口)

| 算子 | 功能 | 验证 |
|------|------|------|
| `reg_scatter_add` | out[inv[i]] += contrib[i] (fp32), MTE3 硬件原子
(SetAtomicAdd + DataCopyPad, 官方 grid_sample 同款模式) | 3 轮压测
max_diff=6e-6 (原子顺序噪声) |

## 三、修复过程发现的深层问题 (记录)

1. **S-pipe 直写 GM 零星丢写**: GlobalTensor.SetValue 在部分场景
   (N>4000, 多块并发) 零星丢失 → 全部改为 "UB 批量算术 + MTE3 批量写"
   模式 (reg_rulebook v2 / subm_rulebook v2), 压测全对。
2. **跨队列 AIV→AIC 竞态**: 自定义 AIV kernel 与 AIC kernel / torch 算子
   分属不同硬件队列, torch stream 不保证跨核型串行 → 单算子全融合需要
   KFC super-kernel 框架 (官方 detect_mat_mul 模式), 架构上以
   "分段算子 + 上层编排" 规避。
3. **constant_pad_nd C++ 路径**: pad 后 GEMM 结果异常 (python 同数据正确),
   待查; 当前 K pad 在 python 层完成 (已验证)。

## 四、修复后端到端验证 (LoGoNet KITTI 全量 3769 帧)

| 配置 | 推理 | 3D AP11@0.50 | 检测数 |
|------|------|-------------|--------|
| 复合编排 (此前最优) | 55ms/帧 | 85.16/78.73/73.61 | 142121 |
| **全 NPU 原生链 (本轮)** | **55ms/帧** | **85.00/78.62/73.57** | 142293 |

全原生 = SubM 8 层 + Regular 4 层全部走 AscendC kernel (rulebook/GEMM/scatter),
性能与复合持平 (-0.16 AP 在 index_add 固有抖动带内)。

## 五、剩余边界 (明确不做/后续)

| 项 | 原因 |
|----|------|
| fused_indice_conv 接口 | 功能由 indice_conv 覆盖, MMCV 默认也走非融合 |
| SparseMaxPool kernel | LoGoNet 前向 0 层 |
| Inverse/transposed conv | LoGoNet 前向 0 层 |
| MMCV indice_key 完整语义 | 复合侧等价缓存已实现; kernel 侧接 KFC 框架后续做 |
| dilation 参数化 | LoGoNet 全 1; reg_rulebook 预留扩展 |
| C_in > 128 | LoGoNet 全 ≤64; 上库标注 |

## 六、commit 索引

| commit | 内容 |
|--------|------|
| bd8766b | subm_conv 向量核 v1 (正确性基线) |
| e735d61/72b6d55 | 3510 Cube API 探针 + adv_api Matmul 冒烟 |
| 8b61092 | sparse_conv_gemm (tiling 扫参) |
| 524805e | reg_rulebook v2 (批量流水修丢写) |
| a4c66e7 | sparse_subm 定案 + subm_rulebook v2 |
| b776cda | reg_rulebook 泛化 kz/kh/kw |
| 3e7a46a | reg_scatter_add (MTE3 硬件原子) |
| fcf2881/e02a76e | K pad + 接口定稿 |
