# sparse_conv/arch35 — Ascend 950PR 稀疏卷积原生算子族

面向 Ascend 950PR（arch35，28 AICore，CANN 9.0.0.beta2）的稀疏卷积
原生 AscendC 算子族，对标 CUDA/MMCV-spconv 的 forward 数学语义与
indice ABI 协议。

## 定位

```text
Level 1  LoGoNet / PV-RCNN 所需 SparseConv forward        ✅ 完成并全量验证
Level 2  SparseConv 算子族 (Conv/SubM/Pool/Inv/Transpose)  ✅ 完成
Level 3  MMCV indice ABI + 系统化 parity                   ✅ 完成 (fp32/fp16)
```

语义参考 = 稠密 conv3d / conv_transpose3d / max_pool3d 在 active 位置的
取值（`tests/sparse_conv/test_pipeline_parity.py` 稠密等价对拍）。

## 文件一览

| 文件 | 行数 | 功能 |
|---|---|---|
| `sparse_rulebook.asc` | 433 | 邻域拓扑：SubM / Regular（含 transposed 几何）rulebook |
| `sparse_scatter.asc` | 172 | 散射数值端：scatter-add / scatter-max（MTE3 硬件原子）|
| `sparse_compact.asc` | 530 | unique/compact 拓扑：AscendC 原生 sort + reg_compact |
| `sparse_pool.asc` | 434 | 稀疏池化：SubM MaxPool / Regular strided MaxPool |
| `sparse_deconv.asc` | 244 | 反向卷积：SparseInverseConv3d / SparseConvTranspose3d |
| `sparse_abi.asc` | 290 | MMCV CUDA ABI 对标层 |
| `sparse_subm.asc` | 408 | SubM 组合入口（rulebook+gather+GEMM 一体）|
| `sparse_conv.asc` | 272 | Cube GEMM 数值核心（adv_api Matmul + K pad）|
| `sparse_conv_probe.asc` | 318 | 3510 Cube API 能力探针（调试用）|
| `sparse_rulebook.inc` | 8 | 各拆分文件共享的 include 头 |

## 各文件详解

### sparse_rulebook.asc — 邻域拓扑生成

稀疏卷积的第一步：把 "输入体素 × kernel tap" 映射为输出 flat 坐标。

* **`subm_rulebook(indices, hash, d, h, w, kz=3, kh=3, kw=3, dl=1) -> nb (N, kvol)`**
  SubM（active set 保持）邻域查询。以当前体素为中心按 tap 查 hash 表
  （占位存 row+1），出界/空邻居记 0。几何：偏移 = (tap − center) × dilation，
  center = size/2，kvol ≤ 343。默认参数与旧 3×3×3 固定版逐位兼容。
* **`reg_rulebook(indices, d_out, h_out, w_out, ks, st, pd, dl, transposed=0)
  -> g_flat (N*kvol,)`**
  Regular（strided）rulebook：`out = (in + pad − tap·dilation) / stride`
  （须整除且在界内，否则 −1）。`transposed=1` 时切换转置几何
  `out = in·stride − pad + tap·dilation`（无整除约束），供转置卷积复用。
  host 入口含 `batch·d_out·h_out·w_out < INT32_MAX` 溢出防护——int32
  flat 坐标在生成处拦截。

### sparse_compact.asc — unique/compact 拓扑

Regular 路径的输出坐标去重与逆映射，完全 AscendC 原生（无 ATen sort）。

* **`reg_sort(vals) -> (sorted, perm)`**
  AscendC 原生排序：512 元素 tile 内 UB 标量 bitonic 全网络（45 级，(val,idx)
  对同步交换）+ 多 launch 两两归并（block/pair 并行，ping-pong 双缓冲）。
  tile 数补 2 的幂保证等长 run 不变量；INT32_MAX 哨兵居尾按 V 截断。
  全程批量 MTE2/MTE3 32B 对齐（无标量 GM 访问）。n=200000 → 32ms。
* **`reg_compact(g_flat, ho, wo) -> (out_buf, inv, stats)`**
  输入**任意序** g_flat（含 −1 无效）：valid filter（行主序，与调用方
  contrib 序一致）→ reg_sort → reg_compact_kernel（unique 边界检测 +
  compact 行号 + decode (z_g,y,x) + inv）→ `inv.scatter_(0, perm, ...)`
  回填 valid 行主序。**permutation 闭环在 op 内部**，返回的 inv 与
  contrib_valid 逐行对齐（int32），调用方零 permutation 处理。
  stats[0] = M（唯一输出体素数）。

### sparse_scatter.asc — 散射数值端

* **`reg_scatter_add(out, contrib, inv, m, co)`**
  `out[inv[i]] += contrib[i]`，SetAtomicAdd + DataCopyPad（MTE3 硬件
  原子，官方 grid_sample 同款模式）。通道按 128 float 分块（kernel 带
  stride/cOff 参数，UB 行距 pad 到 8 float 保 32B 对齐）——**无 C≤128
  限制**。fp32 契约。
* **`reg_scatter_max_kernel`**（内部，供 pool 使用）
  同结构 SetAtomicMax 变体。

### sparse_pool.asc — 稀疏池化

* **`subm_max_pool(indices, features, hash, d, h, w, ks, dl)`**
  SubM maxpool（active set 不变）：`out[i] = max over taps of
  features[nb]`（nb=0 跳过，无有效 tap → 0，mmcv −inf→0 语义）。
  约束：fp32，C ∈ (0,128] 且 %8==0（特征行 MTE2 读对齐）。
* **`reg_max_pool(indices, features, d_out, h_out, w_out, ks, st, pd, dl)
  -> (features, out_indices)`**
  通用 strided 稀疏 maxpool：rulebook → reg_compact → gather →
  reg_scatter_max（通道分块）→ −inf→0。返回 `(M,C) features` +
  `(M,4) [b,z,y,x] out_indices`（stride>1 输出 active set 变化，可直接
  构造 SparseConvTensor）。

### sparse_deconv.asc — 反向卷积

* **`reg_inverse_conv(g_flat, feat_coarse, filters, h_coarse, w_coarse)`**
  SparseInverseConv3d 前向（mmcv inverse=True 语义）：复用编码器存储的
  rulebook（g_flat），通道方向转置 `X_finer[i] += Y_coarse[j] @ W[k].T`
  （[C_out]@[C_out,C_in]→[C_in]）。编排水准：reg_compact → gather →
  分块 GEMM → tap gather → reg_scatter_add 散射回 finer 行。
  注：不覆盖自生成 topology 的通用 TransposedConv（见下）。
* **`reg_transposed_conv(indices, features, filters, d_out, ..., dl)
  -> (features, out_indices)`**
  通用 SparseConvTranspose3d：**自生成 topology**（reg_rulebook
  transposed=1 几何），输出 active set 可为输入超集；权重为转置卷积
  自身定义 (K,C_in,C_out) forward 布局无转置。output_padding 由上层
  折算进 d_out/h_out/w_out 显式传入。

### sparse_abi.asc — MMCV CUDA ABI 对标层

与 `mmcv.ops.sparse_ops` 同构的调用协议（indice_key 缓存由上层
SparseConvTensor 持有、pairs 显式传递——与 mmcv 相同）：

* **`get_indice_pairs(indices, batch, spatial, out_spatial, ks, st, pd,
  dl, subm, transposed=0) -> (outids (M,4), indice_pairs [2,K,Nmax],
  indice_pair_num [K])`**
  SubM 内建 hash；Regular/Transposed 走 rulebook+compact。pairs[0] =
  input 行、pairs[1] = output 行（SubM 方向 = neighbor/center，与
  Regular 相反，已对拍锁定）。
* **`indice_conv_forward(features, filters, pairs, pair_num, num_act,
  inverse, bias?, groups=1)`**
  per-tap gather → GEMM → scatter-add 循环；filters 接受 (K,Cin,Cout)
  或 MMCV 原始 5D (kz,kh,kw,Cin,Cout)；inverse 切换角色 + Wᵀ（输出
  C_in）；groups 组通道块独立 GEMM；fp32/fp16 双路径（fp16 内部升
  fp32 累加后回转）。
* **`indice_max_pool(features, pairs, pair_num, num_act)`**
  per-tap index_reduce amax + −inf→0。

### sparse_subm.asc / sparse_conv.asc — SubM 组合入口与 GEMM 核

* `sparse_subm`：SubM 卷积组合入口（泛化 kz/kh/kw + dilation，kvol≤125，
  FP16 输入），内部含独立 tile 化 rulebook kernel。
* `sparse_conv_gemm(a, wt, bias)`：`[M,K]@[K,N]` Cube 数值核心
  （adv_api Matmul，AIC），K 自动 pad——被 Regular/SubM 前向编排复用。

### sparse_conv_probe.asc — 探针（调试）

3510 Cube/Vec API 能力探针，用于上线前验证硬件行为，不参与业务链。

## 可靠性设计约定（全家族统一）

1. **批量 32B 对齐 IO**：GM 访问全部走批量 MTE2/MTE3；规避标量 GM
   读写与 sub-32B DataCopyPad 的已知丢数据问题。
2. **通道分块**：scatter 端 128 float/块 + UB 行距 pad8，任意 C。
3. **int32 topology 自洽**：flat 坐标 < INT32_MAX 在 rulebook 生成处
   防护；INT32_MAX 保留为 sort 哨兵；inv/perm 全 int32/int64 显式。
4. **事件同步**：MTE2_S / S_MTE3 / MTE3_S 逐批 set/wait（跨迭代
   Set-在尾、Wait-在头；flush 自包含处 Set+Wait 成对）。
5. **Meta/runtime 形状一致**：meta 可精确推导的给精确值，上界推导的
   与 runtime 同一公式（含 inverse 通道、batch 维）。

## 测试

`tests/sparse_conv/`（79 项，全绿）：

* 单算子黑盒对拍（vs torch 参考）：每个注册算子独立文件；
* 稠密等价 parity（`test_pipeline_parity.py`）：conv3d /
  conv_transpose3d / max_pool3d 语义级逐元素对拍；
* ABI 矩阵（`test_abi_parity.py`）：stride{1,2} × pad{0,1} ×
  dilation{1,2} × batch{1,2} × 空输入 × boundary × bias × Cin≠Cout ×
  groups × fp16 × C∈{129,192,256}；
* 端到端：LoGoNet KITTI 全量 3769 帧回归无退化（结果见
  FusionRepo `models/logonet/results/`）。

## 已知限制

* `subm_max_pool` 特征 C ≤ 128 且 %8==0（kernel 内特征行 UB 布局）；
* `subm_rulebook`/`sparse_subm` kvol ≤ 343/125（UB outBuf 尺寸）；
* `reg_scatter_add/max` 数值路径 fp32（fp16 ABI 路径内部升 fp32）；
* scatter kernel 生命周期暂以 thread-local keepAlive 保护
  （正式实现应换 record_stream/确定完成点，见各 host 注释）。
