# nms_bev_iou

> 面向使用者的算子文档。依据算子代码 `applications/autonomous_driving/nms_bev_iou/arch35/nms_bev_iou.asc` 编写。

## 算子概述

`nms_bev_iou` 计算旋转 BEV（俯视）框两两之间的蒙特卡洛 IoU 矩阵，是旋转框非极大值抑制（NMS）的计算热点核心。与复合算子 `fusionrepo.nms_bev`（完整 NMS 流程）互补：本算子只输出 IoU 矩阵，供调用方组合贪心抑制策略。

每个框在其局部坐标系内取 **4×5=20 个确定性分层网格采样点**，旋转到全局坐标系后统计落入对方框的比例来估计相交面积：

```
inter(i, j) ≈ area_j × P(框 j 的采样点落入框 i)
IoU(i, j)   = inter / (area_i + area_j − inter)
```

采样为固定分层网格（非随机数、非共线），**同一输入多次执行结果逐位一致**。

## 接口列表

| 接口名 | 说明 |
|--------|------|
| `torch.ops.ops_multimodal_fusion.nms_bev_iou` | 旋转 BEV 框蒙特卡洛 IoU 矩阵（确定性分层网格采样）|

## 产品支持情况

| 芯片系列 | 支持情况 |
|---------|---------|
| Ascend910B | ❌ |
| Ascend910_93 | ❌ |
| Ascend950 | ✅ |

> 单架构交付：仅 `arch35`（dav-3510 / Ascend950）。

## 接口详情

### `torch.ops.ops_multimodal_fusion.nms_bev_iou`

#### 函数原型

```python
torch.ops.ops_multimodal_fusion.nms_bev_iou(Tensor boxes_bev) -> Tensor
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| boxes_bev | Tensor (N, 7) fp32 | 旋转 3D 框 `[cx, cy, cz, dx, dy, dz, yaw]`（仅使用 BEV 分量）|

#### 返回值

| 值 | 类型 | 说明 |
|----|------|------|
| iou | Tensor (N, N) fp32 | 框对 IoU 矩阵，对角线恒为 1 |

#### 约束

- 输入必须为 2 维 (N, 7) fp32 张量，否则抛 `RuntimeError`
- N = 0 时返回 (0, 0) 空矩阵

## 实现要点

- **Host 预打包**：cos/sin/hw/hh 等在 host 侧用 torch 算好，打包为 (N, 8) fp32 传入 kernel，kernel 内无三角函数调用
- **三层 Reg 向量化**（参照 searchsorted 的成熟结构）：
  - Layer 1 `__simd_vf__`：每对的 20 个采样点占 20 个向量 lane，广播 FMA 旋转 + `Abs`/`Compares` 框内判定 + `Reduce<SUM>` 内点计数一次完成（~10 条向量指令替代 20×8 次标量 FMA）
  - Layer 2 `__aicore__`：LocalTensor 指针桥接
  - Layer 3 kernel：每 block staging 全部框到 UB，逐对调用 vf，结果 UB 暂存后单次 `DataCopyPad` 回写 GM（GlobalTensor 不支持标量写）
- **UB 对齐**：采样表 sampX/sampY 各占独立 TBuf（`LoadAlign` 要求 32B 对齐）

## 精度与性能

### 精度（MERE/MARE 口径，生态算子开源精度标准，fp32 阈值 2⁻¹³）

| 规模 | MERE | MARE | 结论 |
|------|------|------|------|
| N = 1~1000 全部 | ≤ 1.4e-08 | ≤ 8.9e-07 | **达标**（与 torch 组合参考逐位一致级别）|

### 性能（实测，对比 torch 组合实现）

| N | kernel | torch 组合 | 比值 |
|---|--------|-----------|------|
| ≤ 50 | 0.11 ms | 0.31 ms | **2.8~3.7x 快** |
| 200 | 0.22 ms | 0.33 ms | 持平 |
| 500 | 1.17 ms | 0.34 ms | 慢（每对一次 vf_call 调度开销）|

> 已知优化方向：多对打包共享单次 vf_call（每 64 对 = 1280 lane 分 chunk），消除逐对调度开销后大 N 可进一步提速。

## 与复合算子 nms_bev 的关系

| | 本算子（kernel 形态）| fusionrepo.nms_bev（复合形态）|
|---|---|---|
| 接口 | `torch.ops.ops_multimodal_fusion.nms_bev_iou(boxes)` | `torch.ops.fusionrepo.nms_bev(boxes, scores, thr)` |
| 输出 | (N, N) IoU 矩阵 | (M,) keep 索引 |
| 交付 | whl（`applications/autonomous_driving/nms_bev_iou/arch35/`）| 源码 import（`common/ops/nms_bev.py`）|
| 定位 | IoU 矩阵计算热点（可独立复用）| 完整 NMS 流程（排序 + IoU + 贪心不动点抑制）|

二者采样口径完全一致（4×5 分层网格），IoU 矩阵逐位一致，可互换组合。

## 测试

`tests/nms_bev_iou/test_nms_bev_iou.py`：26 用例（L0 门槛 7 + L1 功能 16 + L2 性能 3），golden 为同口径 torch 组合参考实现。

```bash
cd /tmp && pytest <repo>/tests/nms_bev_iou/ -v
```
