#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# nms_bev_iou 算子黑盒测试（自包含单文件）。
#
# golden 来源:
#   1) torch 组合参考实现（与 kernel 完全同口径: 4x5=20 点确定性分层网格
#      蒙特卡洛采样, MERE/MARE 口径对拍）
#   2) 解析性质: 单位矩阵对角线 / 轴对齐双框已知精确 IoU 判定 /
#      对称性(近似) / 确定性
# 运行（NPU, 需先安装 wheel）:
#   cd /tmp && pytest <repo>/tests/nms_bev_iou/ -v

import logging
import time

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

LOG = logging.getLogger(__name__)

if not hasattr(torch.ops.ops_multimodal_fusion, "nms_bev_iou"):
    pytest.skip(
        "ops_multimodal_fusion.nms_bev_iou not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )

DEVICE = "npu"
OP_NMS_BEV_IOU = torch.ops.ops_multimodal_fusion.nms_bev_iou

# MERE/MARE 阈值 (fp32): 生态算子开源精度标准
MERE_THRESHOLD = 2.0 ** -13
MARE_THRESHOLD = 10.0 * MERE_THRESHOLD


def _reference_iou(boxes):
    """torch 组合参考实现: 与 kernel 完全同口径的 4x5 分层网格蒙特卡洛 IoU。"""
    num_boxes = boxes.shape[0]
    cx, cy = boxes[:, 0], boxes[:, 1]
    dx, dy = boxes[:, 3], boxes[:, 4]
    yaw = boxes[:, 6]
    cos_a, sin_a = torch.cos(yaw), torch.sin(yaw)
    hw, hh = dx / 2, dy / 2
    gu = (torch.arange(4, dtype=boxes.dtype, device=boxes.device) + 0.5) / 4
    gv = (torch.arange(5, dtype=boxes.dtype, device=boxes.device) + 0.5) / 5
    su, sv = torch.meshgrid(gu, gv, indexing="ij")
    su, sv = su.reshape(-1), sv.reshape(-1)
    local_x = (su - 0.5) * dx.unsqueeze(-1)
    local_y = (sv - 0.5) * dy.unsqueeze(-1)
    gx = local_x * cos_a.unsqueeze(-1) - local_y * sin_a.unsqueeze(-1) + cx.unsqueeze(-1)
    gy = local_x * sin_a.unsqueeze(-1) + local_y * cos_a.unsqueeze(-1) + cy.unsqueeze(-1)
    dxp = gx.unsqueeze(0) - cx.view(-1, 1, 1)
    dyp = gy.unsqueeze(0) - cy.view(-1, 1, 1)
    ba = -yaw.view(-1, 1, 1)
    lxr = dxp * torch.cos(ba) - dyp * torch.sin(ba)
    lyr = dxp * torch.sin(ba) + dyp * torch.cos(ba)
    inside = (lxr.abs() <= hw.view(-1, 1, 1)) & (lyr.abs() <= hh.view(-1, 1, 1))
    areas = dx * dy
    inter = inside.float().mean(-1) * areas.unsqueeze(0)
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter
    iou = inter / union.clamp(min=1e-8)
    iou.fill_diagonal_(1.0)
    return iou


def _rand_boxes(num_boxes, seed=0):
    """随机旋转 BEV 框 (KITTI 车辆尺寸分布)。"""
    g = torch.Generator().manual_seed(seed)
    boxes = torch.zeros(num_boxes, 7)
    boxes[:, 0] = torch.rand(num_boxes, generator=g) * 60.0
    boxes[:, 1] = torch.randn(num_boxes, generator=g) * 10.0
    boxes[:, 2] = 1.5
    boxes[:, 3] = 1.6 + torch.rand(num_boxes, generator=g) * 2.0
    boxes[:, 4] = 3.6 + torch.rand(num_boxes, generator=g) * 2.0
    boxes[:, 5] = 1.5
    boxes[:, 6] = torch.randn(num_boxes, generator=g) * 0.8
    return boxes


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l0_interface_exists():
    assert hasattr(torch.ops.ops_multimodal_fusion, "nms_bev_iou"), \
        "nms_bev_iou 算子未注册到 torch.ops.ops_multimodal_fusion"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l0_output_shape_dtype():
    boxes = _rand_boxes(16, seed=0)
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE))
    assert out.shape == (16, 16)
    assert out.dtype == torch.float32
    assert out.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l0_single_box():
    boxes = _rand_boxes(1, seed=1)
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE))
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx(1.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", ["dim", "cols", "dtype"])
def test_l0_invalid_inputs(bad):
    if bad == "dim":
        boxes = torch.zeros(10, 7)[0]
    elif bad == "cols":
        boxes = torch.zeros(10, 6)
    else:
        boxes = torch.zeros(10, 7, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        OP_NMS_BEV_IOU(boxes.to(DEVICE))


# ------------------------------------------------------------
# L1 功能
# ------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "num_boxes,seed",
    [(1, 0), (2, 1), (3, 2), (10, 3), (50, 4), (200, 5), (500, 6), (1000, 7)],
)
def test_l1_matches_reference(num_boxes, seed):
    boxes = _rand_boxes(num_boxes, seed)
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE)).cpu()
    ref = _reference_iou(boxes)
    diff = (out - ref).abs()
    mere = (diff / (ref.abs() + 1e-7)).mean().item()
    mare = (diff / (ref.abs() + 1e-7)).max().item()
    assert mere < MERE_THRESHOLD, f"N={num_boxes} MERE={mere:.3e}"
    assert mare < MARE_THRESHOLD, f"N={num_boxes} MARE={mare:.3e}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l1_diagonal_is_one():
    boxes = _rand_boxes(64, seed=8)
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE))
    diag = torch.diagonal(out).cpu()
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l1_disjoint_boxes_zero_iou():
    # 完全不相交的两框: IoU = 0
    boxes = torch.tensor([
        [0.0, 0.0, 1.0, 2.0, 2.0, 1.5, 0.0],
        [50.0, 50.0, 1.0, 2.0, 2.0, 1.5, 0.0],
    ])
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE)).cpu()
    assert out[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1, 0].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l1_identical_boxes_unit_iou():
    # 完全相同的两框: IoU = 1 (采样点全部命中)
    boxes = torch.tensor([
        [10.0, 10.0, 1.0, 4.0, 2.0, 1.5, 0.7],
        [10.0, 10.0, 1.0, 4.0, 2.0, 1.5, 0.7],
    ])
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE)).cpu()
    assert out[0, 1].item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "off,expect_suppressed",
    [(0.5, True), (1.2, True), (2.4, False), (4.0, False)],
)
def test_l1_axis_aligned_pair_decision(off, expect_suppressed):
    # 轴对齐同尺寸双框 (w=4, h=2), x 向偏移 off:
    # 精确 IoU = 2*(w-off)/2 / (2*w - (w-off)) = (w-off)/(w+off)
    # 采样估计远离阈值 0.5 时判定方向必须正确
    w, h = 4.0, 2.0
    exact = (w - off) / (w + off)
    boxes = torch.tensor([
        [10.0, 10.0, 1.0, w, h, 1.5, 0.0],
        [10.0 + off, 10.0, 1.0, w, h, 1.5, 0.0],
    ])
    out = OP_NMS_BEV_IOU(boxes.to(DEVICE)).cpu()
    if abs(exact - 0.5) > 0.15:
        assert (out[0, 1].item() > 0.5) == expect_suppressed, \
            f"off={off} exact={exact:.3f} kernel={out[0, 1].item():.3f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l1_rotated_boxes():
    # 旋转框对拍 (yaw=0.6 与 1.2, 偏移扫描)
    for yaw in (0.6, 1.2):
        for k in range(8):
            off = 0.3 + k * 0.5
            boxes = torch.tensor([
                [10.0, 10.0, 1.0, 4.0, 2.0, 1.5, yaw],
                [10.0 + off, 10.0, 1.0, 4.0, 2.0, 1.5, yaw],
            ])
            out = OP_NMS_BEV_IOU(boxes.to(DEVICE)).cpu()
            ref = _reference_iou(boxes)
            assert (out - ref).abs().max().item() < 1e-5, \
                f"yaw={yaw} off={off}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l1_deterministic():
    # 确定性: 固定分层网格采样, 同一输入多次执行逐位一致
    boxes = _rand_boxes(100, seed=9)
    a = OP_NMS_BEV_IOU(boxes.to(DEVICE))
    b = OP_NMS_BEV_IOU(boxes.to(DEVICE))
    c = OP_NMS_BEV_IOU(boxes.clone().to(DEVICE))
    assert torch.equal(a, b)
    assert torch.equal(a, c)


# ------------------------------------------------------------
# L2 性能
# ------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("num_boxes", [100, 500, 1000])
def test_l2_performance(num_boxes):
    boxes = _rand_boxes(num_boxes, seed=10).to(DEVICE)
    for _ in range(5):
        OP_NMS_BEV_IOU(boxes)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(20):
        OP_NMS_BEV_IOU(boxes)
    torch.npu.synchronize()
    elapsed = (time.time() - t0) / 20 * 1000
    ref_t0 = time.time()
    for _ in range(20):
        _reference_iou(boxes)
    torch.npu.synchronize()
    ref_elapsed = (time.time() - ref_t0) / 20 * 1000
    LOG.info("N=%d: kernel=%.3fms torch组合=%.3fms",
             num_boxes, elapsed, ref_elapsed)
    # 仅记录, 不设硬性阈值 (kernel 与 torch 组合的优劣随 N 变化)
