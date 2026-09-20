# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""AscendC kernel 桥接层 (ops_multimodal_fusion whl)。

将算子仓 sparse_conv / nms_bev_iou 的 AscendC 算子接入 LoGoNet NPU 管线:

  sparse_conv_subm        融合 SubM 稀疏卷积 (hash+rulebook+gather+GEMM 单 op)
  subm_rulebook           SubM 27 邻居查询 (SubM 分步路径)
  reg_rulebook            Regular 下采样 rulebook (Regular 分步路径)
  nms_bev_iou             BEV 旋转框 IoU 矩阵 (旋转 NMS)

调度策略按实测价值确定 (msprof 同规模 A/B, aclnn 有对应物的负载回撤 at:: 组合)。
环境变量 LOGONET_NATIVE_KERNEL=0 全部回退复合实现 (A/B 对照用)。
"""
import os

import torch

try:
    import ops_multimodal_fusion  # noqa: F401
    OPS = torch.ops.ops_multimodal_fusion
    _HAS_WHL = hasattr(OPS, "sparse_conv_subm") or \
        hasattr(OPS, "subm_rulebook")
except Exception:  # whl 未安装
    OPS = None
    _HAS_WHL = False

ENABLED = _HAS_WHL and os.environ.get("LOGONET_NATIVE_KERNEL", "1") == "1"
RULEBOOK_FORCED = os.environ.get("LOGONET_KERNEL_RULEBOOK", "1") == "1"
REG_RULEBOOK_FORCED = os.environ.get("LOGONET_KERNEL_REG_RULEBOOK", "1") == "1"
SUBM_FUSED = os.environ.get("LOGONET_SUBM_FUSED", "1") == "1"

# 实测交叉点 (dav-3510, 28 AICore)
NMS_KERNEL_MAX_N = 256


def native_available():
    return ENABLED


# ------------------------------------------------------------
# 1) furthest_point_sample / ball_query — 当前 whl 未含此算子,
#    恒走复合回退 (保留调度点以兼容完整 PV-RCNN VSA 路径)
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2) nms_bev_iou — (N,7)[cx,cy,z,w,h,z,angle] -> (N,N) 确定性 IoU 矩阵
# ------------------------------------------------------------
def nms_iou_matrix(boxes):
    """boxes: (N,5) fp32 [cx,cy,w,h,angle] -> (N,N) fp32"""
    num_boxes = boxes.shape[0]
    b7 = torch.zeros((num_boxes, 7), dtype=torch.float32, device=boxes.device)
    b7[:, 0] = boxes[:, 0]
    b7[:, 1] = boxes[:, 1]
    b7[:, 3] = boxes[:, 2]
    b7[:, 4] = boxes[:, 3]
    b7[:, 5] = 1.0
    b7[:, 6] = boxes[:, 4]
    return OPS.nms_bev_iou(b7.contiguous())


def nms_use_kernel(num_boxes):
    return ENABLED and hasattr(OPS, "nms_bev_iou") \
        and 1 <= num_boxes <= NMS_KERNEL_MAX_N


def voxel_scatter_enabled():
    # 精简版 whl 不含 voxel_scatter — 恒走复合体素化 (hard_voxelize_npu)
    return False


def reg_compact_enabled():
    # 精简版 whl 不再注册 reg_compact — 恒走 Python unique 分支
    return False


# ------------------------------------------------------------
# 3) sparse_conv_subm — 融合 SubM 稀疏卷积
# ------------------------------------------------------------
def subm_fused_enabled():
    return ENABLED and SUBM_FUSED and OPS is not None and \
        hasattr(OPS, "sparse_conv_subm")


# ------------------------------------------------------------
# 4) subm_rulebook / reg_rulebook — 分步路径的拓扑生成
# ------------------------------------------------------------
def rulebook_kernel_enabled():
    return ENABLED and RULEBOOK_FORCED and hasattr(OPS, "subm_rulebook")


def reg_rulebook_kernel_enabled():
    return ENABLED and REG_RULEBOOK_FORCED and hasattr(OPS, "reg_rulebook")


