# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
voxel_projection_module

说明：
    该模块将 voxel 内点云先计算质心，再将质心投影到图像平面，
    供后续 Global Fusion 模块直接使用。

模块内部调用的通用算子：
    - torch.ops.fusionrepo.compute_voxel_centroid
    - torch.ops.fusionrepo.points_to_image

功能：
    voxel_points + voxel_num_points
        ↓
    compute_voxel_centroid
        ↓
    voxel_centroids
        ↓
    points_to_image
        ↓
    proj_points_2d

输入：
    voxel_points: (M, T, 3)
        每个 voxel 内的点坐标

    voxel_num_points: (M,)
        每个 voxel 内有效点数

    proj_matrix: (3, 4) 或 (B, 3, 4)
        完整投影矩阵

    batch_indices: (M,), optional
        当 proj_matrix 为 (B, 3, 4) 时，
        需要提供每个 voxel 对应的 batch 索引

输出：
    voxel_centroids: (M, 3)
        voxel 质心

    proj_points_2d: (M, 2)
        投影后的像素坐标
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class VoxelProjectionModule(nn.Module):
    """
    体素投影模块：
        voxel_points
            ↓
        voxel_centroid
            ↓
        image projection
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        voxel_points: torch.Tensor,
        voxel_num_points: torch.Tensor,
        proj_matrix: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            voxel_points: (M, T, 3)
            voxel_num_points: (M,)
            proj_matrix: (3, 4) or (B, 3, 4)
            batch_indices: (M,), optional

        Returns:
            {
                "voxel_centroids": (M, 3),
                "proj_points_2d": (M, 2)
            }
        """
        if voxel_points.dim() != 3 or voxel_points.shape[-1] != 3:
            raise ValueError("voxel_points must have shape (M, T, 3)")

        if voxel_num_points.dim() != 1:
            raise ValueError("voxel_num_points must have shape (M,)")

        if voxel_points.shape[0] != voxel_num_points.shape[0]:
            raise ValueError("voxel_points and voxel_num_points must have same first dim")

        if proj_matrix.dim() not in (2, 3):
            raise ValueError("proj_matrix must have shape (3, 4) or (B, 3, 4)")

        if proj_matrix.dim() == 2:
            if proj_matrix.shape != (3, 4):
                raise ValueError("single proj_matrix must have shape (3, 4)")
        else:
            if proj_matrix.shape[1:] != (3, 4):
                raise ValueError("batched proj_matrix must have shape (B, 3, 4)")

            if batch_indices is None:
                raise ValueError("batch_indices is required when proj_matrix is batched")

            if batch_indices.dim() != 1 or batch_indices.shape[0] != voxel_points.shape[0]:
                raise ValueError("batch_indices must have shape (M,)")

        # ------------------------------------------------
        # 1) voxel -> centroid
        # ------------------------------------------------
        voxel_centroids = torch.ops.fusionrepo.compute_voxel_centroid(
            voxel_points,
            voxel_num_points
        )  # (M, 3)

        # ------------------------------------------------
        # 2) centroid -> image
        # ------------------------------------------------
        if proj_matrix.dim() == 2:
            proj_points_2d = torch.ops.fusionrepo.points_to_image(
                voxel_centroids,
                proj_matrix,
                self.eps
            )  # (M, 2)
        else:
            # 为每个 voxel 选择对应 batch 的投影矩阵
            selected_proj = proj_matrix[batch_indices.long()]  # (M, 3, 4)

            # points_to_image 的 batched 版本要求 points 第0维与 proj_matrix 第0维一致
            voxel_centroids_batched = voxel_centroids.unsqueeze(1)   # (M, 1, 3)

            proj_points_2d = torch.ops.fusionrepo.points_to_image(
                voxel_centroids_batched,
                selected_proj,
                self.eps
            ).squeeze(1)  # (M, 2)

        return {
            "voxel_centroids": voxel_centroids,
            "proj_points_2d": proj_points_2d
        }