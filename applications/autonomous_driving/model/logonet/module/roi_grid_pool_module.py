# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
roi_grid_pool_module

说明：
    该模块使用已经提取并注册的通用函数算子，组合实现 LoGoNet 中
    ROI 网格生成 + ROI 内点云特征收集 + 池化 这一阶段逻辑。

模块内部调用的通用算子：
    - torch.ops.fusionrepo.generate_roi_grid_points
    - torch.ops.fusionrepo.gather_roi_features
    - torch.ops.fusionrepo.reduce_pool

功能流程：
    rois
        ↓
    generate_roi_grid_points
        ↓
    grid_points

    rois + point_coords + point_features
        ↓
    gather_roi_features
        ↓
    gathered_features, valid_counts

    gathered_features
        ↓
    reduce_pool
        ↓
    pooled_features

同时补充输出：
    - roi_centers
    - grid_point_counts

输入：
    rois: (K, 7)
        ROI 参数，格式为：
        (x, y, z, dx, dy, dz, yaw)

    point_coords: (N, 3)
        点 / voxel 的三维坐标

    point_features: (N, C)
        点 / voxel 的特征

输出：
    {
        "grid_points": (K, G, 3),
        "roi_centers": (K, 3),
        "grid_point_counts": (K, G),
        "gathered_features": (K, N, C),
        "valid_counts": (K,),
        "pooled_features": (K, C)
    }
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RoIGridPoolModule(nn.Module):
    """
    ROI 网格池化模块：
        ROI
          ↓
        生成网格点
          ↓
        收集 ROI 内点云特征
          ↓
        池化得到 ROI 特征
    """

    def __init__(
        self,
        grid_size: int = 6,
        pool_mode: str = "mean"
    ):
        super().__init__()

        if pool_mode not in ("max", "sum", "mean"):
            raise ValueError("pool_mode must be one of: 'max', 'sum', 'mean'")

        self.grid_size = int(grid_size)
        self.pool_mode = str(pool_mode)

    def forward(
        self,
        rois: torch.Tensor,
        point_coords: torch.Tensor,
        point_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rois: (K, 7)
            point_coords: (N, 3)
            point_features: (N, C)

        Returns:
            {
                "grid_points": (K, G, 3),
                "roi_centers": (K, 3),
                "grid_point_counts": (K, G),
                "gathered_features": (K, N, C),
                "valid_counts": (K,),
                "pooled_features": (K, C)
            }
        """
        if rois.dim() != 2 or rois.shape[1] != 7:
            raise ValueError("rois must have shape (K, 7)")

        if point_coords.dim() != 2 or point_coords.shape[1] != 3:
            raise ValueError("point_coords must have shape (N, 3)")

        if point_features.dim() != 2:
            raise ValueError("point_features must have shape (N, C)")

        if point_coords.shape[0] != point_features.shape[0]:
            raise ValueError("point_coords and point_features must have same first dim")

        # ------------------------------------------------
        # 1) 生成 ROI 网格点 (向量化实现)
        # ------------------------------------------------
        grid_points = torch.ops.fusionrepo.generate_roi_grid_points(
            rois,
            int(self.grid_size)
        )  # (K, G, 3)

        # ------------------------------------------------
        # 2) ROI 中心
        # ------------------------------------------------
        roi_centers = rois[:, 0:3]  # (K, 3)

        # ------------------------------------------------
        # 3+4+5) 计数 + 收集 + 池化
        # mean/sum: 几何算一次, inside mask 同时用于 grid_point_counts 与
        #   池化 GEMM (mask @ features), 不物化 (K,N,C) (下游不使用)
        # max: 保留原 gather op (语义需要全量特征)
        # ------------------------------------------------
        if self.pool_mode == "max":
            grid_point_counts, gathered_features, valid_counts, pooled_features = self._pool_max(
                grid_points=grid_points,
                rois=rois,
                point_coords=point_coords,
                point_features=point_features
            )
        else:
            grid_point_counts, valid_counts, pooled_features = self._pool_mean_or_sum(
                rois=rois,
                point_coords=point_coords,
                point_features=point_features
            )
            gathered_features = None

        return {
            "grid_points": grid_points,
            "roi_centers": roi_centers,
            "grid_point_counts": grid_point_counts,
            "gathered_features": gathered_features,
            "valid_counts": valid_counts,
            "pooled_features": pooled_features
        }

    def _compute_local_geometry(
        self,
        point_coords: torch.Tensor,
        rois: torch.Tensor
    ):
        """
        共享几何计算: 所有 ROI 同时处理, 得到点在 ROI 局部坐标系的 (x, y)。
        直接 2D 旋转公式替代 (K,N,3)@(K,3,3) bmm (旋转矩阵含 2 列零元素)。

        返回:
            rel: (K, N, 3) 相对 ROI 中心的坐标
            xl: (K, N) 局部 x = cos*rx + sin*ry
            yl: (K, N) 局部 y = -sin*rx + cos*ry
            dims: (K, 3) ROI 尺寸
        """
        centers = rois[:, 0:3]   # (K, 3)
        dims = rois[:, 3:6]      # (K, 3)
        yaws = rois[:, 6]        # (K,)

        rel = point_coords.unsqueeze(0) - centers.unsqueeze(1)  # (K, N, 3)

        cos_yaw = torch.cos(yaws).unsqueeze(-1)  # (K, 1)
        sin_yaw = torch.sin(yaws).unsqueeze(-1)  # (K, 1)
        rx = rel[..., 0]                         # (K, N)
        ry = rel[..., 1]                         # (K, N)
        xl = rx * cos_yaw + ry * sin_yaw         # (K, N)
        yl = -rx * sin_yaw + ry * cos_yaw        # (K, N)

        return rel, xl, yl, dims

    def _compute_grid_point_counts_from_geometry(
        self,
        xl: torch.Tensor,
        geom: tuple,
    ) -> torch.Tensor:
        """
        统计每个 ROI 每个网格 cell 的点数 (基于共享几何结果, 避免重复计算)。
        分箱公式与旧 bmm 实现逐位一致: norm=(local+dims/2)/dims, clamp, *gs, trunc。
        """
        yl, rz, dims, inside = geom
        device = xl.device
        dtype = xl.dtype
        num_rois, num_points = xl.shape
        num_cells = self.grid_size ** 3
        gs = self.grid_size

        # 1) 局部坐标 (K,N,3): [xl, yl, rz] — 与旧 bmm(rel, rot) 数学等价
        rel_local = torch.stack([xl, yl, rz], dim=-1)

        # 2) 归一化到 [0,1) 再映射到 cell (与旧实现相同算子顺序)
        dim_safe = dims.clamp(min=1e-6).unsqueeze(1)          # (K, 1, 3)
        norm = (rel_local + (dims / 2.0).unsqueeze(1)) / dim_safe
        norm = norm.clamp(0.0, 1.0 - 1e-6)
        g = (norm * gs).long()                                # (K, N, 3), 值域 [0, gs-1]

        # 3) 展平 cell 索引
        flat_idx = g[..., 0] * (gs * gs) + g[..., 1] * gs + g[..., 2]  # (K, N)

        # 4) 只统计 inside 的点 (outside 落到第 G 槽, 之后丢弃)
        flat_idx_masked = torch.where(inside, flat_idx, torch.full_like(flat_idx, num_cells))

        # 5) 直方图统计
        # 不用 scatter_add_: NPU aclnnScatterAdd 对重复索引串行累加,
        #   outside 点 (约 97%) 全部落到兜底槽 → 严重倾斜 → 8.4ms。
        # bincount (原子/分桶实现) 对倾斜索引 0.18ms, 语义等价。
        batch_offset = torch.arange(num_rois, device=device).unsqueeze(1) * (num_cells + 1)  # (K, 1)
        global_idx = flat_idx_masked + batch_offset                           # (K, N)
        counts = torch.bincount(global_idx.reshape(-1), minlength=num_rois * (num_cells + 1))
        grid_point_counts = counts.reshape(num_rois, num_cells + 1)[:, :num_cells].to(dtype)

        return grid_point_counts

    def _compute_grid_point_counts(
        self,
        grid_points: torch.Tensor,
        point_coords: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """独立入口: 自行计算几何后分箱 (兼容原签名)。"""
        rel, xl, yl, dims = self._compute_local_geometry(point_coords, rois)
        rz = rel[..., 2]
        half_d = dims / 2.0
        inside = (
            (xl.abs() <= half_d[:, 0].unsqueeze(1)) &
            (yl.abs() <= half_d[:, 1].unsqueeze(1)) &
            (rz.abs() <= half_d[:, 2].unsqueeze(1))
        )
        return self._compute_grid_point_counts_from_geometry(
            xl, (yl, rz, dims, inside))

    def _pool_max(
        self,
        grid_points: torch.Tensor,
        rois: torch.Tensor,
        point_coords: torch.Tensor,
        point_features: torch.Tensor
    ):
        """
        max 池化: 保留原 gather op (语义需要全量特征)。

        返回:
            grid_point_counts: (K, G)
            gathered_features: (K, N, C)
            valid_counts: (K,)
            pooled_features: (K, C)
        """
        grid_point_counts = self._compute_grid_point_counts(
            grid_points=grid_points,
            point_coords=point_coords,
            rois=rois
        )  # (K, G)

        gathered_features, valid_counts = torch.ops.fusionrepo.gather_roi_features(
            rois,
            point_coords,
            point_features
        )  # (K, N, C), (K,)

        pooled_features = gathered_features.amax(dim=1)

        return grid_point_counts, gathered_features, valid_counts, pooled_features

    def _pool_mean_or_sum(
        self,
        rois: torch.Tensor,
        point_coords: torch.Tensor,
        point_features: torch.Tensor
    ):
        """
        mean/sum 池化: 共享几何, rel/inside 只算一次, 同时用于 counts + 池化,
        inside mask 同时用于 grid_point_counts 与池化 GEMM (mask @ features),
        不物化 (K,N,C) (下游不使用)。

        返回:
            grid_point_counts: (K, G)
            valid_counts: (K,)
            pooled_features: (K, C)
        """
        rel, xl, yl, dims = self._compute_local_geometry(point_coords, rois)
        rz = rel[..., 2]  # (K, N)
        half_d = dims / 2.0
        inside = (
            (xl.abs() <= half_d[:, 0].unsqueeze(1)) &
            (yl.abs() <= half_d[:, 1].unsqueeze(1)) &
            (rz.abs() <= half_d[:, 2].unsqueeze(1))
        )  # (K, N)

        grid_point_counts = self._compute_grid_point_counts_from_geometry(
            xl, (yl, rz, dims, inside)
        )

        valid_counts = inside.sum(dim=1)  # (K,)
        mask_f = inside.to(point_features.dtype)  # (K, N)
        pooled = torch.mm(mask_f, point_features)  # (K, C)
        if self.pool_mode == "mean":
            pooled_features = pooled / valid_counts.clamp(min=1).to(point_features.dtype).unsqueeze(-1)
        else:
            pooled_features = pooled

        return grid_point_counts, valid_counts, pooled_features