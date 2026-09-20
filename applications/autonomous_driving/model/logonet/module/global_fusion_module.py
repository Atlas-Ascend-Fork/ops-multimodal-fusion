# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
global_fusion_module

说明：
    该模块使用已经提取并注册的通用函数算子，组合实现 LoGoNet 的
    Global Fusion 阶段。

与旧版本相比：
    1. 不再要求 lidar_feat 通道数必须固定等于 lidar_channels
    2. 不再要求 image_feat 通道数必须固定等于 image_channels
    3. 模块内部自动把任意输入通道映射到统一融合维度 fused_channels
    4. 更适合直接接真实后端（如 PV-RCNN / Faster R-CNN）输出

模块内部调用的通用算子：
    - torch.ops.fusionrepo.linear_projection
    - torch.ops.fusionrepo.pixel_to_grid_normalized
    - torch.ops.fusionrepo.feature_sampling_by_grid
    - torch.ops.fusionrepo.feature_fusion_weighted_sum

功能流程：
    lidar_feat
        ↓
    lidar_adapter (可选, 自动建立)
        ↓
    linear_projection                 -> query

    proj_points
        ↓
    pixel_to_grid_normalized          -> grid

    image_feat
        ↓
    image_adapter (可选, 自动建立)
        ↓
    feature_sampling_by_grid          -> sampled_img_feat

    sampled_img_feat
        ↓
    linear_projection                 -> image_emb

    query + image_emb
        ↓
    feature_fusion_weighted_sum       -> fused_feat

输入：
    lidar_feat: (N, C_l_in)
        点云特征

    image_feat: (B, C_i_in, H, W)
        图像特征图

    proj_points: (N, 2)
        点云特征在图像上的投影像素坐标

    batch_indices: (N,), optional
        每个点云特征所属 batch 索引
        当 B > 1 时必须提供

输出：
    fused_feat: (N, fused_channels)
        全局融合后的点云特征
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalFusionModule(nn.Module):
    """
    LoGoNet Global Fusion 模块：
        lidar_feat + image_feat + proj_points
            ↓
        全局跨模态融合

    参数：
        lidar_channels:
            兼容旧接口保留。若 fused_channels=None，则默认作为融合输出通道数。
            注意：它不再表示“输入 lidar_feat 必须是多少通道”。

        image_channels:
            兼容旧接口保留。仅作默认值语义，不再强制要求 image_feat 输入通道固定。

        fused_channels:
            融合统一维度。若不传，则默认使用 lidar_channels。
    """

    def __init__(
        self,
        lidar_channels: int,
        image_channels: int,
        fused_channels: int | None = None
    ):
        super().__init__()

        self.lidar_channels = int(lidar_channels)
        self.image_channels = int(image_channels)
        self.fused_channels = int(fused_channels) if fused_channels is not None else int(lidar_channels)

        # ------------------------------------------------
        # 输入适配层（延迟初始化）
        # ------------------------------------------------
        self.lidar_input_adapter: nn.Linear | None = None
        self.image_input_adapter: nn.Conv2d | None = None

        # ------------------------------------------------
        # query_fc : fused_channels -> fused_channels
        # ------------------------------------------------
        self.query_weight = nn.Parameter(
            torch.empty(self.fused_channels, self.fused_channels)
        )
        self.query_bias = nn.Parameter(
            torch.empty(self.fused_channels)
        )

        # ------------------------------------------------
        # value_fc : fused_channels -> fused_channels
        # ------------------------------------------------
        self.value_weight = nn.Parameter(
            torch.empty(self.fused_channels, self.fused_channels)
        )
        self.value_bias = nn.Parameter(
            torch.empty(self.fused_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_weight)
        nn.init.zeros_(self.query_bias)

        nn.init.xavier_uniform_(self.value_weight)
        nn.init.zeros_(self.value_bias)

        # 延迟初始化层在首次 forward 时创建，因此这里不处理 adapter

    def forward(
        self,
        lidar_feat: torch.Tensor,
        image_feat: torch.Tensor,
        proj_points: torch.Tensor,
        batch_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            lidar_feat: (N, C_l_in)
            image_feat: (B, C_i_in, H, W)
            proj_points: (N, 2)
            batch_indices: (N,), optional

        Returns:
            fused_feat: (N, fused_channels)
        """
        n = self._validate_inputs(lidar_feat, image_feat, proj_points, batch_indices)

        b, _, h, w = image_feat.shape

        # ------------------------------------------------
        # 0) 输入通道统一映射到 fused_channels
        # ------------------------------------------------
        lidar_feat = self._adapt_lidar_feat(lidar_feat)     # (N, C_f)
        image_feat = self._adapt_image_feat(image_feat)     # (B, C_f, H, W)

        _, c_f = lidar_feat.shape
        _, c_img, _, _ = image_feat.shape

        if c_f != self.fused_channels:
            raise ValueError(f"Adapted lidar_feat channels must be {self.fused_channels}")

        if c_img != self.fused_channels:
            raise ValueError(f"Adapted image_feat channels must be {self.fused_channels}")

        # ------------------------------------------------
        # 1) 点云特征 -> Query
        # ------------------------------------------------
        query = torch.ops.fusionrepo.linear_projection(
            lidar_feat,
            self.query_weight,
            self.query_bias
        )  # (N, C_f)

        # ------------------------------------------------
        # 2) 像素坐标 -> 归一化 grid
        # ------------------------------------------------
        grid = torch.ops.fusionrepo.pixel_to_grid_normalized(
            proj_points,
            int(h),
            int(w)
        )  # (N, 2)

        # ------------------------------------------------
        # 3) 按 batch 从图像特征图中采样
        # ------------------------------------------------
        sampled_img_feat = self._sample_image_features(
            image_feat=image_feat,
            grid=grid,
            batch_indices=batch_indices,
            num_points=n
        )

        # ------------------------------------------------
        # 4) 图像特征 -> Value
        # ------------------------------------------------
        image_emb = torch.ops.fusionrepo.linear_projection(
            sampled_img_feat,
            self.value_weight,
            self.value_bias
        )  # (N, C_f)

        # ------------------------------------------------
        # 5) 融合
        # ------------------------------------------------
        fusion_inputs = torch.stack(
            [query, image_emb],
            dim=0
        )  # (2, N, C_f)

        fused_feat = torch.ops.fusionrepo.feature_fusion_weighted_sum(
            fusion_inputs,
            None
        )  # (N, C_f)

        return fused_feat

    def _validate_inputs(
        self,
        lidar_feat: torch.Tensor,
        image_feat: torch.Tensor,
        proj_points: torch.Tensor,
        batch_indices: torch.Tensor | None
    ) -> int:
        """
        校验输入张量形状，返回点数 N。
        """
        if lidar_feat.dim() != 2:
            raise ValueError("lidar_feat must have shape (N, C)")

        if image_feat.dim() != 4:
            raise ValueError("image_feat must have shape (B, C, H, W)")

        if proj_points.dim() != 2 or proj_points.shape[1] != 2:
            raise ValueError("proj_points must have shape (N, 2)")

        if lidar_feat.shape[0] != proj_points.shape[0]:
            raise ValueError("lidar_feat and proj_points must have same first dim")

        n = int(lidar_feat.shape[0])
        b, _, _, _ = image_feat.shape

        if b > 1:
            if batch_indices is None:
                raise ValueError("batch_indices is required when image_feat batch size > 1")
            if batch_indices.dim() != 1 or batch_indices.shape[0] != n:
                raise ValueError("batch_indices must have shape (N,)")

        return n

    def _sample_image_features(
        self,
        image_feat: torch.Tensor,
        grid: torch.Tensor,
        batch_indices: torch.Tensor | None,
        num_points: int
    ) -> torch.Tensor:
        """
        按 batch 从图像特征图中采样。

        输入:
            image_feat: (B, C_f, H, W)
            grid: (N, 2)
            batch_indices: (N,), optional
            num_points: N
        输出:
            sampled_img_feat: (N, C_f)
        """
        b = int(image_feat.shape[0])

        if b == 1:
            grid_for_sample = grid.unsqueeze(0)  # (1, N, 2)

            sampled_img_feat = torch.ops.fusionrepo.feature_sampling_by_grid(
                image_feat,
                grid_for_sample
            )  # (1, C_f, N)

            return sampled_img_feat.squeeze(0).transpose(0, 1).contiguous()  # (N, C_f)

        sampled_list = []

        for batch_id in range(b):
            mask = (batch_indices == batch_id)
            num_cur = int(mask.sum().item())

            if num_cur == 0:
                continue

            grid_cur = grid[mask].unsqueeze(0)              # (1, N_cur, 2)
            feat_cur = image_feat[batch_id:batch_id + 1]    # (1, C_f, H, W)

            sampled_cur = torch.ops.fusionrepo.feature_sampling_by_grid(
                feat_cur,
                grid_cur
            )  # (1, C_f, N_cur)

            sampled_cur = sampled_cur.squeeze(0).transpose(0, 1).contiguous()  # (N_cur, C_f)
            sampled_list.append((mask, sampled_cur))

        sampled_img_feat = torch.zeros(
            (num_points, self.fused_channels),
            device=image_feat.device,
            dtype=image_feat.dtype
        )

        for mask, sampled_cur in sampled_list:
            sampled_img_feat[mask] = sampled_cur

        return sampled_img_feat

    def _adapt_lidar_feat(self, lidar_feat: torch.Tensor) -> torch.Tensor:
        """
        将任意输入点云特征通道映射到 fused_channels。
        输入:
            lidar_feat: (N, C_in)
        输出:
            (N, fused_channels)
        """
        if lidar_feat.dim() != 2:
            raise ValueError(f"lidar_feat must have shape (N, C), got {tuple(lidar_feat.shape)}")

        in_channels = int(lidar_feat.shape[1])

        if in_channels == self.fused_channels:
            return lidar_feat

        if self.lidar_input_adapter is None:
            self.lidar_input_adapter = nn.Linear(
                in_features=in_channels,
                out_features=self.fused_channels,
                bias=True
            ).to(device=lidar_feat.device, dtype=lidar_feat.dtype)

        return self.lidar_input_adapter(lidar_feat)

    def _adapt_image_feat(self, image_feat: torch.Tensor) -> torch.Tensor:
        """
        将任意输入图像特征通道映射到 fused_channels。
        输入:
            image_feat: (B, C_in, H, W)
        输出:
            (B, fused_channels, H, W)
        """
        if image_feat.dim() != 4:
            raise ValueError(f"image_feat must have shape (B, C, H, W), got {tuple(image_feat.shape)}")

        in_channels = int(image_feat.shape[1])

        if in_channels == self.fused_channels:
            return image_feat

        if self.image_input_adapter is None:
            self.image_input_adapter = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.fused_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ).to(device=image_feat.device, dtype=image_feat.dtype)

        return self.image_input_adapter(image_feat)