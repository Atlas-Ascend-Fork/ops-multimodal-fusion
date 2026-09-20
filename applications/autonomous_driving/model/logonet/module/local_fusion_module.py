# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
local_fusion_module

说明：
    该模块使用已经提取并注册的通用函数算子，组合实现 LoGoNet 的
    Local Fusion 阶段。

模块内部调用的通用算子：
    - torch.ops.fusionrepo.linear_projection
    - torch.ops.fusionrepo.pixel_to_grid_normalized
    - torch.ops.fusionrepo.feature_sampling_by_grid
    - torch.ops.fusionrepo.feature_concatenation

模块内部调用的层模块：
    - MLPFusionBlock

功能流程：
    pos_feat
        ↓
    linear_projection                 -> query

    grid_points_2d
        ↓
    pixel_to_grid_normalized          -> grid

    image_feat + grid
        ↓
    feature_sampling_by_grid          -> sampled_img_feat

    sampled_img_feat
        ↓
    linear_projection                 -> value

    query + value
        ↓
    feature_concatenation             -> concat_feat

    concat_feat
        ↓
    MLPFusionBlock                    -> fused_feat

输入：
    pos_feat: (K, G, C_p)
        位置编码特征

    image_feat: (B, C_i, H, W)
        图像特征图

    grid_points_2d: (K, G, 2)
        网格点投影到图像平面的像素坐标

    batch_indices: (K,)
        每个 ROI 对应的 batch 索引

输出：
    fused_feat: (K, G, C_out)
        局部融合后的网格级特征
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 这里改成你项目里的真实导入路径
from fusion_unit.symmetric.feature.block.mlp_fusion_block import MLPFusionBlock


class LocalFusionModule(nn.Module):
    """
    LoGoNet Local Fusion 模块：
        pos_feat + image_feat + grid_points_2d
            ↓
        局部跨模态融合
    """

    def __init__(
        self,
        pos_channels: int,
        image_channels: int,
        out_channels: int,
        mlp_hidden_channels: int | None = None
    ):
        super().__init__()

        self.pos_channels = int(pos_channels)
        self.image_channels = int(image_channels)
        self.out_channels = int(out_channels)

        if mlp_hidden_channels is None:
            mlp_hidden_channels = out_channels

        self.mlp_hidden_channels = int(mlp_hidden_channels)

        # ------------------------------------------------
        # query_fc : pos_channels -> out_channels
        # ------------------------------------------------
        self.query_weight = nn.Parameter(
            torch.empty(out_channels, pos_channels)
        )
        self.query_bias = nn.Parameter(
            torch.empty(out_channels)
        )

        # ------------------------------------------------
        # value_fc : image_channels -> out_channels
        # ------------------------------------------------
        self.value_weight = nn.Parameter(
            torch.empty(out_channels, image_channels)
        )
        self.value_bias = nn.Parameter(
            torch.empty(out_channels)
        )

        # ------------------------------------------------
        # fuse_mlp : (2*out_channels) -> out_channels
        # ------------------------------------------------
        self.fuse_mlp = MLPFusionBlock(
            in_channels=out_channels * 2,
            hidden_channels=self.mlp_hidden_channels,
            out_channels=out_channels
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_weight)
        nn.init.zeros_(self.query_bias)

        nn.init.xavier_uniform_(self.value_weight)
        nn.init.zeros_(self.value_bias)

    def forward(
        self,
        pos_feat: torch.Tensor,
        image_feat: torch.Tensor,
        grid_points_2d: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_feat: (K, G, C_p)
            image_feat: (B, C_i, H, W)
            grid_points_2d: (K, G, 2)
            batch_indices: (K,)

        Returns:
            fused_feat: (K, G, C_out)
        """
        self._validate_inputs(
            pos_feat=pos_feat,
            image_feat=image_feat,
            grid_points_2d=grid_points_2d,
            batch_indices=batch_indices
        )

        _, _, h, w = image_feat.shape

        # ------------------------------------------------
        # 1) 位置特征 -> Query
        # ------------------------------------------------
        query = torch.ops.fusionrepo.linear_projection(
            pos_feat,
            self.query_weight,
            self.query_bias
        )  # (K, G, C_out)

        # ------------------------------------------------
        # 2) 像素坐标 -> 归一化 grid
        # ------------------------------------------------
        grid = torch.ops.fusionrepo.pixel_to_grid_normalized(
            grid_points_2d,
            int(h),
            int(w)
        )  # (K, G, 2)

        # ------------------------------------------------
        # 3) 按 ROI 所属 batch，从对应图像特征图中采样
        # ------------------------------------------------
        sampled_img_feat = self._sample_image_features_by_roi(
            image_feat=image_feat,
            grid=grid,
            batch_indices=batch_indices
        )

        # ------------------------------------------------
        # 4) 图像特征 -> Value
        # ------------------------------------------------
        value = torch.ops.fusionrepo.linear_projection(
            sampled_img_feat,
            self.value_weight,
            self.value_bias
        )  # (K, G, C_out)

        # ------------------------------------------------
        # 5) 特征拼接
        # ------------------------------------------------
        concat_feat = torch.ops.fusionrepo.feature_concatenation(
            [query, value],
            -1
        )  # (K, G, 2*C_out)

        # ------------------------------------------------
        # 6) MLP 融合
        # ------------------------------------------------
        fused_feat = self.fuse_mlp(
            concat_feat
        )  # (K, G, C_out)

        return fused_feat

    def _validate_inputs(
        self,
        pos_feat: torch.Tensor,
        image_feat: torch.Tensor,
        grid_points_2d: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> None:
        """
        校验输入张量形状与通道数。
        """
        if pos_feat.dim() != 3:
            raise ValueError("pos_feat must have shape (K, G, C_p)")

        if image_feat.dim() != 4:
            raise ValueError("image_feat must have shape (B, C_i, H, W)")

        if grid_points_2d.dim() != 3 or grid_points_2d.shape[-1] != 2:
            raise ValueError("grid_points_2d must have shape (K, G, 2)")

        if batch_indices.dim() != 1:
            raise ValueError("batch_indices must have shape (K,)")

        k, g, c_p = pos_feat.shape
        b, c_i, _, _ = image_feat.shape

        if grid_points_2d.shape[0] != k or grid_points_2d.shape[1] != g:
            raise ValueError("grid_points_2d must match pos_feat on (K, G)")

        if batch_indices.shape[0] != k:
            raise ValueError("batch_indices must have shape (K,)")

        if c_p != self.pos_channels:
            raise ValueError(f"pos_feat channels must be {self.pos_channels}")

        if c_i != self.image_channels:
            raise ValueError(f"image_feat channels must be {self.image_channels}")

    def _sample_image_features_by_roi(
        self,
        image_feat: torch.Tensor,
        grid: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        按 ROI 所属 batch，从对应图像特征图中采样。

        输入:
            image_feat: (B, C_i, H, W)
            grid: (K, G, 2)
            batch_indices: (K,)
        输出:
            sampled_img_feat: (K, G, C_i)
        """
        k, g, _ = grid.shape
        b = int(image_feat.shape[0])
        c_i = int(image_feat.shape[1])

        sampled_img_feat = torch.zeros(
            (k, g, c_i),
            device=image_feat.device,
            dtype=image_feat.dtype
        )

        for roi_idx in range(k):
            batch_id = int(batch_indices[roi_idx].item())

            if batch_id < 0 or batch_id >= b:
                raise ValueError("batch_indices contains invalid batch id")

            grid_cur = grid[roi_idx].unsqueeze(0)               # (1, G, 2)
            feat_cur = image_feat[batch_id:batch_id + 1]        # (1, C_i, H, W)

            sampled_cur = torch.ops.fusionrepo.feature_sampling_by_grid(
                feat_cur,
                grid_cur
            )  # (1, C_i, G)

            sampled_cur = sampled_cur.squeeze(0).transpose(0, 1).contiguous()  # (G, C_i)
            sampled_img_feat[roi_idx] = sampled_cur

        return sampled_img_feat