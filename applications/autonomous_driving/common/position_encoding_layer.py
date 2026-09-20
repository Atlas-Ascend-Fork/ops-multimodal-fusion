# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch
import torch.nn as nn


class PositionEncodingLayer(nn.Module):
    """
    PositionEncodingLayer 模块（LoGoNet 局部融合阶段的几何位置编码算子）。

    该模块用于对候选框（RoI）内部的局部网格进行几何信息编码，
    将每个网格的**相对空间位置**以及**点云分布密度信息**映射到
    可学习的特征空间中，从而为后续局部跨模态融合提供结构先验。

    本模块属于 **可学习算子**，其参数通过训练自动学习，
    但其输入仅依赖几何与统计信息，不涉及图像或语义特征。

    ---
    ### 结构说明（按顺序）
    1. **输入拼接**
       - 网格中心相对位置 (dx, dy, dz)
       - 网格内点数量的对数值 log(N + 1)
    2. **Linear → ReLU**
       - 初步映射几何与密度信息
    3. **Linear → ReLU**
       - 提升几何特征表达能力
    4. **Linear**
       - 输出位置编码特征

    ---
    ### 输入
    Args:
        grid_centers (torch.Tensor):
            RoI 内部网格中心点坐标，shape 为 **(K, G, 3)**，
            K 为候选框数量，G 为每个候选框内的网格数。

        roi_centers (torch.Tensor):
            候选框中心坐标，shape 为 **(K, 3)**。

        grid_point_counts (torch.Tensor):
            每个网格内的点数量，shape 为 **(K, G)**。

    ---
    ### 输出
    Returns:
        torch.Tensor:
            网格级几何位置编码特征，shape 为 **(K, G, C)**，
            C 为位置编码特征维度。

    ---
    ### 模块用途
    - 为 RoI 内部局部网格提供结构与密度先验
    - 引导后续局部跨模态融合关注目标内部有效区域
    - 属于 LoGoNet 的 LocalFusion（LoF）阶段关键算子
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(
        self,
        grid_centers: torch.Tensor,
        roi_centers: torch.Tensor,
        grid_point_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播过程。
        """
        # 计算网格中心相对于 RoI 中心的偏移
        rel_pos = grid_centers - roi_centers.unsqueeze(1)  # (K, G, 3)

        # 对网格内点数量取对数，增强数值稳定性
        density = torch.log(grid_point_counts.float() + 1.0).unsqueeze(-1)  # (K, G, 1)

        # 拼接几何与密度信息
        pos_feat = torch.cat([rel_pos, density], dim=-1)  # (K, G, 4)

        # 几何位置编码
        encoded_feat = self.mlp(pos_feat)

        return encoded_feat
