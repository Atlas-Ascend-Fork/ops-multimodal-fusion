# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
position_encoding_module

说明：
    该模块封装 LoGoNet 的位置编码阶段，
    framework 中只调用本模块，
    本模块内部直接调用已有的 PositionEncodingLayer，
    不重复实现其内部逻辑。

功能流程：
    grid_centers + roi_centers + grid_point_counts
        ↓
    PositionEncodingLayer
        ↓
    pos_feat
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# 这里改成你项目里的真实导入路径
from common.position_encoding_layer import PositionEncodingLayer


class PositionEncodingModule(nn.Module):
    """
    Position Encoding 模块：
        framework 只调用这个模块，
        模块内部调用已有 PositionEncodingLayer
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.position_encoding_layer = PositionEncodingLayer(
            out_channels=out_channels
        )

    def forward(
        self,
        grid_centers: torch.Tensor,
        roi_centers: torch.Tensor,
        grid_point_counts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pos_feat = self.position_encoding_layer(
            grid_centers=grid_centers,
            roi_centers=roi_centers,
            grid_point_counts=grid_point_counts
        )

        return {
            "pos_feat": pos_feat
        }