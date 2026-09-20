# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""LoGoNet RoI refine head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class _LocalRoIRefineHead(nn.Module):
    """A stronger RoI head closer to the original LoGoHead FC design."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        box_dim: int = 7,
        hidden_channels: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()

        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, box_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        last_reg = self.reg_head[-1]
        nn.init.normal_(last_reg.weight, mean=0.0, std=0.001)

    def forward(self, roi_feat: torch.Tensor):
        pooled_feat = roi_feat.mean(dim=1)
        shared_feat = self.shared_fc(pooled_feat)
        cls_score = self.cls_head(shared_feat)
        reg_offset = self.reg_head(shared_feat)
        return cls_score, reg_offset


class RoIRefineModule(nn.Module):
    """RoI refine wrapper used by the framework."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        box_dim: int = 7
    ):
        super().__init__()

        self.roi_refine_head = _LocalRoIRefineHead(
            in_channels=in_channels,
            num_classes=num_classes,
            box_dim=box_dim
        )

    def forward(
        self,
        roi_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cls_score, reg_offset = self.roi_refine_head(roi_feat)

        return {
            "cls_score": cls_score,
            "reg_offset": reg_offset
        }
