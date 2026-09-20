# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
feature_aggregation_module

说明：
    该模块使用已经提取并注册的通用函数算子，组合实现 LoGoNet 的
    Feature Dynamic Aggregation（FDA）阶段。

模块内部调用的通用算子：
    - torch.ops.fusionrepo.multihead_attention_fusion
    - torch.ops.fusionrepo.transformer_ffn_block

功能流程：
    fused_feat
        ↓
    multihead_attention_fusion
        ↓
    residual + layer_norm
        ↓
    transformer_ffn_block
        ↓
    out

输入：
    fused_feat: (K, G, C)
        网格级融合特征

输出：
    out: (K, G, C)
        聚合后的网格级特征
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAggregationModule(nn.Module):
    """
    LoGoNet FDA 模块：
        fused_feat
            ↓
        multihead attention
            ↓
        residual + norm
            ↓
        transformer ffn block
            ↓
        out
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        norm_eps: float = 1e-5
    ):
        super().__init__()

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.ffn_ratio = int(ffn_ratio)
        self.norm_eps = float(norm_eps)

        hidden_dim = self.channels * self.ffn_ratio

        # ------------------------------------------------
        # 第一层 LayerNorm（attention 后）
        # ------------------------------------------------
        self.norm1_weight = nn.Parameter(
            torch.ones(self.channels)
        )
        self.norm1_bias = nn.Parameter(
            torch.zeros(self.channels)
        )

        self.ffn_w1 = nn.Parameter(
            torch.empty(hidden_dim, self.channels)
        )
        self.ffn_b1 = nn.Parameter(
            torch.empty(hidden_dim)
        )

        self.ffn_w2 = nn.Parameter(
            torch.empty(self.channels, hidden_dim)
        )
        self.ffn_b2 = nn.Parameter(
            torch.empty(self.channels)
        )

        self.norm2_weight = nn.Parameter(
            torch.ones(self.channels)
        )
        self.norm2_bias = nn.Parameter(
            torch.zeros(self.channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ffn_w1)
        nn.init.zeros_(self.ffn_b1)

        nn.init.xavier_uniform_(self.ffn_w2)
        nn.init.zeros_(self.ffn_b2)

    def forward(
        self,
        fused_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fused_feat: (K, G, C)

        Returns:
            out: (K, G, C)
        """
        if fused_feat.dim() != 3:
            raise ValueError("fused_feat must have shape (K, G, C)")

        k, g, c = fused_feat.shape

        if c != self.channels:
            raise ValueError(f"fused_feat channels must be {self.channels}")

        if self.channels % self.num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        # ------------------------------------------------
        # 1) Self-Attention
        # ------------------------------------------------
        attn_out = torch.ops.fusionrepo.multihead_attention_fusion(
            fused_feat,
            fused_feat,
            fused_feat,
            int(self.num_heads)
        )  # (K, G, C)

        # ------------------------------------------------
        # 2) Residual + LayerNorm
        # ------------------------------------------------
        x = F.layer_norm(
            fused_feat + attn_out,
            normalized_shape=(self.channels,),
            weight=self.norm1_weight,
            bias=self.norm1_bias,
            eps=self.norm_eps
        )  # (K, G, C)

        # ------------------------------------------------
        # 3) Transformer FFN Block
        # ------------------------------------------------
        out = torch.ops.fusionrepo.transformer_ffn_block(
            x,
            self.ffn_w1,
            self.ffn_b1,
            self.ffn_w2,
            self.ffn_b2,
            self.norm2_weight,
            self.norm2_bias,
            float(self.norm_eps)
        )  # (K, G, C)

        return out