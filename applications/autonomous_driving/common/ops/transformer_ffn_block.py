# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
transformer_ffn_block

说明：
    该文件实现 Transformer 风格的 FFN Block 原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.transformer_ffn_block 调用。

功能：
    对输入特征执行标准 Transformer FFN Block：

        x
        ↓
        Linear(C -> hidden_dim)
        ↓
        ReLU
        ↓
        Linear(hidden_dim -> C)
        ↓
        Residual Add
        ↓
        LayerNorm

    即：
        out = LayerNorm(x + FFN(x))

    该算子用于补齐 LoGoNet 中 FeatureAggregationLayer 在
    MultiHeadAttention 之后的前馈增强部分。

输入：
    x: (B, N, C)
        输入特征

    w1: (hidden_dim, C)
        第一层线性层权重

    b1: (hidden_dim,)
        第一层线性层偏置

    w2: (C, hidden_dim)
        第二层线性层权重

    b2: (C,)
        第二层线性层偏置

    ln_weight: (C,)
        LayerNorm 缩放参数

    ln_bias: (C,)
        LayerNorm 偏置参数

    eps: float
        LayerNorm 数值稳定项

输出：
    out: (B, N, C)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library


# ============================================================
# 1) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _transformer_ffn_block_impl(*args):
    """torch.library 算子实现: 位置参数透传 (schema 绑定, eps 缺省由 schema 处理)"""
    x, w1, b1, w2, b2, ln_weight, ln_bias = args[:7]
    eps = args[7] if len(args) > 7 else 1e-5
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, seq_len, channels)")

    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError("w1 and w2 must be 2D tensors")

    if b1.dim() != 1 or b2.dim() != 1:
        raise ValueError("b1 and b2 must be 1D tensors")

    if ln_weight.dim() != 1 or ln_bias.dim() != 1:
        raise ValueError("ln_weight and ln_bias must be 1D tensors")

    batch, seq_len, channels = x.shape

    if w1.shape[1] != channels:
        raise ValueError("w1 shape must be (hidden_dim, channels)")

    hidden_dim = w1.shape[0]

    if b1.shape[0] != hidden_dim:
        raise ValueError("b1 shape must be (hidden_dim,)")

    if w2.shape[0] != channels or w2.shape[1] != hidden_dim:
        raise ValueError("w2 shape must be (channels, hidden_dim)")

    if b2.shape[0] != channels:
        raise ValueError("b2 shape must be (channels,)")

    if ln_weight.shape[0] != channels or ln_bias.shape[0] != channels:
        raise ValueError("ln_weight and ln_bias shape must be (channels,)")

    x_flat = x.reshape(batch * seq_len, channels)

    hidden = F.linear(x_flat, w1, b1)   # (batch*seq_len, hidden_dim)
    hidden = F.relu(hidden)
    ffn_out = F.linear(hidden, w2, b2)  # (batch*seq_len, channels)

    ffn_out = ffn_out.reshape(batch, seq_len, channels)

    out = x + ffn_out
    out = F.layer_norm(
        out,
        normalized_shape=(channels,),
        weight=ln_weight,
        bias=ln_bias,
        eps=float(eps)
    )

    return out


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transformer_ffn_block", _transformer_ffn_block_impl)


# ============================================================
# 2) Meta 实现
# ============================================================
def _transformer_ffn_block_meta(*args):
    """torch.library Meta 实现: 位置参数透传 (schema 绑定 8 参)"""
    x = args[0]
    return torch.empty(
        x.shape,
        device="meta",
        dtype=x.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transformer_ffn_block", _transformer_ffn_block_meta)
