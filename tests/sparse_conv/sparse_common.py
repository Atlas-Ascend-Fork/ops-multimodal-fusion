# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 稀疏卷积测试公共工具: 用例生成 / 稠密展开 / 权重重排。
# 由 tests/sparse_conv/ 下各对拍测试共享 (pytest 将本目录注入 sys.path)。

from collections import namedtuple

import torch

DEVICE = "npu"
OPS = torch.ops.ops_multimodal_fusion

Case = namedtuple("Case", ["n", "d", "h", "w", "batch"])


def make_case(case, seed):
    """按 Case 随机生成互异体素坐标 (b, z, y, x) int32 (N, 4)"""
    torch.manual_seed(seed)
    idx = torch.stack([
        torch.randint(0, case.batch, (case.n,), device=DEVICE),
        torch.randint(0, case.d, (case.n,), device=DEVICE),
        torch.randint(0, case.h, (case.n,), device=DEVICE),
        torch.randint(0, case.w, (case.n,), device=DEVICE)], 1).unique(dim=0)
    return idx.to(torch.int32).contiguous()


def to_dense(indices, features, d, h, w):
    """稀疏 → 稠密 (B, C, D, H, W), 稠密 conv 参考的公共前处理"""
    batch = int(indices[:, 0].max()) + 1
    dense = torch.zeros(batch, features.size(1), d, h, w, device=DEVICE)
    # scatter 展开: permute 成 (B,D,H,W,C) 格子序, flat = 格子线性号
    flat = (indices[:, 0].long() * (d * h * w) +
            indices[:, 1].long() * (h * w) +
            indices[:, 2].long() * w + indices[:, 3].long())
    dense_g = dense.permute(0, 2, 3, 4, 1).reshape(-1, features.size(1))
    dense_g.index_copy_(0, flat, features)
    dense[:] = dense_g.reshape(batch, d, h, w, features.size(1)) \
                   .permute(0, 4, 1, 2, 3)
    return dense


def dense_rows(dense, indices4):
    """从稠密体按 (b, z, y, x) 行取值 — 与 out_indices 逐行对齐"""
    return dense[indices4[:, 0].long(), :,
                 indices4[:, 1].long(), indices4[:, 2].long(),
                 indices4[:, 3].long()]


def w_dense(weight, kz, kh, kw):
    """(kvol, Cin, Cout) → conv3d 权重 (Cout, Cin, kz, kh, kw), tap 序对齐"""
    kvol = kz * kh * kw
    return (weight.reshape(kz, kh, kw, *weight.shape[1:])
            .permute(4, 3, 0, 1, 2).contiguous())
