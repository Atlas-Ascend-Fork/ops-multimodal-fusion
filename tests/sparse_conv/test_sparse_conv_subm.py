#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 融合 SubM 稀疏卷积对拍: sparse_conv_subm (单次调用) vs 稠密 conv3d 金标准。
# mmcv python hash 语义 = 稠密 conv3d 在 active 位置的取值 (SubM)。
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/test_sparse_conv_subm.py -v

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from sparse_common import (DEVICE, OPS, Case, make_case, to_dense,
                           dense_rows, w_dense)
import ops_multimodal_fusion  # noqa: F401


@dataclass
class SubmSpec:
    """sparse_conv_subm 调用参数 (空间尺寸 + 卷积核, SubM 无 stride/pad)"""
    d: int
    h: int
    w: int
    kz: int = 3
    kh: int = 3
    kw: int = 3


def run_op(idx, feat, weight, bias, spec):
    return OPS.sparse_conv_subm(
        feat, idx, spec.d, spec.h, spec.w, weight, bias,
        spec.kz, spec.kh, spec.kw, 1, 1, 1)[0]


def test_parity_3x3x3_with_bias():
    torch.manual_seed(0)
    kz = kh = kw = 3
    ci, co = 4, 6
    case = Case(40, 5, 5, 5, 1)
    idx = make_case(case, seed=0)
    n = idx.size(0)
    feat = torch.randn(n, ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    bias = torch.randn(co, device=DEVICE)
    out = run_op(idx, feat, w, bias, SubmSpec(5, 5, 5))
    ref = F.conv3d(to_dense(idx, feat, 5, 5, 5), w_dense(w, kz, kh, kw),
                   padding=1) + bias.view(1, -1, 1, 1, 1)
    got = dense_rows(ref, idx)
    assert torch.allclose(out, got, atol=1e-2, rtol=1e-2), \
        f"parity: {(out - got).abs().max()}"


def test_parity_unaligned_no_bias():
    # 奇数形状: N=37, C_in=5, C_out=7 (K=135 非 16 对齐 → pad 路径)
    torch.manual_seed(1)
    kz = kh = kw = 3
    ci, co = 5, 7
    case = Case(37, 4, 6, 7, 1)
    idx = make_case(case, seed=1)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    out = run_op(idx, feat, w, None, SubmSpec(4, 6, 7))
    ref = F.conv3d(to_dense(idx, feat, 4, 6, 7), w_dense(w, kz, kh, kw),
                   padding=1)
    got = dense_rows(ref, idx)
    assert torch.allclose(out, got, atol=1e-2, rtol=1e-2), \
        f"unaligned: {(out - got).abs().max()}"


def test_parity_batch2():
    torch.manual_seed(2)
    kz = kh = kw = 3
    ci, co = 3, 4
    case = Case(30, 3, 4, 4, 2)
    idx = make_case(case, seed=2)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    bias = torch.randn(co, device=DEVICE)
    out = run_op(idx, feat, w, bias, SubmSpec(3, 4, 4))
    ref = F.conv3d(to_dense(idx, feat, 3, 4, 4), w_dense(w, kz, kh, kw),
                   padding=1) + bias.view(1, -1, 1, 1, 1)
    got = dense_rows(ref, idx)
    assert torch.allclose(out, got, atol=1e-2, rtol=1e-2), \
        f"batch2: {(out - got).abs().max()}"


def test_parity_kernel_5x1x1():
    torch.manual_seed(3)
    kz, kh, kw = 5, 1, 1
    ci, co = 4, 3
    case = Case(20, 7, 4, 4, 1)
    idx = make_case(case, seed=3)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    out = run_op(idx, feat, w, None, SubmSpec(7, 4, 4, kz=5, kh=1, kw=1))
    ref = F.conv3d(to_dense(idx, feat, 7, 4, 4), w_dense(w, kz, kh, kw),
                   padding=(2, 0, 0))
    got = dense_rows(ref, idx)
    assert torch.allclose(out, got, atol=1e-2, rtol=1e-2), \
        f"5x1x1: {(out - got).abs().max()}"


def test_cache_token_consistency():
    # 同 token 两次调用 (第 2 次命中缓存) 结果一致
    torch.manual_seed(4)
    kz = kh = kw = 3
    ci, co = 4, 5
    case = Case(25, 4, 4, 4, 1)
    idx = make_case(case, seed=4)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    w2 = torch.randn(kz * kh * kw, ci, co, device=DEVICE)   # 第 2 层不同权重
    out1 = OPS.sparse_conv_subm(feat, idx, 4, 4, 4, w, None,
                                3, 3, 3, 1, 1, 1, 77)[0]
    out2 = OPS.sparse_conv_subm(feat, idx, 4, 4, 4, w2, None,
                                3, 3, 3, 1, 1, 1, 77)[0]   # 命中 rulebook 缓存
    ref2 = F.conv3d(to_dense(idx, feat, 4, 4, 4), w_dense(w2, kz, kh, kw),
                    padding=1)
    got2 = dense_rows(ref2, idx)
    assert torch.allclose(out2, got2, atol=1e-2, rtol=1e-2), \
        f"cached 2nd layer: {(out2 - got2).abs().max()}"
    # 换 token (新帧) 仍正确
    out3 = OPS.sparse_conv_subm(feat, idx, 4, 4, 4, w, None,
                                3, 3, 3, 1, 1, 1, 78)[0]
    assert torch.allclose(out1, out3, atol=1e-3, rtol=1e-3)


def test_zero_rows():
    idx = torch.zeros(0, 4, dtype=torch.int32, device=DEVICE)
    feat = torch.zeros(0, 8, device=DEVICE)
    w = torch.randn(27, 8, 8, device=DEVICE)
    out_f, out_i = OPS.sparse_conv_subm(feat, idx, 4, 4, 4, w, None)
    assert out_f.shape == (0, 8) and out_i.shape == (0, 4)


def test_fp16_features():
    torch.manual_seed(5)
    kz = kh = kw = 3
    ci, co = 4, 6
    case = Case(25, 5, 5, 5, 1)
    idx = make_case(case, seed=5)
    feat = torch.randn(idx.size(0), ci, device=DEVICE).half()
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    bias = torch.randn(co, device=DEVICE)
    out = run_op(idx, feat, w, bias, SubmSpec(5, 5, 5))
    assert out.dtype == torch.float16
    ref = F.conv3d(to_dense(idx, feat.float(), 5, 5, 5),
                   w_dense(w, kz, kh, kw), padding=1) + bias.view(1, -1, 1, 1, 1)
    got = dense_rows(ref, idx)
    assert torch.allclose(out.float(), got, atol=2e-2, rtol=2e-2), \
        f"fp16: {(out.float() - got).abs().max()}"
