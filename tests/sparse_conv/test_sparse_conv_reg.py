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

# 融合 Regular 稀疏卷积对拍: sparse_conv_reg (单次调用) vs 稠密 conv3d 金标准。
# stride=2/pad=1 下采样: out_indices 升序 = 稠密 conv 在有效输出格的取值。
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/test_sparse_conv_reg.py -v

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from sparse_common import (DEVICE, OPS, Case, make_case, to_dense, w_dense)
import ops_multimodal_fusion  # noqa: F401


@dataclass
class RegSpec:
    """sparse_conv_reg 调用参数 (空间/卷积核/stride/padding)"""
    di: int
    hi: int
    wi: int
    kz: int = 3
    kh: int = 3
    kw: int = 3
    st: tuple = (2, 2, 2)
    pd: tuple = (1, 1, 1)


def _out_shape(spec):
    """下采样输出空间尺寸"""
    do = (spec.di + 2 * spec.pd[0] - spec.kz) // spec.st[0] + 1
    ho = (spec.hi + 2 * spec.pd[1] - spec.kh) // spec.st[1] + 1
    wo = (spec.wi + 2 * spec.pd[2] - spec.kw) // spec.st[2] + 1
    return do, ho, wo


def _dense_reference(idx, feat, weight, bias, spec):
    """金标准: 稠密 conv3d (stride/pad)"""
    ref = F.conv3d(to_dense(idx, feat, spec.di, spec.hi, spec.wi),
                   w_dense(weight, spec.kz, spec.kh, spec.kw),
                   stride=spec.st, padding=spec.pd)
    if bias is not None:
        ref = ref + bias.view(1, -1, 1, 1, 1)
    return ref


def check_reg(idx, feat, weight, bias, spec):
    do, ho, wo = _out_shape(spec)
    out_f, out_i = OPS.sparse_conv_reg(
        feat, idx, do, ho, wo, weight, bias,
        spec.kz, spec.kh, spec.kw,
        spec.st[0], spec.st[1], spec.st[2],
        spec.pd[0], spec.pd[1], spec.pd[2], 1, 1, 1)
    ref = _dense_reference(idx, feat, weight, bias, spec)
    _check_positional(out_f, out_i, ref, spec)
    _check_index_set(idx, out_i, spec)


def _check_positional(out_f, out_i, ref, spec):
    # ① out_indices 逐位正确: 先转 (格数, C) 布局再按格索引 (避免 advanced
    #    indexing 把 ':' 挤到末尾导致 (C, M) 转置陷阱)
    do, ho, wo = _out_shape(spec)
    ref_rows = ref.permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
    lin = (out_i[:, 0].long() * (do * ho * wo) + out_i[:, 1].long() * (ho * wo)
           + out_i[:, 2].long() * wo + out_i[:, 3].long())
    got = ref_rows[lin]
    diff = (out_f - got).abs().max(dim=1).values
    assert (diff <= 1e-2 + 1e-2 * got.abs().max(dim=1).values).all(), \
        f"positional: max diff {diff.max().item()}"


def _check_index_set(idx, out_i, spec):
    # ② out_indices 集合完整: 与 g_flat 公式 (非 bias 噪声) 推出的有效输出格一致
    do, ho, wo = _out_shape(spec)
    coords = idx[:, 1:4].long()
    kvol = spec.kz * spec.kh * spec.kw
    ar = torch.arange(kvol, device=DEVICE)
    kd = (ar // (spec.kh * spec.kw)).reshape(1, -1)
    khh = ((ar // spec.kw) % spec.kh).reshape(1, -1)
    kww = (ar % spec.kw).reshape(1, -1)
    oz_num = coords[:, 0:1] + spec.pd[0] - kd
    oy_num = coords[:, 1:2] + spec.pd[1] - khh
    ox_num = coords[:, 2:3] + spec.pd[2] - kww
    div = ((oz_num % spec.st[0] == 0) & (oy_num % spec.st[1] == 0) &
           (ox_num % spec.st[2] == 0) &
           (oz_num >= 0) & (oy_num >= 0) & (ox_num >= 0))
    oz = torch.where(div, oz_num // spec.st[0], torch.zeros_like(oz_num))
    oy = torch.where(div, oy_num // spec.st[1], torch.zeros_like(oy_num))
    ox = torch.where(div, ox_num // spec.st[2], torch.zeros_like(ox_num))
    inb = div & (oz < do) & (oy < ho) & (ox < wo)
    flats = (idx[:, 0].long().unsqueeze(1) * (do * ho * wo) +
             oz * (ho * wo) + oy * wo + ox)[inb].unique()
    op_flats = (out_i[:, 0].long() * (do * ho * wo) +
                out_i[:, 1].long() * (ho * wo) +
                out_i[:, 2].long() * wo + out_i[:, 3].long())
    assert torch.equal(flats.sort().values, op_flats.sort().values), \
        f"out_indices set mismatch: {flats.numel()} vs {op_flats.numel()}"


def test_parity_stride2_bias():
    torch.manual_seed(0)
    case = Case(40, 5, 5, 5, 1)
    idx = make_case(case, seed=0)
    feat = torch.randn(idx.size(0), 4, device=DEVICE)
    w = torch.randn(27, 4, 6, device=DEVICE)
    bias = torch.randn(6, device=DEVICE)
    check_reg(idx, feat, w, bias, RegSpec(5, 5, 5))


def test_parity_batch2():
    torch.manual_seed(1)
    case = Case(50, 5, 6, 6, 2)
    idx = make_case(case, seed=1)
    feat = torch.randn(idx.size(0), 3, device=DEVICE)
    w = torch.randn(27, 3, 5, device=DEVICE)
    bias = torch.randn(5, device=DEVICE)
    check_reg(idx, feat, w, bias, RegSpec(5, 6, 6))


def test_parity_kernel_5x1x1():
    # 5×1×1 核 (非 3³), z 向下采样
    torch.manual_seed(2)
    case = Case(30, 9, 4, 4, 1)
    idx = make_case(case, seed=2)
    feat = torch.randn(idx.size(0), 4, device=DEVICE)
    w = torch.randn(5, 4, 3, device=DEVICE)
    spec = RegSpec(9, 4, 4, kz=5, kh=1, kw=1, st=(2, 1, 1), pd=(2, 0, 0))
    check_reg(idx, feat, w, None, spec)


def test_parity_unaligned_channels():
    # 奇数通道 C_in=5/C_out=7 (对拍 fp32 路径)
    torch.manual_seed(3)
    case = Case(25, 5, 5, 5, 1)
    idx = make_case(case, seed=3)
    feat = torch.randn(idx.size(0), 5, device=DEVICE)
    w = torch.randn(27, 5, 7, device=DEVICE)
    bias = torch.randn(7, device=DEVICE)
    check_reg(idx, feat, w, bias, RegSpec(5, 5, 5))


def test_fp16_features():
    torch.manual_seed(4)
    case = Case(40, 5, 5, 5, 1)
    idx = make_case(case, seed=4)
    feat = torch.randn(idx.size(0), 4, device=DEVICE).half()
    w = torch.randn(27, 4, 6, device=DEVICE)
    bias = torch.randn(6, device=DEVICE)
    out_f, out_i = OPS.sparse_conv_reg(
        feat, idx, 3, 3, 3, w, bias, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1)
    assert out_f.dtype == torch.float16
    ref = F.conv3d(to_dense(idx, feat.float(), 5, 5, 5),
                   w_dense(w, 3, 3, 3), stride=2, padding=1) \
        + bias.view(1, -1, 1, 1, 1)
    ref_rows = ref.permute(0, 2, 3, 4, 1).reshape(-1, 6)
    lin = (out_i[:, 0].long() * 27 + out_i[:, 1].long() * 9
           + out_i[:, 2].long() * 3 + out_i[:, 3].long())
    got = ref_rows[lin]
    assert torch.allclose(out_f.float(), got, atol=2e-2, rtol=2e-2), \
        f"fp16: {(out_f.float() - got).abs().max()}"


def test_zero_rows():
    idx = torch.zeros(0, 4, dtype=torch.int32, device=DEVICE)
    feat = torch.zeros(0, 8, device=DEVICE)
    w = torch.randn(27, 8, 8, device=DEVICE)
    out_f, out_i = OPS.sparse_conv_reg(feat, idx, 3, 3, 3, w, None)
    assert out_f.shape == (0, 8) and out_i.shape == (0, 4)
