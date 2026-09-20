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

# Reference parity: NPU 稀疏算子链 (5 必需算子端到端) vs 稠密卷积语义。
# out_indices 排序对齐后逐项比较 out_features — mmcv python hash 语义 =
# 稠密 conv3d 在 active 位置的取值。
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/test_pipeline_parity.py -v

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from sparse_common import (DEVICE, OPS, Case, make_case, to_dense,
                           dense_rows, w_dense)
import ops_multimodal_fusion  # noqa: F401


def test_parity_regular_conv():
    torch.manual_seed(0)
    kz = kh = kw = 3
    ci, co = 4, 5
    idx = make_case(Case(20, 5, 5, 5, 1), seed=0)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    g = OPS.reg_rulebook(idx, 4, 4, 4, kz, kh, kw,
                         1, 1, 1, 1, 1, 1, 1, 1, 1).reshape(-1)
    vals = g[g >= 0]
    uf, inv = torch.unique(vals, return_inverse=True)
    m = uf.numel()
    kvol = kz * kh * kw
    pos = torch.arange(g.numel(), device=DEVICE)[g >= 0]
    a = feat[(pos // kvol).long()]
    contrib = (a @ w.permute(1, 0, 2).reshape(ci, -1)).reshape(
        -1, kvol, co)[torch.arange(pos.numel(), device=DEVICE), pos % kvol]
    out = torch.zeros(m, co, device=DEVICE)
    out.index_add_(0, inv, contrib)
    # 参照: 稠密 conv3d (pd=1, st=1) — out_indices 升序 flat = 行主序
    zgl = uf // 16          # (b*16 + oz) — batch=1
    oy = uf % 16
    idx4 = torch.stack([zgl // 16, zgl % 16, oy // 4, oy % 4], 1)
    ref = F.conv3d(to_dense(idx, feat, 5, 5, 5), w_dense(w, kz, kh, kw),
                   padding=1)
    assert torch.allclose(out, dense_rows(ref, idx4), atol=1e-2, rtol=1e-2), \
        f"regular conv parity: {(out - dense_rows(ref, idx4)).abs().max()}"


def test_parity_subm_conv():
    torch.manual_seed(1)
    kz = kh = kw = 3
    ci, co = 4, 6
    idx = make_case(Case(15, 4, 4, 4, 1), seed=1)
    feat = torch.randn(idx.size(0), ci, device=DEVICE)
    w = torch.randn(kz * kh * kw, ci, co, device=DEVICE)
    n = idx.size(0)
    hash_t = torch.zeros(16, 4, 4, dtype=torch.int32, device=DEVICE).flatten()
    flat = (idx[:, 0].long() * 64 + idx[:, 1].long() * 16 +
            idx[:, 2].long() * 4 + idx[:, 3].long())
    hash_t.scatter_(0, flat, torch.arange(
        1, n + 1, device=DEVICE, dtype=torch.int32))
    nb = OPS.subm_rulebook(idx, hash_t, 4, 4, 4)
    kvol = kz * kh * kw
    a = feat[(nb - 1).clamp(min=0).long()].reshape(n, kvol, ci)
    contrib = a * (nb > 0).long().unsqueeze(-1)             # 无效 tap → 0
    out = torch.einsum('ntc,tco->nto', contrib.cpu(),
                       w.cpu().reshape(kvol, ci, co)).sum(1)
    out = out.to(DEVICE)
    ref = F.conv3d(to_dense(idx, feat, 4, 4, 4), w_dense(w, kz, kh, kw),
                   padding=1)
    got = dense_rows(ref, idx)
    assert torch.allclose(out, got, atol=1e-2, rtol=1e-2), \
        f"subm parity: {(out - got).abs().max()}"
