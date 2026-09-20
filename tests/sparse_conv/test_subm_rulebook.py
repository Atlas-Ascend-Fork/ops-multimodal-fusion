#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
#
# subm_rulebook 泛化黑盒对拍: kvol <= 343 + dilation (与 sparse_subm 几何恒等)
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/test_subm_rulebook.py -v

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "subm_rulebook"):
    pytest.skip("subm_rulebook not registered", allow_module_level=True)

DEVICE = "npu"
OP = torch.ops.ops_multimodal_fusion.subm_rulebook


def composite_ref(indices, hash_t, spatial_shape, ks, dl):
    d, h, w = spatial_shape
    kz, kh, kw = ks
    kvol = kz * kh * kw
    out = torch.zeros(indices.size(0), kvol, dtype=torch.int64,
                      device=indices.device)
    idx = indices.long()
    cz, cy, cx = kz // 2, kh // 2, kw // 2
    for t in range(kvol):
        td, th, tw = t // (kh * kw), (t // kw) % kh, t % kw
        nz = idx[:, 1] + (td - cz) * dl[0]
        ny = idx[:, 2] + (th - cy) * dl[1]
        nx = idx[:, 3] + (tw - cx) * dl[2]
        ok = ((nz >= 0) & (nz < d) & (ny >= 0) & (ny < h) &
              (nx >= 0) & (nx < w))
        flat = (idx[:, 0] * d * h * w + nz.clamp(0, d - 1) * h * w +
                ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1))
        nb = torch.where(ok, hash_t[flat].long(), torch.zeros_like(flat))
        out[:, t] = nb
    return out


def make_case(n, spatial_shape, batch, seed):
    d, h, w = spatial_shape
    torch.manual_seed(seed)
    indices = torch.zeros(n, 4, dtype=torch.int32, device=DEVICE)
    indices[:, 0] = torch.randint(0, batch, (n,), device=DEVICE)
    indices[:, 1] = torch.randint(0, d, (n,), device=DEVICE)
    indices[:, 2] = torch.randint(0, h, (n,), device=DEVICE)
    indices[:, 3] = torch.randint(0, w, (n,), device=DEVICE)
    indices = torch.unique(indices, dim=0).to(torch.int32).contiguous()
    hash_t = torch.zeros(batch * d * h * w, dtype=torch.int32, device=DEVICE)
    flat = (indices[:, 0].long() * d * h * w + indices[:, 1].long() * h * w +
            indices[:, 2].long() * w + indices[:, 3].long())
    hash_t[flat] = torch.arange(1, flat.numel() + 1, device=DEVICE,
                                dtype=torch.int32)
    return indices, hash_t


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("cfg", [
    (300, (1, 6, 8, 8), (3, 3, 3), (1, 1, 1), 0),    # 向后兼容形态
    (2000, (2, 6, 8, 8), (3, 3, 3), (1, 1, 1), 1),   # 多 block + 批量流水
    (500, (1, 6, 8, 8), (5, 5, 5), (1, 1, 1), 2),    # kvol=125
    (500, (1, 9, 12, 12), (3, 3, 3), (2, 2, 2), 3),  # dilation=2
    (400, (1, 9, 6, 6), (7, 1, 1), (1, 1, 1), 4),    # 非对称 kvol=7
    (400, (1, 6, 8, 8), (1, 5, 3), (1, 1, 1), 5),    # kvol=15
])
def test_match(cfg):
    n, (b, d, h, w), ks, dl, seed = cfg
    indices, hash_t = make_case(n, (d, h, w), b, seed)
    kz, kh, kw = ks
    got = OP(indices, hash_t, d, h, w, kz, kh, kw, *dl)
    ref = composite_ref(indices, hash_t, (d, h, w), ks, dl)
    assert torch.equal(got.long(), ref), f"cfg={ks}dl={dl}"


def test_backward_compat_default_args():
    indices, hash_t = make_case(400, (6, 8, 8), 1, 7)
    got5 = OP(indices, hash_t, 6, 8, 8)
    got10 = OP(indices, hash_t, 6, 8, 8, 3, 3, 3, 1, 1, 1)
    assert torch.equal(got5, got10)
    assert got5.shape == (indices.size(0), 27)
