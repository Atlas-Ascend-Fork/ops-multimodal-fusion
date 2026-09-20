#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
#
# reg_rulebook 黑盒对拍 (批量流水 v2): Regular sparse conv 输出索引
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/ -v

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "reg_rulebook"):
    pytest.skip("reg_rulebook not registered", allow_module_level=True)

DEVICE = "npu"
OP = torch.ops.ops_multimodal_fusion.reg_rulebook


def composite_ref(indices, out_shape, st, pd):
    d_out, h_out, w_out = out_shape
    kflat = torch.arange(27, device=indices.device)
    kd = (kflat // 9).reshape(1, -1)
    khh = ((kflat // 3) % 3).reshape(1, -1)
    kww = (kflat % 3).reshape(1, -1)
    coords = indices[:, 1:4].long()
    oz_num = coords[:, 0:1] + pd[0] - kd
    oy_num = coords[:, 1:2] + pd[1] - khh
    ox_num = coords[:, 2:3] + pd[2] - kww
    divisible = ((oz_num % st[0] == 0) & (oy_num % st[1] == 0) & (ox_num % st[2] == 0))
    oz = oz_num // st[0]
    oy = oy_num // st[1]
    ox = ox_num // st[2]
    valid = (divisible & (oz >= 0) & (oz < d_out) &
             (oy >= 0) & (oy < h_out) & (ox >= 0) & (ox < w_out))
    g = (indices[:, 0].long().unsqueeze(1) * d_out * h_out * w_out +
         (oz * h_out + oy) * w_out + ox)
    return torch.where(valid, g, torch.full_like(g, -1)).reshape(-1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n,do,ho,wo,seed", [
    (100, 4, 8, 8, 0),
    (8000, 11, 400, 352, 1),
    (3000, 6, 200, 176, 2),
    (16000, 21, 800, 704, 3),
])
def test_match(n, do, ho, wo, seed):
    torch.manual_seed(seed)
    indices = torch.zeros(n, 4, dtype=torch.int32, device=DEVICE)
    indices[:, 1] = torch.randint(0, do * 2, (n,), device=DEVICE)
    indices[:, 2] = torch.randint(0, ho * 2, (n,), device=DEVICE)
    indices[:, 3] = torch.randint(0, wo * 2, (n,), device=DEVICE)
    key = indices[:, 1].long() * ho * wo * 2 + indices[:, 2].long() * wo * 2 + indices[:, 3].long() * 2
    indices = indices[key.argsort()].contiguous()
    got = OP(indices, do, ho, wo, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1).long()
    ref = composite_ref(indices, (do, ho, wo), (2, 2, 2), (1, 1, 1))
    assert torch.equal(got, ref), f"n={n}: max_diff={(got - ref).abs().max().item()}"
