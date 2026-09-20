#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
#
# reg_compact 黑盒对拍: Regular 路径 unique/compact/out_indices/inv 原生化
# 契约: 输入任意序 g_flat (含 -1 无效); 返回 (out_buf (vpad,3), inv (nv,)
# 与 valid 行主序对齐, stats[0]=M)。sort/permutation 闭环在 op 内。
# 运行: cd /tmp && pytest <repo>/tests/sparse_conv/test_reg_compact.py -v

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "reg_compact"):
    pytest.skip("reg_compact not registered", allow_module_level=True)

DEVICE = "npu"
OP = torch.ops.ops_multimodal_fusion.reg_compact


def composite_ref(g_flat, ho, wo):
    vals = g_flat[g_flat >= 0]
    u, inv_ref = torch.unique(vals, return_inverse=True)
    hw = ho * wo
    rem = u % hw
    rows = torch.stack([u // hw, rem // wo, rem % wo], dim=1).to(torch.int32)
    return rows, inv_ref.to(torch.int32)


def run_op(g_flat, ho, wo):
    out_buf, inv, stats = OP(g_flat, ho, wo)
    m = int(stats[0].item())
    return out_buf[:m], inv, m


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("vals,ho,wo", [
    (torch.tensor([0, 3, 7, 11, 15]), 2, 2),            # 历史失败用例形态
    (torch.tensor([11, -1, 7, 3, -1, 15, 0]), 2, 2),    # 乱序 + 无效
    (torch.tensor([20, 5, 20]), 2, 2),                  # 评审方反例 (重复+乱序)
    (torch.tensor([-1, -1, 5, 5, 9]), 3, 4),            # -1 前缀 + 重复
    (torch.tensor([7, 7, 7]), 2, 2),                    # 单一 unique
    (torch.tensor([-1, -1]), 2, 2),                     # 全无效 M=0
    (torch.tensor([], dtype=torch.int32), 2, 2),        # 空
    (torch.tensor([4]), 3, 3),                          # 单元素 (尾段补齐)
    (torch.tensor([9, 1, 5, 3, 8, 2, 7, 4, 6]), 2, 2),  # 乱序跨 8 行边界 + 尾段
])
def test_small(vals, ho, wo):
    g = vals.to(torch.int32).to(DEVICE)
    got_rows, got_inv, m = run_op(g, ho, wo)
    ref_rows, ref_inv = composite_ref(g, ho, wo)
    assert m == ref_rows.size(0)
    assert torch.equal(got_rows, ref_rows)
    assert torch.equal(got_inv, ref_inv)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n,grid,seed", [
    (1000, 512, 0),
    (8192, 512, 1),
    (70000, 4096, 2),   # 多 tile + 尾段
])
def test_random(n, grid, seed):
    torch.manual_seed(seed)
    g = torch.randint(-1, grid, (n,), device=DEVICE, dtype=torch.int32)
    got_rows, got_inv, m = run_op(g, 8, 8)
    ref_rows, ref_inv = composite_ref(g, 8, 8)
    assert m == ref_rows.size(0)
    assert torch.equal(got_rows, ref_rows)
    assert torch.equal(got_inv, ref_inv)


# ---------------------------------------------------------------------------
# 端到端 scatter 闭环: 调用方编排
#   reg_compact(g_flat) → reg_scatter_add(out, contrib, inv)
# 直接满足 out[inv[i]] += contrib[i] — inv (int32) 与 contrib 行序由 op
# 内部对齐 (perm scatter 回填), dtype/顺序契约直连, 对拍 unique+index_add。
# ---------------------------------------------------------------------------
def _scatter_chain(g_flat, contrib, ho, wo):
    out_buf, inv, stats = OP(g_flat, ho, wo)
    assert inv.dtype == torch.int32   # 与 reg_scatter_add 契约直连
    m = int(stats[0].item())
    out = torch.zeros(m, contrib.size(1), device=g_flat.device,
                      dtype=torch.float32)
    return torch.ops.ops_multimodal_fusion.reg_scatter_add(
        out, contrib, inv, m, contrib.size(1))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("n,grid,co,seed", [
    (1000, 512, 4, 0),
    (8192, 4096, 16, 1),
    (30000, 100000, 32, 2),   # 乱序 + 重复 + 无效
])
def test_e2e_scatter(n, grid, co, seed):
    torch.manual_seed(seed)
    g = torch.randint(-1, grid, (n,), device=DEVICE, dtype=torch.int32)
    vmask = g >= 0
    contrib = torch.randn(int(vmask.sum()), co, device=DEVICE)
    got = _scatter_chain(g, contrib, 8, 8)
    uf, inv = torch.unique(g[vmask], return_inverse=True)
    ref = torch.zeros(uf.numel(), co, device=DEVICE)
    ref.index_add_(0, inv, contrib)
    assert got.size(0) == ref.size(0)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
        f"max_diff={(got - ref).abs().max().item()}"
