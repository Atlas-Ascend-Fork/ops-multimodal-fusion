#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# sparse_conv_gemm 黑盒对拍: adv_api Matmul 托管 GEMM+bias vs torch 参考。
# 运行 (NPU, 需先安装 wheel):
#   cd /tmp && pytest <repo>/tests/sparse_conv/ -v

import pytest
import torch
import torch_npu  # noqa: F401
import ops_multimodal_fusion  # noqa: F401

if not hasattr(torch.ops.ops_multimodal_fusion, "sparse_conv_gemm"):
    pytest.skip("sparse_conv_gemm not registered", allow_module_level=True)

DEVICE = "npu"
OP = torch.ops.ops_multimodal_fusion.sparse_conv_gemm


def _rand_gemm(m, n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.3).half()
    wt = (torch.randn(n, k, generator=g) * 0.3).half()      # (N, K) n-major
    bias = (torch.randn(n, generator=g) * 0.3).half()
    return a, wt, bias


def _ref(a, wt, bias, use_bias):
    r = a.float() @ wt.float().t()
    if use_bias:
        r = r + bias.float()
    return r


# ------------------------------------------------------------
# L0 门槛
# ------------------------------------------------------------
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l0_shape_dtype():
    a, wt, bias = _rand_gemm(32, 32, 64)
    out = OP(a.to(DEVICE), wt.to(DEVICE), bias.to(DEVICE))
    assert out.shape == (32, 32)
    assert out.dtype == torch.float16


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("bad", ["a_dim", "dtype", "bias_dim"])
def test_l0_invalid(bad):
    a, wt, bias = _rand_gemm(32, 32, 64)
    if bad == "a_dim":
        args = (a[0].to(DEVICE), wt.to(DEVICE), None)
    elif bad == "k_mismatch":
        pass  # K pad 吸收差异, 校验移除 (行为变化见 commit)

    elif bad == "dtype":
        args = (a.float().to(DEVICE), wt.to(DEVICE), None)
    else:
        args = (a.to(DEVICE), wt.to(DEVICE), bias[:16].to(DEVICE))
    with pytest.raises(RuntimeError):
        OP(*args)


# ------------------------------------------------------------
# L1 功能: 多形状 + bias 开关
# ------------------------------------------------------------
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("m,n,k,seed", [
    (16, 16, 16, 1),
    (32, 32, 64, 2),
    (17, 16, 48, 3),       # M tail
    (128, 64, 144, 4),     # K = 9*16 (SubM C_in=16)
    (256, 64, 576, 5),     # K = 9*64
])
def test_l1_match(m, n, k, seed):
    a, wt, bias = _rand_gemm(m, n, k, seed)
    for use_bias in [True, False]:
        out = OP(a.to(DEVICE), wt.to(DEVICE),
                 bias.to(DEVICE) if use_bias else None)
        ref = _ref(a, wt, bias, use_bias)
        diff = (out.float().cpu() - ref).abs().max().item()
        rel = diff / max(ref.abs().max().item(), 1e-6)
        assert rel < 5e-2, f"m={m} n={n} k={k} bias={use_bias}: diff={diff:.4f}"


# ------------------------------------------------------------
# L2 管线规模 (SubM 深层: N=6000, K=27*128, Co=64)
# ------------------------------------------------------------
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l2_pipeline_shape():
    a, wt, bias = _rand_gemm(6000, 64, 27 * 128, seed=7)
    out = OP(a.to(DEVICE), wt.to(DEVICE), bias.to(DEVICE))
    ref = _ref(a, wt, bias, True)
    diff = (out.float().cpu() - ref).abs().max().item()
    rel = diff / ref.abs().max().item()
    assert rel < 5e-2, f"pipeline shape: diff={diff:.4f} rel={rel:.4f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_l2_zero_and_large_m():
    # M=1 极端 + M 跨核不整除 (6000/16=375 带 tail)
    for m in [1, 33, 999]:
        a, wt, bias = _rand_gemm(m, 32, 64, seed=m)
        out = OP(a.to(DEVICE), wt.to(DEVICE), bias.to(DEVICE))
        ref = _ref(a, wt, bias, True)
        diff = (out.float().cpu() - ref).abs().max().item()
        assert diff < 1.0, f"m={m}: diff={diff:.4f}"
