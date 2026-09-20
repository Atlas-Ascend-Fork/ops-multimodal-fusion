# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.library import Library


def _linear_projection_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    return F.linear(x, weight, bias)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("linear_projection", _linear_projection_impl)


# Meta
_lib_meta = Library("fusionrepo", "IMPL", "Meta")


def _linear_projection_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    out_features = weight.shape[0]
    shape = list(x.shape)
    shape[-1] = out_features
    return torch.empty(tuple(shape), device="meta", dtype=x.dtype)

_lib_meta.impl(
    "linear_projection",
    _linear_projection_meta
)


def linear_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    return torch.ops.fusionrepo.linear_projection(x, weight, bias)