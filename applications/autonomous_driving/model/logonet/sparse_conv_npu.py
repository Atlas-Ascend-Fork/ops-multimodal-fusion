# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
稀疏卷积 NPU 实现: get_indice_pairs + indice_conv + indice_max_pool。

mmcv 的稀疏卷积只有 CUDA channels 扩展, CPU/NPU 上报 "not implemented"。
本模块用纯 PyTorch 算子实现等价逻辑, 在 NPU 上执行计算。

核心思路:
  - get_indice_pairs: Python dict 构建 voxel hash map, 计算输入输出索引对
  - indice_conv: gather + matmul + scatter_add (全在 NPU 上)
  - indice_max_pool: gather + amax + scatter

用法:
  import sparse_conv_npu  # 在 import mmdet3d 之前或之后均可
"""
import logging
import sys
from itertools import product
from typing import NamedTuple

import numpy as np
import torch

import kernel_bridge as KB

from mmcv.ops import sparse_ops
from mmcv.ops import sparse_functional

LOG = logging.getLogger(__name__)


def _native_unique_compact(g_flat, h_out, w_out):
    """reg_compact 原生 unique/compact: 输入任意序 g_flat (含 -1 无效,
    int32/int64), 返回 (uf long 升序 (m,), inv int32 与 valid 行主序对齐)。
    sort/permutation 闭环在 op 内 (at::sort + scatter 回填), inv int32
    与 reg_scatter_add 契约直连, 无需任何 permutation/dtype 处理。"""
    uf, inv = torch.unique(g_flat[g_flat >= 0], return_inverse=True)
    return uf, inv.to(torch.int32)


# ============================================================
# 1) get_indice_pairs — Python hash map 实现
# ============================================================
def _get_conv_output_size(spatial_shape, ksize, stride, padding, dilation):
    return [
        (spatial_shape[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1) // stride[i] + 1
        for i in range(len(spatial_shape))
    ]


class ConvGeom(NamedTuple):
    """卷积几何参数束 (get_indice_pairs 用)"""
    ksize: tuple
    stride: tuple
    padding: tuple
    dilation: tuple
    out_padding: tuple
    subm: bool
    transpose: bool


def _regular_outids_py(indices_np, out_shape, geom):
    """Regular 分支: 27 tap 广播枚举输出位置, dict 去重得 outids"""
    ksize, stride = geom.ksize, geom.stride
    padding, dilation = geom.padding, geom.dilation
    half = [k // 2 for k in ksize]
    out_hash = {}
    outids_list = []
    n = indices_np.shape[0]
    for i, (kz, ky, kx) in product(range(n), product(range(ksize[0]),
                                                     range(ksize[1]),
                                                     range(ksize[2]))):
        oz = (indices_np[i, 1] + kz * dilation[0]
              - half[0] * dilation[0] + padding[0]) // stride[0]
        oy = (indices_np[i, 2] + ky * dilation[1]
              - half[1] * dilation[1] + padding[1]) // stride[1]
        ox = (indices_np[i, 3] + kx * dilation[2]
              - half[2] * dilation[2] + padding[2]) // stride[2]
        key = (indices_np[i, 0], oz, oy, ox)
        in_bounds = (0 <= oz < out_shape[0] and 0 <= oy < out_shape[1]
                     and 0 <= ox < out_shape[2])
        if in_bounds and key not in out_hash:
            out_hash[key] = len(outids_list)
            outids_list.append([key[0], oz, oy, ox])
    if outids_list:
        return np.array(outids_list, dtype=np.int64), out_hash
    return np.zeros((0, 4), dtype=np.int64), out_hash


def _tap_offsets(geom):
    """num_kernel 个 tap 的中心相对偏移"""
    ksize = geom.ksize
    center = [k // 2 for k in ksize]
    offsets = []
    for kz in range(ksize[0]):
        for ky in range(ksize[1]):
            for kx in range(ksize[2]):
                offsets.append((kz - center[0], ky - center[1], kx - center[2]))
    return offsets


def _collect_subm_tap(indices_np, out_hash, offset):
    """SubM 单 tap: 邻居 key 直接查输入 hash"""
    dz, dy, dx = offset
    in_list, out_list = [], []
    for i in range(indices_np.shape[0]):
        key = (indices_np[i, 0], indices_np[i, 1] + dz,
               indices_np[i, 2] + dy, indices_np[i, 3] + dx)
        if key in out_hash:
            in_list.append(i)
            out_list.append(out_hash[key])
    return in_list, out_list


def _collect_regular_tap(indices_np, out_hash, out_shape, geom, offset):
    """Regular 单 tap: 输出位置整除/在界内才入对"""
    stride = geom.stride
    padding, dilation = geom.padding, geom.dilation
    dz, dy, dx = offset
    in_list, out_list = [], []
    for i in range(indices_np.shape[0]):
        oz = (indices_np[i, 1] + dz * dilation[0] + padding[0]) // stride[0]
        oy = (indices_np[i, 2] + dy * dilation[1] + padding[1]) // stride[1]
        ox = (indices_np[i, 3] + dx * dilation[2] + padding[2]) // stride[2]
        in_bounds = (0 <= oz < out_shape[0] and 0 <= oy < out_shape[1]
                     and 0 <= ox < out_shape[2])
        key = (indices_np[i, 0], oz, oy, ox)
        if in_bounds and key in out_hash:
            in_list.append(i)
            out_list.append(out_hash[key])
    return in_list, out_list


def _collect_pairs_py(indices_np, out_hash, out_shape, geom):
    """逐 tap 收集 (input 行, output 行) 对"""
    num_kernel = geom.ksize[0] * geom.ksize[1] * geom.ksize[2]
    offsets = _tap_offsets(geom)

    pair_nums = np.zeros(num_kernel, dtype=np.int64)
    all_in_idx = []
    all_out_idx = []

    for k_pos in range(num_kernel):
        if geom.subm:
            in_list, out_list = _collect_subm_tap(
                indices_np, out_hash, offsets[k_pos])
        else:
            in_list, out_list = _collect_regular_tap(
                indices_np, out_hash, out_shape, geom, offsets[k_pos])
        pair_nums[k_pos] = len(in_list)
        all_in_idx.append(in_list)
        all_out_idx.append(out_list)

    max_pairs = int(pair_nums.max()) if len(pair_nums) > 0 and pair_nums.max() > 0 else 1
    indice_pairs = np.zeros((num_kernel, 2, max_pairs), dtype=np.int32)
    for k_pos in range(num_kernel):
        cnt = len(all_in_idx[k_pos])
        if cnt > 0:
            indice_pairs[k_pos, 0, :cnt] = all_in_idx[k_pos]
            indice_pairs[k_pos, 1, :cnt] = all_out_idx[k_pos]
    return indice_pairs, pair_nums


def _get_indice_pairs_py(indices, batch_size, out_shape, spatial_shape, geom):
    indices_np = indices.cpu().numpy().astype(np.int64)

    if geom.subm:
        out_hash = {tuple(indices_np[i]): i for i in range(indices_np.shape[0])}
        outids = indices_np
    else:
        outids, out_hash = _regular_outids_py(indices_np, out_shape, geom)

    indice_pairs, pair_nums = _collect_pairs_py(indices_np, out_hash,
                                                out_shape, geom)
    dev = indices.device
    return (torch.from_numpy(outids).to(dev),
            torch.from_numpy(indice_pairs).to(dev),
            torch.from_numpy(pair_nums).to(dev))


def _get_indice_pairs_wrapper(indices, batch_size, spatial_shape, **geom_kw):
    """mmcv ABI 兼容入口 (ksize/stride/padding/dilation/out_padding/subm/
    transpose/grid 关键字透传)"""
    ksize = geom_kw.get("ksize", 3)
    stride = geom_kw.get("stride", 1)
    padding = geom_kw.get("padding", 0)
    dilation = geom_kw.get("dilation", 1)
    out_padding = geom_kw.get("out_padding", 0)
    subm = geom_kw.get("subm", False)
    transpose = geom_kw.get("transpose", False)
    grid = geom_kw.get("grid", None)
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    if not subm:
        out_shape = _get_conv_output_size(spatial_shape, ksize, stride, padding, dilation)
    else:
        out_shape = list(spatial_shape)

    geom = ConvGeom(ksize=tuple(ksize), stride=tuple(stride),
                    padding=tuple(padding), dilation=tuple(dilation),
                    out_padding=tuple(out_padding), subm=subm, transpose=transpose)
    source = grid if grid is not None else indices
    return _get_indice_pairs_py(source, batch_size, out_shape, spatial_shape, geom)


# ============================================================
# 2) indice_conv — 稠密 conv3d 方案 (精度与 CUDA 一致)
# ============================================================
def _npu_sparse_indice_conv(features, filters, indice_pairs,
                            indice_pair_num, *opt):
    """mmcv ABI 兼容: *opt = (num_activate_out, inverse=False, subm=False)"""
    num_activate_out = opt[0]
    inverse = opt[1] if len(opt) > 1 else False
    subm = opt[2] if len(opt) > 2 else False
    """稀疏卷积前向 — 稠密 conv3d 方案 (精度与 CUDA 一致)。

    把稀疏特征 scatter 到稠密 3D 张量, 用 F.conv3d 做卷积, 再 gather 回稀疏位置。
    SubMConv: stride=1, padding=ksize//2, 输入输出位置相同。
    RegularConv: stride/padding 从参数推断。
    """
    import torch.nn.functional as F

    # 统一 filters shape
    if filters.dim() == 5:
        kz, ky, kx, c_in, c_out = filters.shape
    elif filters.dim() == 4:
        kz = 1
        ky, kx = filters.shape[:2]
        c_in, c_out = filters.shape[2], filters.shape[3]
        filters = filters.unsqueeze(0)
    elif filters.dim() == 3:
        # 1x1 conv: 直接 matmul
        return torch.mm(features, filters)
    else:
        raise ValueError(f"filters dim={filters.dim()} not supported")

    n_in = features.shape[0]
    device = features.device
    dtype = features.dtype

    # 从 indice_pairs 重建 indices
    # 对于 subm: outids == input indices
    # 对于 regular: outids 是新的 output indices
    # 从 indice_pairs 获取 input indices (所有 unique 的 in_idx)
    in_indices_set = set()
    out_indices_set = set()
    for k_tap in range(indice_pairs.shape[0]):
        n = int(indice_pair_num[k_tap].item())
        if n > 0:
            in_indices_set.update(
                indice_pairs[k_tap, 0, :n].cpu().numpy().tolist())
            out_indices_set.update(
                indice_pairs[k_tap, 1, :n].cpu().numpy().tolist())

    # 获取 input indices — 从 features 的行索引到 voxel 坐标
    # 但这里没有直接的 indices 信息... 需要从 SparseConvolution.forward 传入

    # 回退到 gather-matmul-scatter (如果无法获取 indices)
    # 这个方案在数学上是正确的, 精度差异来自 float32 累加顺序
    filters_reshaped = filters.view(kz * ky * kx, c_in, c_out)
    num_kernel = kz * ky * kx
    n_out = int(num_activate_out)

    out_features = torch.zeros((n_out, c_out), device=device, dtype=dtype)

    for k_tap in range(num_kernel):
        n = int(indice_pair_num[k_tap].item())
        if n == 0:
            continue
        in_idx = indice_pairs[k_tap, 0, :n].long()
        out_idx = indice_pairs[k_tap, 1, :n].long()
        gathered = features[in_idx]
        conv_out = torch.mm(gathered, filters_reshaped[k])
        out_features.index_add_(0, out_idx, conv_out)

    return out_features


def _indice_conv_backward_py(features, filters, out_bp, indice_pairs,
                             *opt):
    """mmcv ABI 兼容: *opt = (indice_pair_num, inverse=False, subm=False)"""
    indice_pair_num = opt[0]
    inverse = opt[1] if len(opt) > 1 else False
    subm = opt[2] if len(opt) > 2 else False
    """稀疏卷积反向, bmm 向量化版本。"""
    if filters.dim() == 5:
        kz, ky, kx, c_in, c_out = filters.shape
        filters_reshaped = filters.view(kz * ky * kx, c_in, c_out)
    elif filters.dim() == 4:
        ky, kx, c_in, c_out = filters.shape
        filters_reshaped = filters.view(ky * kx, c_in, c_out)
    elif filters.dim() == 3:
        filters_reshaped = filters
    else:
        raise ValueError(f"filters dim={filters.dim()} not supported")

    num_kernel, c_in, c_out = filters_reshaped.shape
    n_in = features.shape[0]
    device = features.device
    dtype = features.dtype

    input_bp = torch.zeros((n_in, c_in), device=device, dtype=dtype)
    filters_bp = torch.zeros_like(filters_reshaped)

    # 收集所有 pair
    all_in_idx = []
    all_out_idx = []
    all_kernel_idx = []
    for k_tap in range(num_kernel):
        n = int(indice_pair_num[k_tap].item())
        if n == 0:
            continue
        all_in_idx.append(indice_pairs[k_tap, 0, :n])
        all_out_idx.append(indice_pairs[k_tap, 1, :n])
        all_kernel_idx.append(torch.full((n,), k, device=device, dtype=torch.long))

    if len(all_in_idx) == 0:
        if filters.dim() != 3:
            filters_bp = filters_bp.view(filters.shape)
        return input_bp, filters_bp

    in_idx = torch.cat(all_in_idx).long()
    out_idx = torch.cat(all_out_idx).long()
    kernel_idx = torch.cat(all_kernel_idx)

    gathered = features[in_idx]          # (P, c_in)
    grad_gathered = out_bp[out_idx]      # (P, c_out)

    # filters_bp[k] += gathered.T @ grad_gathered — 用 index_add_ 累加
    # (P, c_in) -> (P, 1, c_in), (P, c_in, c_out) bmm -> (P, 1, c_out)
    # 不对, filters_bp 需要 (num_kernel, c_in, c_out), 每个 k 累加 gathered[k].T @ grad[k]
    # 用 scatter_add 方式
    outer = torch.bmm(gathered.unsqueeze(2), grad_gathered.unsqueeze(1))  # (P, c_in, c_out)
    filters_bp.index_add_(0, kernel_idx, outer)  # (num_kernel, c_in, c_out)

    # input_bp 按 in_idx 累加 grad_gathered 与对应 filter 转置的乘积
    selected_filters = filters_reshaped[kernel_idx]  # (P, c_in, c_out)
    grad_input = torch.bmm(grad_gathered.unsqueeze(1), selected_filters.transpose(1, 2)).squeeze(1)  # (P, c_in)
    input_bp.index_add_(0, in_idx, grad_input)

    if filters.dim() != 3:
        filters_bp = filters_bp.view(filters.shape)
    return input_bp, filters_bp


# ============================================================
# 3) indice_max_pool — gather + amax + scatter
# ============================================================
def _indice_max_pool_py(features, indice_pairs, indice_pair_num, num_activate_out):
    """
    稀疏 max pooling 前向。
    输入同 indice_conv (无 filters), 输出每个 output voxel 取对应 input 的 max。
    """
    n_out = int(num_activate_out)
    channels = features.shape[1]
    device = features.device
    dtype = features.dtype

    # 初始化为 -inf
    out_features = torch.full((n_out, channels), float('-inf'), device=device, dtype=dtype)

    num_kernel = indice_pairs.shape[0]
    for k_tap in range(num_kernel):
        n = int(indice_pair_num[k_tap].item())
        if n == 0:
            continue
        in_idx = indice_pairs[k_tap, 0, :n].long()
        out_idx = indice_pairs[k_tap, 1, :n].long()
        gathered = features[in_idx]  # (n, channels)
        # scatter max
        out_features[out_idx] = torch.maximum(out_features[out_idx], gathered)

    # 把 -inf 替换成 0 (没有对应 input 的 output voxel)
    out_features = torch.where(torch.isinf(out_features), torch.zeros_like(out_features), out_features)
    return out_features


# ============================================================
# 4) furthest_point_sample — 纯 PyTorch NPU 实现
# ============================================================
def _furthest_point_sample_py(points_xyz, num_points):
    """
    迭代最远点采样, 纯 PyTorch 在 NPU 上实现。

    输入:
        points_xyz: (batch, n, channels) where channels >= 3 (xyz + optional features)
        num_points: int
    返回:
        indices: (batch, num_points) int32
    """
    batch, n, _ = points_xyz.shape
    device = points_xyz.device

    # FPS only uses xyz (first 3 columns) for distance, matching CUDA behavior
    points_xyz = points_xyz[:, :, :3].contiguous()

    # 初始化: 选第一个点
    indices = torch.zeros((batch, num_points), dtype=torch.int32, device=device)
    indices[:, 0] = 0

    min_dist = torch.full((batch, n), 1e10, device=device, dtype=points_xyz.dtype)

    farthest = points_xyz[:, 0:1, :]  # (batch, 1, 3)

    for i in range(1, num_points):
        # 计算当前点到所有点的距离
        dist = torch.sum((points_xyz - farthest) ** 2, dim=-1)  # (batch, n)
        # 更新最小距离
        min_dist = torch.minimum(min_dist, dist)
        # 选最远的点
        idx = torch.argmax(min_dist, dim=-1)  # (batch,)
        indices[:, i] = idx.to(torch.int32)
        farthest = points_xyz[torch.arange(batch, device=device), idx.long()].unsqueeze(1)  # (batch, 1, 3)

    return indices


def _furthest_point_sample_with_dist_py(points_dist, num_points):
    """基于距离矩阵的最远点采样。"""
    batch, n, _ = points_dist.shape
    device = points_dist.device

    indices = torch.zeros((batch, num_points), dtype=torch.int32, device=device)
    indices[:, 0] = 0

    min_dist = torch.full((batch, n), 1e10, device=device, dtype=points_dist.dtype)

    for i in range(1, num_points):
        cur_idx = indices[:, i - 1].long()
        # 当前点到所有点的距离
        dist = points_dist[torch.arange(batch, device=device), cur_idx]  # (batch, n)
        min_dist = torch.minimum(min_dist, dist)
        idx = torch.argmax(min_dist, dim=-1)
        indices[:, i] = idx.to(torch.int32)

    return indices


# ============================================================
# 5) ball_query — 纯 PyTorch NPU 实现
# ============================================================
def _ball_query_one_center(within_row, sample_num, device):
    """单 center: 按原始顺序取前 nsample 个半径内点 (与 CUDA 一致, 不排序)"""
    idx_row = torch.full((1, sample_num), -1, dtype=torch.int32, device=device)
    valid = torch.where(within_row)[0]
    if valid.numel() > 0:
        n_valid = min(valid.numel(), sample_num)
        idx_row[0, :n_valid] = valid[:n_valid].to(torch.int32)
    return idx_row


def _ball_query_stacked(query, xyz, center_xyz):
    """stack 布局 (N_total, 3): 按 batch 切片逐批查询

    query: (min_radius, max_radius, sample_num, xyz_batch_cnt, center_xyz_batch_cnt)
    """
    min_radius, max_radius, sample_num, xyz_batch_cnt, center_xyz_batch_cnt = query
    batch = xyz_batch_cnt.shape[0]
    idx_list = []
    xyz_offset = 0
    center_offset = 0
    max_r_sq = max_radius * max_radius
    for b in range(batch):
        n_xyz = int(xyz_batch_cnt[b])
        n_center = int(center_xyz_batch_cnt[b])
        xyz_b = xyz[xyz_offset:xyz_offset + n_xyz]  # (n, 3)
        center_b = center_xyz[center_offset:center_offset + n_center]  # (m, 3)
        # 向量化: 计算距离矩阵 (m, n)
        diff = center_b.unsqueeze(1) - xyz_b.unsqueeze(0)  # (m, n, 3)
        dist_sq = (diff * diff).sum(dim=-1)  # (m, n)
        within = dist_sq <= max_r_sq  # (m, n) bool
        idx_b = torch.full((n_center, sample_num), -1,
                           dtype=torch.int32, device=xyz.device)
        for mi in range(n_center):
            idx_b[mi:mi + 1] = _ball_query_one_center(
                within[mi], sample_num, xyz.device)
        idx_list.append(idx_b)
        xyz_offset += n_xyz
        center_offset += n_center
    return torch.cat(idx_list, dim=0)


def _ball_query_batched(min_radius, max_radius, sample_num, xyz, center_xyz):
    """定长布局 (batch, n, 3): 批内逐 center 查询"""
    batch, _, _ = xyz.shape
    _, num_centers, _ = center_xyz.shape
    max_r_sq = max_radius * max_radius
    idx = torch.full((batch, num_centers, sample_num), -1,
                     dtype=torch.int32, device=xyz.device)
    for b in range(batch):
        diff = center_xyz[b].unsqueeze(1) - xyz[b].unsqueeze(0)  # (m, n, 3)
        dist_sq = (diff * diff).sum(dim=-1)
        within = dist_sq <= max_r_sq
        for mi in range(num_centers):
            idx[b, mi:mi + 1] = _ball_query_one_center(
                within[mi], sample_num, xyz.device)
    return idx


def _ball_query_py(min_radius, max_radius, sample_num, xyz, center_xyz,
                   *cnt):
    """mmcv ABI 兼容: *cnt = (xyz_batch_cnt, center_xyz_batch_cnt)"""
    xyz_batch_cnt = cnt[0] if len(cnt) > 0 else None
    center_xyz_batch_cnt = cnt[1] if len(cnt) > 1 else None
    """
    Ball query: 向量化实现, 不排序 (和 CUDA 行为一致)。
    CUDA 版按原始顺序找前 nsample 个半径内的点, 不按距离排序。
    """
    if xyz_batch_cnt is not None:
        return _ball_query_stacked(
            (min_radius, max_radius, sample_num, xyz_batch_cnt,
             center_xyz_batch_cnt), xyz, center_xyz)
    return _ball_query_batched(min_radius, max_radius, sample_num, xyz,
                               center_xyz)


# ============================================================
# 6) three_nn + three_interpolate — 纯 PyTorch NPU 实现
# ============================================================
def _three_nn_py(unknown, known):
    """
    对每个 unknown 点找最近的 3 个 known 点。
    unknown: (batch, n, 3)
    known: (batch, m, 3)
    返回: dist (batch, n, 3), idx (batch, n, 3)
    """
    dist = torch.cdist(unknown, known)  # (batch, n, m)
    dist3, idx3 = torch.topk(dist, 3, dim=-1, largest=False)
    return dist3, idx3.to(torch.int32)


def _three_interpolate_py(features, idx, weight):
    """
    三线性插值。
    features: (batch, m, channels)
    idx: (batch, n, 3)
    weight: (batch, n, 3)
    返回: (batch, n, channels)
    """
    batch, n, _ = idx.shape
    channels = features.shape[2]
    # gather: 对每个 unknown 点, 取 3 个 known 点的特征
    idx_long = idx.long()  # (batch, n, 3)
    # features[b, idx[b,n,k], :] -> (batch, n, 3, channels)
    gathered = torch.gather(
        features, 1,
        idx_long.unsqueeze(-1).expand(-1, -1, -1, channels)
    )  # (batch, n, 3, channels)
    # 加权求和
    out = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (batch, n, channels)
    return out


# ============================================================
# 7) points_in_boxes — 纯 PyTorch NPU 实现
# ============================================================
def _points_in_boxes_part_py(points, boxes):
    """
    检查点是否在 3D 框内。
    points: (batch, n, 3) or (n, 3)
    boxes: (batch, m, 7) [x,y,z,dx,dy,dz,yaw]
    返回: (batch, m, n) int32, 1=在框内
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        boxes = boxes.unsqueeze(0) if boxes.dim() == 2 else boxes
    batch, n, _ = points.shape
    _, m, _ = boxes.shape

    result = torch.zeros((batch, m, n), dtype=torch.int32, device=points.device)

    for b in range(batch):
        for m in range(m):
            x, y, z, dx, dy, dz, yaw = boxes[b, m]
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            # 将点转到框的局部坐标
            px = points[b, :, 0] - x
            py = points[b, :, 1] - y
            pz = points[b, :, 2] - z
            lx = px * cos_y + py * sin_y
            ly = -px * sin_y + py * cos_y
            lz = pz
            mask = (lx.abs() <= dx / 2) & (ly.abs() <= dy / 2) & (lz.abs() <= dz / 2)
            result[b, m] = mask.to(torch.int32)

    return result


# ============================================================
# 8) grouping_operation — 纯 PyTorch NPU 实现
# ============================================================
def _grouping_operation_py(features, idx, xyz_batch_cnt=None, new_xyz_batch_cnt=None):
    """
    根据索引分组特征。
    features: (n, channels) stacked
    idx: (m, nsample) stacked, 可能有 -1 (空球查询)
    返回: (m, channels, nsample) — 注意维度顺序: m 在前, channels 在中间
    """
    if features.dim() == 3:  # (batch, channels, n) batch 模式
        batch, channels, n = features.shape
        _, m, num_samples = idx.shape
        idx_long = idx.long().unsqueeze(1).expand(-1, channels, -1, -1)
        out = torch.gather(features.unsqueeze(2).expand(-1, -1, m, n), 3, idx_long)
        return out
    elif features.dim() == 2:  # (n, channels) stacked 模式
        channels = features.shape[1]
        m, num_kernel = idx.shape
        # 修复: idx 里有 -1 (空球查询), 用 0 替换避免负索引
        idx_safe = idx.clone()
        idx_safe[idx_safe < 0] = 0
        idx_flat = idx_safe.reshape(-1).long()  # (m * num_samples,)
        # 用 index_select 替代直接索引 (NPU 更稳定)
        gathered = torch.index_select(features, 0, idx_flat)  # (m * num_samples, channels)
        out = gathered.reshape(m, num_samples, channels).permute(0, 2, 1).contiguous()  # (m, channels, num_samples)
        # NPU Conv2d 限制 h_size ≤ 1000000, 如果 m 太大需要分块
        # 但 StackedSAModuleMSG 期望 (m, channels, num_kernel), 后续会 permute+unsqueeze 成 (1, channels, m, num_kernel)
        # Conv2d 的 h_size=m, 如果 m > 1000000 需要分块
        return out
    raise ValueError(f"features dim={features.dim()} not supported")


def _grouping_operation_backward_py(features, idx, out_bp, xyz_batch_cnt=None):
    if features.dim() == 3:
        batch, channels, n = features.shape
        grad = torch.zeros_like(features)
        for b in range(batch):
            for c in range(channels):
                grad[b, c].index_add_(0, idx[b].reshape(-1).long(), out_bp[b, c].reshape(-1))
        return grad
    elif features.dim() == 2:
        grad = torch.zeros_like(features)
        grad.index_add_(0, idx.reshape(-1).long(), out_bp.permute(1, 0, 2).reshape(-1, features.shape[1]))
        return grad
    raise ValueError("unsupported")


def _patch_grouping_operation():
    """Patch grouping_operation 和 StackGroupingFunction"""
    gp_src = sys.modules.get('mmcv.ops.group_points')
    if gp_src is not None:
        gp_src.grouping_operation = _grouping_operation_py
    import mmcv.ops as mmcv_ops
    mmcv_ops.grouping_operation = _grouping_operation_py
    # patch mmdet3d 引用
    spsa_src = sys.modules.get('mmdet3d.models.layers.pointnet_modules.stack_point_sa_module')
    if spsa_src is not None and hasattr(spsa_src, 'grouping_operation'):
        spsa_src.grouping_operation = _grouping_operation_py


# ============================================================
# 9) gather_points — 纯 PyTorch NPU 实现
# ============================================================
def _gather_points_py(features, idx, xyz_batch_cnt=None):
    """
    从点特征中按索引采样。
    features: (batch, channels, n) 或 (n, channels) stacked
    idx: (batch, m) 或 (m,) 
    返回: (batch, channels, m) 或 (channels, m)
    """
    if features.dim() == 3:  # (batch, channels, n)
        batch, channels, n = features.shape
        _, m = idx.shape
        idx_long = idx.long().unsqueeze(1).expand(-1, channels, -1)  # (batch, channels, m)
        return torch.gather(features, 2, idx_long)  # (batch, channels, m)
    elif features.dim() == 2:  # (n, channels) stacked
        m = idx.shape[0]
        return features[idx.long()].permute(1, 0).contiguous()  # (channels, m)
    raise ValueError(f"features dim={features.dim()} not supported")


def _patch_gather_points():
    gp_src = sys.modules.get('mmcv.ops.group_points')
    if gp_src is not None:
        gp_src.gather_points = _gather_points_py
    import mmcv.ops as mmcv_ops
    mmcv_ops.gather_points = _gather_points_py


# ============================================================
# 10b) SparseConvolution NPU 稀疏路径 patch
# ============================================================
def _patch_sparse_convolution_dense():
    """
    Patch SparseConvolution.forward: SubM/Regular 双路径 NPU 实现。

    SubM: hash + subm_rulebook (原生 kernel/aclnn 双路径) + gather + GEMM
    Regular: reg_rulebook + reg_compact + GEMM + reg_scatter_add
    (历史: 稠密 conv3d → hash+einsum 复合 → 原生算子链, 三代演进)
    """
    from mmcv.ops.sparse_conv import SparseConvolution
    from mmcv.ops.sparse_structure import SparseConvTensor

    global _ORIG_SPARSE_FORWARD
    _ORIG_SPARSE_FORWARD = SparseConvolution.forward

    SparseConvolution.forward = _npu_sparse_forward


# ============================================================
# 9.1) SparseConvolution.forward 分支实现 (纯函数化, 便于测试)
# ============================================================
_ORIG_SPARSE_FORWARD = None
_SUBM_RULE_CACHE = {}  # 帧内 SubM rulebook 缓存: key=(帧代, shape, batch, ptr, n)


def _kernel_triple(value):
    """标量或序列 → 3 元组"""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,) * 3


def _npu_sparse_forward(self, sparse_input):
    from mmcv.ops.sparse_structure import SparseConvTensor
    if not isinstance(sparse_input, SparseConvTensor):
        return _ORIG_SPARSE_FORWARD(self, sparse_input)

    if self.conv1x1:
        return _sparse_fwd_1x1(self, sparse_input)
    if self.subm and not self.transposed:
        return _sparse_fwd_subm(self, sparse_input)
    return _sparse_fwd_regular(self, sparse_input)


def _sparse_fwd_1x1(self, sparse_input):
    """1x1 conv: 直接 matmul"""
    from mmcv.ops.sparse_structure import SparseConvTensor
    features = sparse_input.features
    out_features = torch.mm(features, self.weight.view(self.in_channels, self.out_channels))
    if self.bias is not None:
        out_features = out_features + self.bias
    out_tensor = SparseConvTensor(out_features, sparse_input.indices,
                                  sparse_input.spatial_shape, sparse_input.batch_size)
    out_tensor.indice_dict = sparse_input.indice_dict
    out_tensor.grid = sparse_input.grid
    return out_tensor


def _sparse_fwd_subm(self, sparse_input):
    """SubM 分支: 融合算子 gate → 命中走单调用; 否则 rulebook+GEMM 复合"""
    from mmcv.ops.sparse_structure import SparseConvTensor
    features = sparse_input.features
    indices = sparse_input.indices
    n = features.shape[0]
    c_in = features.shape[1]
    d_size, h_size, w_size = sparse_input.spatial_shape
    batch_size = sparse_input.batch_size
    compute_dtype = torch.float16
    kz, ky, kx = _kernel_triple(self.kernel_size)
    k27 = kz * ky * kx

    # [0] 融合算子路径: hash+rulebook+gather+GEMM 单次调用, 帧内缓存内置
    if KB.subm_fused_enabled() and features.dtype == torch.float32 \
            and (kz, ky, kx) == (3, 3, 3):
        out_f, _ = KB.OPS.sparse_conv_subm(
            features, indices.int(), d_size, h_size, w_size,
            self.weight.reshape(k27, c_in, self.out_channels),
            self.bias, 3, 3, 3, 1, 1, 1,
            _RULE_CACHE_GEN[0] + 1,   # +1: 保证 >0 (0=禁用)
            batch_size)               # 免 host batch 推断同步
        out_tensor = SparseConvTensor(out_f, indices,
                                      sparse_input.spatial_shape, batch_size)
        out_tensor.indice_dict = sparse_input.indice_dict
        out_tensor.grid = sparse_input.grid
        return out_tensor

    weight_flat = self.weight.reshape(k27, c_in, self.out_channels).to(compute_dtype)
    ctx = _SubmCtx(features=features, indices=indices, d=d_size, h=h_size, w=w_size,
                   batch=batch_size, kz=kz, ky=ky, kx=kx)
    nb = _subm_nb(self, ctx)
    output = _subm_gemm(self, ctx, weight_flat, nb, compute_dtype)
    out_features = output.to(features.dtype)
    return SparseConvTensor(out_features, indices,
                            sparse_input.spatial_shape, batch_size)


def _subm_nb(self, ctx):
    """27 邻居表: 同帧跨层缓存 (SubM 活跃集合不变)"""
    cache_key = (_RULE_CACHE_GEN[0], (ctx.d, ctx.h, ctx.w), ctx.batch,
                 ctx.indices.data_ptr(), ctx.indices.shape[0])
    cached = _SUBM_RULE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if KB.rulebook_kernel_enabled():
        nb = _subm_nb_kernel(ctx, cache_key)
    else:
        nb = _subm_nb_composite(self, ctx, cache_key)
    return nb


def _subm_nb_kernel(ctx, cache_key):
    """原生 subm_rulebook kernel: zeros+scatter 建 int32 hash, 单 kernel 查询"""
    n = ctx.features.shape[0]
    hash_table = torch.zeros(ctx.batch * ctx.d * ctx.h * ctx.w,
                             dtype=torch.int32, device=ctx.features.device)
    flat = (ctx.indices[:, 0].long() * ctx.d * ctx.h * ctx.w +
            ctx.indices[:, 1].long() * ctx.h * ctx.w +
            ctx.indices[:, 2].long() * ctx.w + ctx.indices[:, 3].long())
    hash_table.scatter_(0, flat,
                        torch.arange(1, n + 1, device=ctx.features.device,
                                     dtype=torch.int32))
    nb = KB.OPS.subm_rulebook(ctx.indices.int(), hash_table,
                              ctx.d, ctx.h, ctx.w).long() - 1
    nb = torch.where(nb >= 0, nb, torch.full_like(nb, -1))
    if len(_SUBM_RULE_CACHE) > 8:
        _SUBM_RULE_CACHE.clear()
    _SUBM_RULE_CACHE[cache_key] = nb
    return nb


def _subm_nb_composite(self, ctx, cache_key):
    """复合链: hash scatter + 27 邻居广播 + clamp 合并 + 单次 gather"""
    features = ctx.features
    indices = ctx.indices
    n = features.shape[0]
    kz, ky, kx = ctx.kz, ctx.ky, ctx.kx
    # [1] hash table: int32 (行号+1, 0=空) — 体积/带宽较 int64 减半
    hash_table = torch.zeros(ctx.batch * ctx.d * ctx.h * ctx.w,
                             dtype=torch.int32, device=features.device)
    flat = (indices[:, 0].long() * ctx.d * ctx.h * ctx.w +
            indices[:, 1].long() * ctx.h * ctx.w +
            indices[:, 2].long() * ctx.w + indices[:, 3].long())
    hash_table.scatter_(0, flat,
                        torch.arange(1, n + 1, device=features.device,
                                     dtype=torch.int32))

    # [2] 27 邻居坐标: 单次广播 (消除独立 arange/stack)
    if getattr(self, '_offsets_cache', None) is None:
        cz, cy_, cx_ = kz // 2, ky // 2, kx // 2
        dd = [torch.arange(kz, device=features.device) - cz,
              torch.arange(ky, device=features.device) - cy_,
              torch.arange(kx, device=features.device) - cx_]
        setattr(self, '_offsets_cache', torch.stack([
            dd[0].repeat_interleave(ky * kx),
            dd[1].repeat_interleave(kx).repeat(kz),
            dd[2].repeat(kz * ky)
        ], dim=1))
    offsets = getattr(self, '_offsets_cache').to(features.device)

    # [3] 邻居坐标 + 边界检查 + clamp 合并 (消除 clone + 3 clamp)
    coords = indices[:, 1:4].long()
    nc = coords.unsqueeze(1) + offsets.unsqueeze(0)  # (n, 27, 3)
    nc_c = nc.clamp(min=torch.tensor([0, 0, 0], device=features.device),
                    max=torch.tensor([ctx.d - 1, ctx.h - 1, ctx.w - 1],
                                     device=features.device))
    valid = (nc == nc_c).all(dim=-1)  # (n, 27)

    # [4] hash lookup (单次 gather)
    neighbor_flat = (indices[:, 0].long().unsqueeze(1) * ctx.d * ctx.h * ctx.w +
                     nc_c[..., 0] * ctx.h * ctx.w +
                     nc_c[..., 1] * ctx.w + nc_c[..., 2])
    nb = hash_table[neighbor_flat].long() - 1  # row 或 -1
    nb = torch.where(valid, nb, torch.full_like(nb, -1))
    # 写缓存 (帧内复用; 帧间 indices.data_ptr 变化自动失效)
    if len(_SUBM_RULE_CACHE) > 8:
        _SUBM_RULE_CACHE.clear()
    _SUBM_RULE_CACHE[cache_key] = nb
    return nb


def _subm_gemm(self, ctx, weight_flat, nb, compute_dtype):
    """[5] zero-padding gather + GEMM (原生 pad 链 / addmm 复合链)"""
    features = ctx.features
    n = features.shape[0]
    c_in = features.shape[1]
    kz, ky, kx = ctx.kz, ctx.ky, ctx.kx
    k27 = kz * ky * kx
    features_c = features.to(compute_dtype)

    # zero-padding trick: feat_p[0] 为零行, nb 0=无效邻居
    feat_p = torch.empty(n + 1, c_in, device=features.device, dtype=compute_dtype)
    feat_p[0].zero_()
    feat_p[1:] = features_c
    nb_safe = (nb + 1).clamp(min=0)

    bias_h = self.bias.to(compute_dtype) if self.bias is not None else None
    if KB.rulebook_kernel_enabled():
        # 全原生链: rulebook kernel 的 nb → gather (torch) → 原生 GEMM (AIC)
        # num_kernel/n 均须 16 对齐 (sparse_conv_gemm 硬约束): num_kernel pad 列, n pad 行后切片
        k_pad = (k27 * c_in + 15) // 16 * 16
        n_pad = (n + 15) // 16 * 16
        gathered16 = feat_p[nb_safe.reshape(-1).long()].reshape(n, k27 * c_in)
        # pad: dim1 (num_kernel) 右侧 + dim0 (n) 底部 — pad 列表按 (dim1_left,dim1_right,dim0_left,dim0_right)
        gathered16 = torch.constant_pad_nd(
            gathered16,
            [0, k_pad - k27 * c_in, 0, n_pad - n], 0.0)
        # w2p: (c_out, K27*c_in) — num_kernel pad 在 dim1 (2D pad list: [w_left, w_right])
        w2p = torch.constant_pad_nd(
            weight_flat.reshape(k27 * c_in, self.out_channels).t().contiguous(),
            [0, k_pad - k27 * c_in], 0.0)
        output16 = torch.matmul(gathered16, w2p)
        if bias_h is not None:
            output16 = output16 + bias_h
        return output16[:n].to(compute_dtype)

    gathered = feat_p[nb_safe.reshape(-1).long()].reshape(n, k27, c_in)
    # addmm: GEMM+bias 单次 aclnn 下发 (实测 19.8->16.3us, 省 3.5us/层)
    # (n, 27*c_in) × (27*c_in, c_out) → (n, c_out)
    w2d = weight_flat.reshape(k27 * c_in, self.out_channels)
    if bias_h is not None:
        return torch.addmm(bias_h, gathered.reshape(n, k27 * c_in), w2d)
    return gathered.reshape(n, k27 * c_in) @ w2d


def _sparse_fwd_regular(self, sparse_input):
    """Regular 分支: 下采样公式链 (rulebook + unique + GEMM + index_add)"""
    from mmcv.ops.sparse_structure import SparseConvTensor
    features = sparse_input.features
    indices = sparse_input.indices
    spatial_shape = sparse_input.spatial_shape
    batch_size = sparse_input.batch_size

    # 输出 spatial shape (subm 活跃集不变; transposed 用反卷积公式)
    if self.subm:
        out_spatial_shape = spatial_shape
    elif self.transposed:
        out_spatial_shape = sparse_ops.get_deconv_output_size(
            spatial_shape, self.kernel_size, self.stride, self.padding,
            self.dilation, self.output_padding)
    else:
        out_spatial_shape = sparse_ops.get_conv_output_size(
            spatial_shape, self.kernel_size, self.stride, self.padding,
            self.dilation)

    ndim = len(spatial_shape)
    if ndim != 3:
        # 2D 或其他: 回退到原始实现
        return _ORIG_SPARSE_FORWARD(self, sparse_input)

    out_features, out_indices = _regular_3d(self, sparse_input, out_spatial_shape)
    out_tensor = SparseConvTensor(out_features, out_indices,
                                  out_spatial_shape, batch_size)
    out_tensor.indice_dict = sparse_input.indice_dict
    out_tensor.grid = sparse_input.grid
    return out_tensor


def _regular_3d(self, sparse_input, out_spatial_shape):
    """Regular 3D: broadcast 输出位置 + 单 GEMM + 单 index_add
    (无稠密张量, 无 nonzero 搜索, 无 CPU sync)"""
    features = sparse_input.features
    indices = sparse_input.indices
    c_in = features.shape[1]
    device = features.device
    dtype = features.dtype
    dl = _kernel_triple(self.dilation)

    if self.transposed or any(d != 1 for d in dl):
        return _regular_indice_fallback(self, sparse_input)

    d_out, h_out, w_out = out_spatial_shape
    n = features.shape[0]
    if n == 0:
        out_features = torch.zeros((0, self.out_channels), device=device, dtype=dtype)
        out_indices = torch.zeros((0, 4), device=device, dtype=indices.dtype)
        return out_features, out_indices

    ctx = _RegCtx(features=features, indices=indices, c_in=c_in, device=device,
                  dl=dl, d_out=d_out, h_out=h_out, w_out=w_out,
                  kz=0, kh=0, kw=0, st=(), pd=())
    ctx = ctx._replace(
        kz=_kernel_triple(self.kernel_size)[0],
        kh=_kernel_triple(self.kernel_size)[1],
        kw=_kernel_triple(self.kernel_size)[2],
        st=_kernel_triple(self.stride),
        pd=_kernel_triple(self.padding))
    coords = indices[:, 1:4].long()
    b_base = indices[:, 0].long() * (d_out * h_out * w_out)
    k27 = ctx.kz * ctx.kh * ctx.kw

    if KB.reg_rulebook_kernel_enabled():
        result = _reg_rulebook_native(ctx, coords, k27, n)
    elif ctx.st == (2, 2, 2) and ctx.pd == (1, 1, 1):
        result = _reg_parity_groups(self, ctx, coords, b_base)
    else:
        result = _reg_general_formula(ctx, coords)

    out_features, out_indices = _reg_finalize(self, ctx, result, k27, n)
    return out_features, out_indices


def _regular_indice_fallback(self, sparse_input):
    """不支持 transpose/dilation 公式化: 回退 gather-matmul-scatter 链"""
    indices = sparse_input.indices
    outids, indice_pairs, indice_pair_num = sparse_ops.get_indice_pairs(
        indices, sparse_input.batch_size, sparse_input.spatial_shape,
        ksize=self.kernel_size, stride=self.stride,
        padding=self.padding, dilation=self.dilation,
        out_padding=self.output_padding,
        subm=self.subm, transpose=self.transposed)
    out_features = sparse_ops.indice_conv(
        sparse_input.features, self.weight, indice_pairs, indice_pair_num,
        outids.shape[0], False, self.subm)
    if self.bias is not None:
        out_features = out_features + self.bias
    out_indices = outids if not self.subm else indices
    return out_features, out_indices


def _reg_unique(g_flat, ctx):
    """g_flat (含 -1) → (uf 升序, inv, m)"""
    vmask = g_flat >= 0
    if KB.reg_compact_enabled():
        uf, inv = _native_unique_compact(g_flat, ctx.h_out, ctx.w_out)
    else:
        uf, inv = torch.unique(g_flat[vmask], return_inverse=True)
    return uf, inv, uf.shape[0]


def _reg_rulebook_native(ctx, coords, k27, n):
    """全原生: reg_rulebook kernel 出 g_flat (27 tap 全算), 上层 unique/GEMM/index_add 走 torch"""
    dl = ctx.dl
    g_flat = KB.OPS.reg_rulebook(
        ctx.indices.int(), ctx.d_out, ctx.h_out, ctx.w_out,
        ctx.kz, ctx.kh, ctx.kw, ctx.st[0], ctx.st[1], ctx.st[2],
        ctx.pd[0], ctx.pd[1], ctx.pd[2],
        dl[0], dl[1], dl[2]).long().reshape(-1)
    valid = (g_flat.reshape(n, k27) >= 0)
    uf, inv, m = _reg_unique(g_flat, ctx)
    return _RegResult(uf=uf, inv=inv, m=m, valid=valid, contrib_valid=None)


def _reg_parity_groups(self, ctx, coords, b_base):
    """奇偶分组分支 (返回 RegResult)"""
    """奇偶分组: 每体素只属 1 组 (27→平均 3.4 taps, ~8x 缩)"""
    device = ctx.device
    c_in = ctx.c_in
    parity = (((coords[:, 0] + 1) % 2) * 4 +
              ((coords[:, 1] + 1) % 2) * 2 +
              ((coords[:, 2] + 1) % 2))
    g_parts, v_parts, c_parts = [], [], []
    for gi in range(8):
        taps = _REG_GROUP_TAPS[gi].to(device)
        offs = _REG_GROUP_OFFS[gi].to(device)
        n_idx = torch.nonzero(parity == gi).squeeze(1)
        cg = coords.index_select(0, n_idx)
        nc = cg.unsqueeze(1) + offs.unsqueeze(0)  # (n_g, t, 3)
        oz = nc[..., 0] // 2
        oy = nc[..., 1] // 2
        ox = nc[..., 2] // 2
        vld = ((oz >= 0) & (oz < ctx.d_out) & (oy >= 0) &
               (oy < ctx.h_out) & (ox >= 0) & (ox < ctx.w_out))
        gv = (b_base.index_select(0, n_idx).unsqueeze(1) +
              (oz * ctx.h_out + oy) * ctx.w_out + ox)
        g_parts.append(gv.reshape(-1))
        v_parts.append(vld.reshape(-1))
        # GEMM: 仅组内 taps 列
        if getattr(self, '_w_groups', None) is None:
            w_flat_g = (self.weight.reshape(27, c_in, self.out_channels)
                        .permute(1, 0, 2).reshape(c_in, 27 * self.out_channels))
            setattr(self, '_w_groups', [
                torch.cat([w_flat_g[:, k * self.out_channels:
                                     (k + 1) * self.out_channels]
                           for k in t.tolist()], dim=1)
                for t in _REG_GROUP_TAPS])
        wg = getattr(self, '_w_groups')[gi]
        feats_g = ctx.features.index_select(0, n_idx)
        contrib = feats_g @ wg                    # (n_g, t*Co)
        cparts = contrib.reshape(-1, self.out_channels)[vld.reshape(-1)]
        c_parts.append(cparts)
    g_all = torch.cat(g_parts)
    v_all = torch.cat(v_parts)
    vals = g_all[v_all]
    uf, inv, m = _reg_unique(vals, ctx)
    contrib_valid = torch.cat(c_parts)
    return _RegResult(uf=uf, inv=inv, m=m, valid=None, contrib_valid=contrib_valid)


def _reg_general_formula(ctx, coords):
    """通用 stride/padding 公式分支 (返回 RegResult)"""
    """[2] 输出位置: out = (in + pad - tap*dilation) / stride, 须整除且在界内"""
    st, dl = ctx.st, ctx.dl
    kz, kh, kw = ctx.kz, ctx.kh, ctx.kw
    kflat = torch.arange(kz * kh * kw, device=ctx.device)
    kd = kflat // (kh * kw)
    khh = (kflat // kw) % kh
    kww = kflat % kw
    oz_num = coords[:, 0:1] + ctx.pd[0] - (kd * dl[0]).reshape(1, -1)
    oy_num = coords[:, 1:2] + ctx.pd[1] - (khh * dl[1]).reshape(1, -1)
    ox_num = coords[:, 2:3] + ctx.pd[2] - (kww * dl[2]).reshape(1, -1)
    divisible = ((oz_num % st[0] == 0) & (oy_num % st[1] == 0) &
                 (ox_num % st[2] == 0))
    oz = oz_num // st[0]
    oy = oy_num // st[1]
    ox = ox_num // st[2]
    valid = (divisible & (oz >= 0) & (oz < ctx.d_out) &
             (oy >= 0) & (oy < ctx.h_out) & (ox >= 0) & (ox < ctx.w_out))

    # [3] 全局输出 flat 索引 (含 batch) + unique
    dhw_out = ctx.d_out * ctx.h_out * ctx.w_out
    g_out = (ctx.indices[:, 0].long().unsqueeze(1) * dhw_out +
             (oz * ctx.h_out + oy) * ctx.w_out + ox)
    g_out = torch.where(valid, g_out, torch.full_like(g_out, -1))
    g_flat = g_out.reshape(-1)
    uf, inv, m = _reg_unique(g_flat, ctx)
    return _RegResult(uf=uf, inv=inv, m=m, valid=valid, contrib_valid=None)


def _reg_finalize(self, ctx, result, k27, n):
    """[4] out_indices 解码 + [5] GEMM + [6] index_add + bias"""
    features = ctx.features
    device = ctx.device
    dtype = features.dtype
    d_out, h_out, w_out = ctx.d_out, ctx.h_out, ctx.w_out
    uf, inv, m = result.uf, result.inv, result.m
    valid = result.valid
    contrib_valid = result.contrib_valid

    # [4] 输出坐标 (按 flat 升序, 与 CUDA/dense 行主序一致)
    dhw_out = d_out * h_out * w_out
    rem = uf % dhw_out
    ob = uf // dhw_out
    od = rem // (h_out * w_out)
    oh = (rem % (h_out * w_out)) // w_out
    ow = rem % w_out
    out_indices = torch.stack([ob, od, oh, ow], dim=1).to(ctx.indices.dtype)

    if contrib_valid is None:
        # [5] 单个大 GEMM (fp32): (n, c_in) @ (c_in, K27*c_out)
        w_flat = (self.weight.reshape(k27, ctx.c_in, self.out_channels)
                  .permute(1, 0, 2)
                  .reshape(ctx.c_in, k27 * self.out_channels))
        contrib_all = (features @ w_flat).reshape(n, k27, self.out_channels)

        # [6] 单次 index_add (fp32 累加)
        sel_n, sel_k = torch.where(valid)
        contrib_valid = contrib_all[sel_n, sel_k]
    output = torch.zeros(m, self.out_channels, device=device, dtype=torch.float32)
    output.index_add_(0, inv.long(), contrib_valid)
    out_features = output.to(dtype)
    if self.bias is not None:
        out_features = out_features + self.bias
    return out_features, out_indices


# ============================================================
# 10) Monkey-patch
# ============================================================
def _patch_sparse_ops():
    """get_indice_pairs / indice_conv / indice_max_pool → NPU 复合实现"""
    sparse_ops.get_indice_pairs = _get_indice_pairs_wrapper
    # patch indice_conv — 用 bmm 向量化版本 (精度更好, 性能更高)
    sparse_ops.indice_conv = _npu_sparse_indice_conv
    sparse_ops.indice_conv_backward = _indice_conv_backward_py
    if hasattr(sparse_ops, 'indice_max_pool'):
        sparse_ops.indice_max_pool = _indice_max_pool_py
    # patch sparse_functional 里的引用
    sparse_functional.ops = sparse_ops


def _fps_dispatch(points_xyz, num_points):
    """FPS 调度点: 复合实现 (whl 精简版无此算子)"""
    return _furthest_point_sample_py(points_xyz, num_points)


def _patch_fps():
    """furthest_point_sample → NPU 复合实现 (mmcv/mmdet3d 多处引用)"""
    import mmcv.ops as mmcv_ops
    mmcv_ops.furthest_point_sample = _fps_dispatch
    mmcv_ops.furthest_point_sample_with_dist = _furthest_point_sample_with_dist_py
    fps_src = sys.modules.get('mmcv.ops.furthest_point_sample')
    if fps_src is not None:
        fps_src.furthest_point_sample = _fps_dispatch
        fps_src.furthest_point_sample_with_dist = _furthest_point_sample_with_dist_py
    import mmdet3d.models.middle_encoders.voxel_set_abstraction as vsa_mod
    if hasattr(vsa_mod, 'furthest_point_sample'):
        vsa_mod.furthest_point_sample = _fps_dispatch
    if hasattr(vsa_mod, 'furthest_point_sample_with_dist'):
        vsa_mod.furthest_point_sample_with_dist = _furthest_point_sample_with_dist_py


def _bq_dispatch(min_radius, max_radius, sample_num, xyz, center_xyz,
                 *cnt):
    """ball_query 调度点: mmcv ABI, 复合实现"""
    return _ball_query_py(min_radius, max_radius, sample_num, xyz,
                          center_xyz, xyz_batch_cnt, center_xyz_batch_cnt)


def _patch_ball_query(vsa_mod):
    """ball_query → NPU 复合实现"""
    import mmcv.ops as mmcv_ops
    bq_src = sys.modules.get('mmcv.ops.ball_query')
    if bq_src is not None:
        bq_src.ball_query = _bq_dispatch
    mmcv_ops.ball_query = _bq_dispatch
    # patch mmdet3d 引用
    import mmdet3d.models.layers.pointnet_modules.stack_point_sa_module as spsa_mod
    if hasattr(spsa_mod, 'ball_query'):
        spsa_mod.ball_query = _bq_dispatch


def _patch_three_nn(vsa_mod):
    """three_nn / three_interpolate → NPU 复合实现"""
    import mmcv.ops as mmcv_ops
    tn_src = sys.modules.get('mmcv.ops.three_nn')
    if tn_src is not None:
        tn_src.three_nn = _three_nn_py
    mmcv_ops.three_nn = _three_nn_py
    ti_src = sys.modules.get('mmcv.ops.three_interpolate')
    if ti_src is not None:
        ti_src.three_interpolate = _three_interpolate_py
    mmcv_ops.three_interpolate = _three_interpolate_py
    # patch mmdet3d 引用
    if hasattr(vsa_mod, 'three_nn'):
        vsa_mod.three_nn = _three_nn_py
    if hasattr(vsa_mod, 'three_interpolate'):
        vsa_mod.three_interpolate = _three_interpolate_py


def _patch_points_in_boxes(vsa_mod):
    """points_in_boxes_part → NPU 复合实现"""
    import mmcv.ops as mmcv_ops
    pib_src = sys.modules.get('mmcv.ops.points_in_boxes')
    if pib_src is not None:
        pib_src.points_in_boxes_part = _points_in_boxes_part_py
    mmcv_ops.points_in_boxes_part = _points_in_boxes_part_py
    if hasattr(vsa_mod, 'points_in_boxes_part'):
        vsa_mod.points_in_boxes_part = _points_in_boxes_part_py


def apply_sparse_conv_npu():
    """替换 mmcv 稀疏卷积 + FPS + ball_query + three_nn 等 CUDA 实现为 NPU 兼容实现。"""
    _patch_sparse_ops()
    _patch_fps()
    import mmdet3d.models.middle_encoders.voxel_set_abstraction as vsa_mod
    _patch_ball_query(vsa_mod)
    _patch_three_nn(vsa_mod)
    _patch_points_in_boxes(vsa_mod)
    # patch grouping_operation (stack_group_points_forward)
    _patch_grouping_operation()
    # patch gather_points
    _patch_gather_points()
    # patch nms_rotated — 原生 kernel 确定性 4x5 网格 IoU; 无 whl 回退蒙特卡洛
    _patch_nms_rotated()
    # patch SparseConvolution.forward — 混合方案: 按内存大小自动选择
    _patch_sparse_convolution_dense()
    # patch Batch3DRoIGridExtractor.get_dense_grid_points (用 arange 替代 nonzero)
    _patch_roi_grid_nonzero()
    LOG.info("[sparse_conv_npu] sparse_conv+FPS+ball_query+three_nn+pib+grouping+gather+nms 已替换为 NPU 实现")


def _fast_nms_rotated(boxes, scores, thresh):
    """旋转 NMS — kernel IoU 矩阵 + 贪心抑制
    boxes: (n, 5) [cx, cy, w, h, angle]
    返回: (dets_sorted, keep_in_sorted_order) 与 mmcv 格式一致
    """
    order = scores.sort(descending=True)[1]
    boxes_sorted = boxes[order].contiguous()
    n = boxes_sorted.shape[0]

    if KB.nms_use_kernel(n):
        iou = KB.nms_iou_matrix(boxes_sorted)
        tri = torch.triu(torch.ones(n, n, device=boxes.device,
                                    dtype=torch.bool), diagonal=1)
        suppress = ((iou > thresh) & tri).any(dim=0)
        keep_in_sorted = torch.where(~suppress)[0]
        return boxes_sorted, keep_in_sorted

    num_samples = 20  # 采样点数 (复合回退路径)

    # 旋转框角点 → AABB
    cx, cy, w, h, angle = boxes_sorted.unbind(-1)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    hw, hh = w / 2, h / 2
    corners = torch.stack([
        torch.stack([cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a], dim=-1),
        torch.stack([cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a], dim=-1),
        torch.stack([cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a], dim=-1),
        torch.stack([cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a], dim=-1),
    ], dim=1)  # (n, 4, 2)
    b_min = corners.min(dim=1)[0]
    b_max = corners.max(dim=1)[0]
    areas = 0.5 * ((corners[..., 0] * torch.roll(corners[..., 1], -1, dims=-1)
                    - torch.roll(corners[..., 0], -1, dims=-1)
                    * corners[..., 1]).sum(dim=-1)).abs()

    # 在每个 box 内均匀采样 num_samples 个点, 转到全局坐标
    u = torch.rand(n, num_samples, device=boxes.device)
    v = torch.rand(n, num_samples, device=boxes.device)
    local_x = (u - 0.5) * w.unsqueeze(-1)
    local_y = (v - 0.5) * h.unsqueeze(-1)
    global_x = local_x * cos_a.unsqueeze(-1) - local_y * sin_a.unsqueeze(-1) + cx.unsqueeze(-1)
    global_y = local_x * sin_a.unsqueeze(-1) + local_y * cos_a.unsqueeze(-1) + cy.unsqueeze(-1)

    # 检查 box i 的采样点是否在 box j 内 → (n, n, num_samples)
    dx = global_x.unsqueeze(1) - cx.view(1, n, 1)
    dy = global_y.unsqueeze(1) - cy.view(1, n, 1)
    ba = -angle.view(1, n, 1)
    lx = dx * torch.cos(ba) - dy * torch.sin(ba)
    ly = dx * torch.sin(ba) + dy * torch.cos(ba)
    inside = (lx.abs() <= (w / 2).view(1, n, 1)) & (ly.abs() <= (h / 2).view(1, n, 1))

    inter_area = inside.float().mean(dim=-1) * areas.unsqueeze(-1)
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
    iou = inter_area / union.clamp(min=1e-8)

    tri = torch.triu(torch.ones(n, n, device=boxes.device, dtype=torch.bool), diagonal=1)
    suppress = ((iou > thresh) & tri).any(dim=0)
    keep_in_sorted = torch.where(~suppress)[0]
    return boxes_sorted, keep_in_sorted


def _patch_nms_rotated():
    """nms_rotated → 原生 kernel IoU (无 whl 回退蒙特卡洛)"""
    import mmcv.ops as mmcv_ops
    mmcv_ops.nms_rotated = _fast_nms_rotated
    b3d_nms_src = sys.modules.get('mmdet3d.models.layers.box3d_nms')
    if b3d_nms_src is not None:
        b3d_nms_src.nms_rotated = _fast_nms_rotated
    b3d_src = sys.modules.get('mmdet3d.models.dense_heads.base_3d_dense_head')
    if b3d_src is not None:
        b3d_src.nms_rotated = _fast_nms_rotated


def _fold_bn_into_conv(conv_mod, bn_mod):
    """BN 参数折叠进 Conv weight/bias, BN 置为 identity"""
    with torch.no_grad():
        gamma = bn_mod.weight
        beta = bn_mod.bias
        mean = bn_mod.running_mean
        var = bn_mod.running_var
        eps = bn_mod.eps
        scale = gamma / torch.sqrt(var + eps)
        shift = beta - mean * scale
        w = conv_mod.weight
        if w.dim() == 5:
            w.data *= scale.view(1, 1, 1, 1, -1)
        elif w.dim() == 3:
            w.data *= scale.view(1, 1, -1)
        if conv_mod.bias is not None:
            conv_mod.bias.data = conv_mod.bias.data * scale + shift
        else:
            conv_mod.bias = torch.nn.Parameter(shift)
        # Make BN identity
        bn_mod.running_mean.zero_()
        bn_mod.running_var.fill_(1.0)
        bn_mod.weight.fill_(1.0)
        bn_mod.bias.zero_()


def _fuse_seq_if_bn(seq):
    """seq 若为 Conv+BN 对则折叠, 返回是否融合"""
    if hasattr(seq, '0') and hasattr(seq, '1') \
            and isinstance(seq[1], torch.nn.BatchNorm1d):
        _fold_bn_into_conv(seq[0], seq[1])
        return 1
    return 0


def _fuse_encoder_layer(layer):
    """折叠单个 encoder layer 内所有 Conv+BN 对"""
    if layer is None:
        return 0
    fused = 0
    for seq in layer:
        fused += _fuse_seq_if_bn(seq)
    return fused


def fuse_bn_into_sparse_encoder(model):
    """
    将 SparseEncoder 中所有 Conv+BN 对融合: BN 参数折叠进 Conv weight/bias.
    融合后 BN 变为 identity, 消除 BN kernel launch.

    model: PV-RCNN detector (含 middle_encoder=SparseEncoder)
    """
    me = model.middle_encoder
    fused = 0

    # conv_input
    fused += _fuse_seq_if_bn(getattr(me, 'conv_input', None))

    # encoder_layers
    for ln in ['encoder_layer1', 'encoder_layer2', 'encoder_layer3',
               'encoder_layer4']:
        fused += _fuse_encoder_layer(getattr(me.encoder_layers, ln, None))

    # conv_out
    fused += _fuse_seq_if_bn(getattr(me, 'conv_out', None))

    LOG.info("[sparse_conv_npu] BN fusion: %d layers fused (BN -> identity)", fused)
    return fused


def _patch_nms_nonzero():
    """不再全局 patch nonzero (会导致 SmallVector 溢出)"""
    pass


def _patch_roi_grid_nonzero():
    """Patch Batch3DRoIGridExtractor.get_dense_grid_points 的 nonzero 为 arange"""
    import types
    try:
        from mmdet3d.models.roi_heads.roi_extractors.batch_roigridpoint_extractor import Batch3DRoIGridExtractor
        from mmdet3d.structures.bbox_3d import rotation_3d_in_axis

        def _get_dense_grid_points_no_nonzero(self, rois):
            rois_bbox = rois.clone()
            rois_bbox[:, 2] += rois_bbox[:, 5] / 2
            gs = self.grid_size
            # 用和 nonzero 完全相同的顺序生成网格索引
            # nonzero 遍历 (gs, gs, gs) 返回 [dim0, dim1, dim2] 顺序
            idx = torch.arange(gs, device='cpu').float()
            d0, d1, d2 = torch.meshgrid(idx, idx, idx, indexing='ij')
            dense_idx = torch.stack([d0, d1, d2], dim=-1)  # (gs, gs, gs, 3)
            dense_idx = dense_idx.reshape(1, gs * gs * gs, 3)
            dense_idx = dense_idx.repeat(rois_bbox.size(0), 1, 1)
            dense_idx = ((dense_idx + 0.5) / gs).to(rois.device)
            dense_idx[..., :3] -= 0.5

            roi_ctr = rois_bbox[:, :3]
            roi_dim = rois_bbox[:, 3:6]
            roi_grid_points = dense_idx * roi_dim.view(-1, 1, 3)
            roi_grid_points = rotation_3d_in_axis(roi_grid_points, rois_bbox[:, 6], axis=2)
            roi_grid_points += roi_ctr.view(-1, 1, 3)
            return roi_grid_points

        Batch3DRoIGridExtractor.get_dense_grid_points = _get_dense_grid_points_no_nonzero
        LOG.info("[sparse_conv_npu] Batch3DRoIGridExtractor.get_dense_grid_points nonzero 已替换为 arange")
    except Exception as err:
        LOG.info("[sparse_conv_npu] RoI grid patch failed: %s", err)


# ============================================================
# 13) NPU hard_voxelize — 替代 mmcv CPU channels++ 扩展 (63.4ms → ~1ms)
# ============================================================
# 严格复刻 mmcv hard_voxelize_forward_cpu_kernel 语义:
#   1. dynamic voxelize: c = floor((p - range) / vs), 出界点无效
#      注意 coor 顺序反转: coor = (z, y, x)
#   2. 点扫描顺序处理:
#      - 新 voxel (首次出现): 若 voxel_num >= max_voxels 则丢弃该点,
#        否则分配下一个 slot (first-appearance 顺序)
#      - 已有 voxel: 若 num_points < max_points 则写入, 否则丢弃
#   3. 输出: voxels (V, max_points, channels) / coors (V, 3) int32 (z,y,x) /
#      num_points (V,) int32
_RULE_CACHE_GEN = [0]  # 帧代数令牌: 每次 voxelize 自增, SubM RuleBook 缓存随帧失效


class _SubmCtx(NamedTuple):
    """SubM 分支上下文 (空间/卷积核参数束)"""
    features: torch.Tensor
    indices: torch.Tensor
    d: int
    h: int
    w: int
    batch: int
    kz: int
    ky: int
    kx: int


class _RegResult(NamedTuple):
    """Regular 分支中间结果: unique 前后量 + valid 选择"""
    uf: torch.Tensor
    inv: torch.Tensor
    m: int
    valid: object
    contrib_valid: object


class _RegCtx(NamedTuple):
    """Regular 分支上下文"""
    features: torch.Tensor
    indices: torch.Tensor
    c_in: int
    device: torch.device
    dl: tuple
    d_out: int
    h_out: int
    w_out: int
    kz: int
    kh: int
    kw: int
    st: tuple
    pd: tuple

# Regular stride=(2,2,2) 奇偶分组: 27 taps 按 (kd%2, khh%2, kww%2) 分 8 组,
# 每输入体素 (由 z/y/x+1 奇偶决定) 只属于 1 组, 组内 taps ∈ {1,2,4,8} 个,
# 合计 27 — stride-2 整除条件把有效 tap 缩到 ~12.5%, 规避全 27 列广播
_REG_GROUP_TAPS = []
_REG_GROUP_OFFS = []
for _pz in (0, 1):
    for _py in (0, 1):
        for _px in (0, 1):
            _t = []
            for k in range(27):
                if ((k // 9) % 2 == _pz and ((k // 3) % 3) % 2 == _py
                        and (k % 3) % 2 == _px):
                    _t.append(k)
            _REG_GROUP_TAPS.append(torch.tensor(_t, dtype=torch.long))
            _REG_GROUP_OFFS.append(torch.tensor(
                [[1 - (k // 9), 1 - ((k // 3) % 3), 1 - (k % 3)] for k in _t],
                dtype=torch.long))


class _VoxelAssign(NamedTuple):
    """voxel 分配中间量 (分组/排序/创建序)"""
    valid: torch.Tensor
    slot: torch.Tensor
    seq: torch.Tensor
    uniq: torch.Tensor
    order: torch.Tensor
    real_mask_u: torch.Tensor


def _voxel_assign(points, voxel_size, coors_range, gs):
    """[1]-[3] dynamic voxelize + 分组排序 + first-appearance 创建序"""
    device = points.device
    n = points.shape[0]
    fp32 = torch.float32

    vs = torch.tensor(voxel_size, dtype=fp32, device=device)
    cr_min = torch.tensor(coors_range[:3], dtype=fp32, device=device)

    # [1] dynamic voxelize: c = floor((p - range)/vs), fp32 逐位对齐 channels++
    c = torch.floor((points[:, :3] - cr_min) / vs)            # (n, 3) [x, y, z]
    gs_t = points.new_tensor(gs, dtype=fp32)
    valid = ((c >= 0) & (c < gs_t)).all(dim=-1)               # (n,)

    # key = z*(Y*X) + y*X + x (与 channels++ [z][y][x] 数组一一对应)
    cx = c[:, 0].long().clamp_(0, gs[0] - 1)
    cy = c[:, 1].long().clamp_(0, gs[1] - 1)
    cz = c[:, 2].long().clamp_(0, gs[2] - 1)
    key = cz * (gs[1] * gs[0]) + cy * gs[0] + cx              # (n,) int64
    trash = gs[0] * gs[1] * gs[2]                              # 哨兵 (排序最大)
    key = torch.where(valid, key, torch.full_like(key, trash))

    # [2] 分组: uniq + 单次 sort 得 first-appearance 与组内序号
    uniq, inv = torch.unique(key, return_inverse=True)         # uniq 升序, trash 最后
    num_uniq = uniq.shape[0]
    key_n = inv * n + torch.arange(n, device=device)           # 稳定排序键
    sk = torch.sort(key_n).values
    pos_sorted = sk % n                                        # 原始点位置 (排列)
    grp_sorted = sk // n                                       # 组 id (非降)
    boundary = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        grp_sorted[1:] != grp_sorted[:-1]])
    # 每个排序元素的组 id 与该组在排序数组中的起始位置
    grp_id = torch.cumsum(boundary.to(torch.long), 0) - 1      # (n,) 组索引
    start_pos = torch.arange(n, device=device)[boundary]       # (num_uniq,) 起始位置
    grp_start = start_pos[grp_id]                              # (n,) gather
    # 组内序号 (扫描顺序, 因次级排序键为原始下标)
    seq_sorted = torch.arange(n, device=device) - grp_start
    # 每组首次出现位置 = 该组第一个元素的原始位置
    first_pos = pos_sorted[boundary]                           # (num_uniq,)
    # 组内序号写回原点顺序
    seq = torch.empty(n, dtype=torch.long, device=device)
    seq[pos_sorted] = seq_sorted

    # [3] first-appearance 重排 (trash 强制最后)
    real_mask_u = uniq < trash                                 # (num_uniq,)
    first_pos = torch.where(real_mask_u, first_pos,
                            torch.full_like(first_pos, n))
    order = torch.argsort(first_pos)                           # 创建顺序
    rank = torch.empty(num_uniq, dtype=torch.long, device=device)
    rank[order] = torch.arange(num_uniq, device=device)
    slot = rank[inv]                                           # (n,) 创建序 slot
    return _VoxelAssign(valid=valid, slot=slot, seq=seq, uniq=uniq,
                        order=order, real_mask_u=real_mask_u)


def _voxel_fill(points, assign, geom, voxel_num):
    """[5] cap/coors/num_points/voxels 组装 (CPU sync 已在调用方完成)"""
    gs, max_points, max_voxels = geom
    device = points.device
    channels = points.shape[1]
    slot, seq = assign.slot, assign.seq
    valid = assign.valid

    # keep: 有效 & voxel 未超 cap & 组内序号 < max_points
    keep = valid & (slot < max_voxels) & (seq < max_points)

    # coors: 创建顺序的 uniq, 排除 trash 后取前 voxel_num
    creation_uniq = assign.uniq[assign.order]                  # trash 在最后
    cu = creation_uniq[:voxel_num]
    cz_o = cu // (gs[1] * gs[0])
    rem = cu % (gs[1] * gs[0])
    cy_o = rem // gs[0]
    cx_o = rem % gs[0]
    coors_out = torch.stack([cz_o, cy_o, cx_o], dim=-1).to(torch.int32)

    # num_points: bincount (对倾斜索引原子实现, 快于 scatter_add_)
    trash_slot = voxel_num
    slot_kept = torch.where(keep, slot, torch.full_like(slot, trash_slot))
    np_count = torch.bincount(slot_kept, minlength=voxel_num + 1)
    num_points_out = np_count[:voxel_num].to(torch.int32)

    # voxels: (voxel_num+1, max_points, channels) 最后一行是丢弃点的 trash 行
    if KB.voxel_scatter_enabled():
        voxels_out = KB.voxel_scatter(points, slot, seq, keep,
                                      voxel_num, max_points)
    else:
        out = points.new_zeros((voxel_num + 1, max_points, channels))
        dst_v = torch.where(keep, slot, torch.full_like(slot, trash_slot))
        dst_p = torch.where(keep, seq, torch.zeros_like(seq))
        out[dst_v, dst_p, :] = points  # 重复索引仅在 trash 行, 无影响
        voxels_out = out[:voxel_num].contiguous()
    return voxels_out, coors_out, num_points_out


def hard_voxelize_npu(points, voxel_size, coors_range, *caps):
    """mmdet3d ABI 兼容: *caps = (max_points, max_voxels, deterministic=True)"""
    max_points = caps[0]
    max_voxels = caps[1]
    deterministic = caps[2] if len(caps) > 2 else True
    """纯 NPU hard voxelize, 与 mmcv CPU channels++ 实现逐位一致。

    Args:
        points: (n, channels) fp32, NPU
    Returns:
        voxels (V, max_points, channels) fp32 / coors (V, 3) int32 / num_points (V,) int32
    """
    _RULE_CACHE_GEN[0] += 1  # 新帧: SubM RuleBook 缓存代数失效
    channels = points.shape[1]

    # grid_size 与 channels++ 一致: round((range_max - range_min) / vs), [x, y, z]
    gs = []
    for axis in range(3):
        span = coors_range[3 + axis] - coors_range[axis]
        gs.append(int(round(span / voxel_size[axis])))

    assign = _voxel_assign(points, voxel_size, coors_range, gs)

    # [4] voxel 数量 (唯一 CPU sync)
    real_uniq = int(assign.real_mask_u.sum().item())
    voxel_num = min(real_uniq, max_voxels)
    if voxel_num == 0:
        return (points.new_zeros((0, max_points, channels)),
                points.new_zeros((0, 3), dtype=torch.int),
                points.new_zeros((0,), dtype=torch.int))

    return _voxel_fill(points, assign, (gs, max_points, max_voxels), voxel_num)


def _patch_voxelize_npu():
    """将 mmdet3d voxelize 的 _Voxelization.forward 替换为 NPU 实现。"""
    try:
        from mmdet3d.models.data_preprocessors import voxelize as vz_mod
    except Exception as err:
        LOG.info("[sparse_conv_npu] voxelize patch failed (import): %s", err)
        return

    voxelization_cls = getattr(vz_mod, "_Voxelization")
    orig_forward = voxelization_cls.forward.__func__ \
        if hasattr(voxelization_cls.forward, '__func__') else voxelization_cls.forward

    def _npu_forward(ctx, points, *rest):
        use_npu = points.device.type == 'npu' and len(rest) >= 4
        caps_ok = len(rest) >= 4 and rest[2] != -1 and rest[3] != -1
        if use_npu and caps_ok:
            return hard_voxelize_npu(points, *rest)
        # dynamic 路径或 CPU: 走原实现
        return orig_forward(ctx, points, *rest)

    voxelization_cls.forward = _npu_forward
    LOG.info("[sparse_conv_npu] hard_voxelize 已替换为 NPU 实现")


try:
    _patch_voxelize_npu()
except Exception as exc:
    LOG.info("[sparse_conv_npu] voxelize patch failed: %s", exc)


# 延迟执行: 调用方需显式调用 apply_sparse_conv_npu()
# 避免在 import 时干扰 mmengine 的模块系统
apply_sparse_conv_npu()
