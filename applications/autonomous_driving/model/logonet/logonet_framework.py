# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
LoGoNetFramework

功能：
    LoGoNet 融合框架调度层（不包含融合逻辑细节）

职责：
    1. 调用图像特征提取接口生成 image features
    2. 调用点云感知接口生成点云特征 / voxel 信息 / 3D proposals
    3. 调用各阶段模块完成：
        - voxel 投影
        - 全局融合
        - ROI 网格池化
        - 位置编码
        - 局部融合
        - 特征动态聚合
        - 最终候选框分类与回归
    4. 提供 train / test 接口

注意：
    本文件不直接调用任何 fusionrepo 算子
    所有融合逻辑封装在 module 中
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone 接口
from backbone.image_feature_interface import ImageFeatureInterface
from backbone.point_cloud_perception_interface import PointCloudPerceptionInterface

# 模块
from model.logonet.module.voxel_projection_module import VoxelProjectionModule
from model.logonet.module.global_fusion_module import GlobalFusionModule
from model.logonet.module.roi_grid_pool_module import RoIGridPoolModule
from model.logonet.module.position_encoding_module import PositionEncodingModule
from model.logonet.module.local_fusion_module import LocalFusionModule
from model.logonet.module.feature_aggregation_module import FeatureAggregationModule
from model.logonet.module.roi_refine_module import RoIRefineModule


class LoGoNetFramework(nn.Module):

    def __init__(
        self,
        image_backbone: ImageFeatureInterface,
        point_cloud_backbone: PointCloudPerceptionInterface,
        lidar_channels: int,
        image_channels: int,
        pos_channels: int,
        roi_grid_size: int = 6,
        roi_pool_mode: str = "mean",
        num_heads: int = 4,
        ffn_ratio: int = 4,
        num_classes: int = 1,
        box_dim: int = 7
    ):
        super().__init__()

        self.image_backbone = image_backbone
        self.point_cloud_backbone = point_cloud_backbone

        self.voxel_projection_module = VoxelProjectionModule()
       
        self.global_fusion_module = GlobalFusionModule(
            lidar_channels=lidar_channels,
            image_channels=image_channels
        )

        self.roi_grid_pool_module = RoIGridPoolModule(
            grid_size=roi_grid_size,
            pool_mode=roi_pool_mode
        )

        self.position_encoding_module = PositionEncodingModule(
            out_channels=pos_channels
        )

        self.local_fusion_module = LocalFusionModule(
            pos_channels=pos_channels,
            image_channels=image_channels,
            out_channels=pos_channels
        )

        self.feature_aggregation_module = FeatureAggregationModule(
            channels=pos_channels,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio
        )

        self.roi_refine_module = RoIRefineModule(
            in_channels=pos_channels,
            num_classes=num_classes,
            box_dim=box_dim
        )

    def train_forward(self, batch: dict):
        return self._forward_impl(batch=batch, rescale=False)

    @torch.no_grad()
    def test_forward(self, batch: dict):
        self.eval()
        return self._forward_impl(batch=batch, rescale=True)

    def _forward_impl(self, batch: dict, rescale: bool):
        images = batch["images"]
        points = batch["points"]
        img_metas = batch.get("img_metas", None)
        proj_matrix = batch["proj_matrix"]

        # 1-3) 图像特征提取 + 点云感知 + voxel 投影
        pc_inputs = {"img_metas": img_metas, "proj_matrix": proj_matrix,
                     "rescale": rescale}
        image_feat, pc_out, voxel_batch_indices, voxel_centroids, proj_points_2d = \
            self._extract_multimodal_features(
                images=images, points=points, pc_inputs=pc_inputs)

        # 4-5) 全局融合 + ROI 网格池化
        fusion_inputs = {"proj_points": proj_points_2d,
                         "batch_indices": voxel_batch_indices,
                         "voxel_centroids": voxel_centroids}
        grid_points, roi_centers, grid_point_counts, pooled_features = \
            self._run_global_fusion_and_pool(
                pc_out=pc_out, image_feat=image_feat,
                fusion_inputs=fusion_inputs)

        # 6) 位置编码
        pos_feat = self.position_encoding_module(
            grid_centers=grid_points,
            roi_centers=roi_centers,
            grid_point_counts=grid_point_counts
        )["pos_feat"]

        # 7) ROI 网格点投影到图像（按 batch 分开投影）
        grid_points_2d = self._project_grid_points_to_image(
            grid_points=grid_points,
            proj_matrix=proj_matrix,
            batch_indices=pc_out["flat_roi_batch_indices"],
            img_metas=img_metas
        )

        # 8) 局部融合
        local_fused_feat = self.local_fusion_module(
            pos_feat=pos_feat,
            image_feat=image_feat,
            grid_points_2d=grid_points_2d,
            batch_indices=pc_out["flat_roi_batch_indices"]
        )

        # 9) 特征动态聚合（FDA）
        global_roi_feat = pooled_features.unsqueeze(1).expand(-1, local_fused_feat.shape[1], -1)
        fused_feat = global_roi_feat + local_fused_feat + pos_feat
        aggregated_feat = self.feature_aggregation_module(
            fused_feat=fused_feat
        )

        # 10) 最终分类与回归
        refine_out = self.roi_refine_module(
            roi_feat=aggregated_feat
        )

        return {
            "cls_score": refine_out["cls_score"],
            "reg_offset": refine_out["reg_offset"],
            "flat_rois": pc_out["flat_rois"],
            "flat_roi_batch_indices": pc_out["flat_roi_batch_indices"],
            "roi_scores": pc_out.get("roi_scores", None),
            "roi_labels": pc_out.get("roi_labels", None)
        }

    def _extract_multimodal_features(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        pc_inputs: dict,
    ):
        """
        图像特征提取 + 点云感知 + voxel 投影。

        返回:
            image_feat: 图像特征图
            pc_out: 点云感知输出字典
            voxel_batch_indices: (V,) voxel 所属 batch 索引（可能为 None）
            voxel_centroids: voxel 质心坐标
            proj_points_2d: voxel 投影到图像的像素坐标
        """
        # 1) 图像特征提取
        image_feat = self.image_backbone(images)

        # 2) 点云感知（点云特征提取 + 3D proposals）
        pc_out = self.point_cloud_backbone(
            points=points,
            img=images,
            img_metas=pc_inputs["img_metas"],
            rescale=pc_inputs["rescale"]
        )

        voxel_coords = pc_out.get("voxel_coords", None)
        voxel_batch_indices = voxel_coords[:, 0] if voxel_coords is not None else None

        # 3) voxel 投影
        voxel_proj_out = self.voxel_projection_module(
            voxel_points=pc_out["voxels"],
            voxel_num_points=pc_out["voxel_num_points"],
            proj_matrix=pc_inputs["proj_matrix"],
            batch_indices=voxel_batch_indices
        )

        return (
            image_feat,
            pc_out,
            voxel_batch_indices,
            voxel_proj_out["voxel_centroids"],
            voxel_proj_out["proj_points_2d"]
        )

    def _run_global_fusion_and_pool(
        self,
        pc_out: dict,
        image_feat: torch.Tensor,
        fusion_inputs: dict,
    ):
        """
        全局融合 + ROI 网格池化。

        返回:
            grid_points: (K, G, 3)
            roi_centers: (K, 3)
            grid_point_counts: (K, G)
            pooled_features: (K, C)
        """
        # 4) 全局融合
        global_fused_feat = self.global_fusion_module(
            lidar_feat=pc_out["pts_feats"],
            image_feat=image_feat,
            proj_points=fusion_inputs["proj_points"],
            batch_indices=fusion_inputs["batch_indices"]
        )

        # 5) ROI 网格池化
        roi_pool_out = self.roi_grid_pool_module(
            rois=pc_out["flat_rois"],
            point_coords=fusion_inputs["voxel_centroids"],
            point_features=global_fused_feat
        )

        return (
            roi_pool_out["grid_points"],
            roi_pool_out["roi_centers"],
            roi_pool_out["grid_point_counts"],
            roi_pool_out["pooled_features"]
        )

    def _project_grid_points_to_image(
        self,
        grid_points: torch.Tensor,
        proj_matrix: torch.Tensor,
        batch_indices: torch.Tensor,
        img_metas
    ) -> torch.Tensor:
        """
        ROI 网格点投影到图像平面，并 clamp 到图像边界，
        避免 grid_sample 外推。
        """
        grid_points_2d = self._project_points_by_batch(
            points_3d=grid_points,
            proj_matrix=proj_matrix,
            batch_indices=batch_indices
        )

        img_shape = img_metas[0].get("img_shape", (375, 1242)) if img_metas else (375, 1242)
        img_h, img_w = float(img_shape[0]), float(img_shape[1])
        grid_points_2d[..., 0] = grid_points_2d[..., 0].clamp(min=-1.0, max=img_w)
        grid_points_2d[..., 1] = grid_points_2d[..., 1].clamp(min=-1.0, max=img_h)

        return grid_points_2d

    def _project_points_by_batch(
        self,
        points_3d: torch.Tensor,
        proj_matrix: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        按 batch 分开调用 points_to_image，避免 (N,3) 点与 (B,3,4) 投影矩阵直接混用。
        输入:
            points_3d:      (N, 3) 或 (K, G, 3)
            proj_matrix:    (3,4) or (B,3,4)
            batch_indices:  当 points_3d 为 (N,3) 时是 (N,)
                            当 points_3d 为 (K,G,3) 时是 (K,)
        输出:
            points_2d:      与输入前两维保持一致，最后一维为 2
        """
        if points_3d.dim() == 2:
            if points_3d.shape[-1] != 3:
                raise ValueError(f"points_3d must end with 3 channels, got {tuple(points_3d.shape)}")

            if points_3d.numel() == 0:
                return points_3d.new_zeros((0, 2))

            if proj_matrix.dim() == 2:
                return torch.ops.fusionrepo.points_to_image(points_3d, proj_matrix)

            if proj_matrix.dim() != 3:
                raise ValueError(f"proj_matrix must be (3,4) or (B,3,4), got {tuple(proj_matrix.shape)}")

            out = points_3d.new_zeros((points_3d.shape[0], 2))
            unique_bs = torch.unique(batch_indices)

            for b in unique_bs.tolist():
                mask = (batch_indices == b)
                if mask.any():
                    out[mask] = torch.ops.fusionrepo.points_to_image(
                        points_3d[mask],
                        proj_matrix[int(b)]
                    )
            return out

        if points_3d.dim() == 3:
            if points_3d.shape[-1] != 3:
                raise ValueError(f"points_3d must end with 3 channels, got {tuple(points_3d.shape)}")

            k, g, _ = points_3d.shape
            if points_3d.numel() == 0:
                return points_3d.new_zeros((k, g, 2))

            pts_flat = points_3d.reshape(-1, 3)

            if proj_matrix.dim() == 2:
                out_flat = torch.ops.fusionrepo.points_to_image(pts_flat, proj_matrix)
                return out_flat.reshape(k, g, 2)

            if proj_matrix.dim() != 3:
                raise ValueError(f"proj_matrix must be (3,4) or (B,3,4), got {tuple(proj_matrix.shape)}")

            if batch_indices.dim() != 1 or batch_indices.shape[0] != k:
                raise ValueError("batch_indices must have shape (K,) when points_3d is (K, G, 3)")

            out = points_3d.new_zeros((k, g, 2))
            unique_bs = torch.unique(batch_indices)
            for b in unique_bs.tolist():
                roi_mask = (batch_indices == b)
                if roi_mask.any():
                    pts_b = points_3d[roi_mask].reshape(-1, 3)
                    out_b = torch.ops.fusionrepo.points_to_image(pts_b, proj_matrix[int(b)])
                    out[roi_mask] = out_b.reshape(-1, g, 2)
            return out

        raise ValueError(f"points_3d must be (N,3) or (K,G,3), got {tuple(points_3d.shape)}")
