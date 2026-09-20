# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Point Cloud Perception Interface

用于统一调用点云感知后端，使 LoGoNet framework
能够无缝接入不同来源的 3D 检测器 / 点云 backbone
（当前主要适配 MMDetection3D 的 PV-RCNN）。

设计目标：
    1. 统一“点云特征提取 + 3D 框提取”接口
    2. 与后续 framework / module 输入输出格式对齐
    3. framework 中只调用 interface，不写底层 detector 细节

标准输出字段：
    points:
        list[Tensor]，每个 batch 一个点云 Tensor，shape 例如 (Ni, C)

    voxels:
        Tensor or None
        体素内点坐标张量，shape 例如 (M, T, 3)

    voxel_num_points:
        Tensor or None
        每个 voxel 内的有效点数，shape 例如 (M,)

    voxel_coords:
        Tensor or None
        voxel 坐标，通常为 (M, 4)，格式一般是 [batch_idx, z, y, x]

    pts_feats:
        Tensor
        与 voxel_coords 一一对应的 learned voxel features，shape (M, C)

    rois:
        list[Tensor]
        每个 batch 的 3D 框，shape 为 (Ki, box_dim)

    roi_scores:
        list[Tensor]
        每个 batch 的 3D 框得分，shape 为 (Ki,)

    roi_labels:
        list[Tensor]
        每个 batch 的 3D 框类别，shape 为 (Ki,)

    flat_rois:
        Tensor
        所有 batch 的 3D 框拼接结果，shape 为 (K_total, box_dim)

    flat_roi_batch_indices:
        Tensor
        与 flat_rois 对齐的 batch 索引，shape 为 (K_total,)

    raw_predictions:
        Any
        后端原始预测结果，保留便于后续调试
"""

from __future__ import annotations

from typing import NamedTuple, Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode



class DetectorInputs(NamedTuple):
    """检测器前向输入三件套 (batch_inputs_dict / batch_data_samples / points_list)"""
    batch_inputs_dict: Dict[str, Any]
    batch_data_samples: List[Det3DDataSample]
    points_list: List[torch.Tensor]


class PointCloudPerceptionInterface(nn.Module):
    """
    点云感知接口。

    参数:
        detector:
            外部点云感知后端，当前建议直接传入 MMDetection3D detector

        feat_index:
            如果后端返回 list/tuple 且你想固定取某层，可以指定
            若为 None，则自动按“与 voxel_coords 对齐”规则挑选

        use_voxelize:
            是否尝试使用后端 voxelize / data_preprocessor 的体素化结果

        prediction_branch:
            从预测结果里取哪一个分支
            默认优先使用 'pts_bbox'
    """

    def __init__(
        self,
        detector: nn.Module,
        feat_index: Optional[int] = None,
        use_voxelize: bool = True,
        prediction_branch: str = "pts_bbox",
        allowed_labels: Optional[Sequence[int]] = None
    ):
        super().__init__()
        self.detector = detector
        self.feat_index = feat_index
        self.use_voxelize = use_voxelize
        self.prediction_branch = prediction_branch
        self.allowed_labels = None if allowed_labels is None else [int(x) for x in allowed_labels]

    # ============================================================
    # 1) 基础工具
    def forward(
        self,
        points: Sequence[torch.Tensor] | torch.Tensor,
        img: Optional[torch.Tensor] = None,
        img_metas: Optional[List[dict]] = None,
        rescale: bool = False
    ) -> Dict[str, Any]:
        """
        统一执行：
            点云特征提取 + 3D 框提取

        返回一个标准字典，供后续 framework / module 直接调用。
        """
        det_inputs = self._prepare_detector_inputs(
            points=points,
            img_metas=img_metas
        )
        points_list = det_inputs.points_list
        batch_inputs_dict = det_inputs.batch_inputs_dict
        batch_data_samples = det_inputs.batch_data_samples

        voxels, voxel_num_points, voxel_coords = self._extract_voxel_tensors_from_inputs(batch_inputs_dict)

        if voxels is None or voxel_num_points is None or voxel_coords is None:
            raise ValueError(
                "Failed to get voxel tensors from detector inputs. "
                "LoGoNet requires voxels / voxel_num_points / voxel_coords."
            )

        # 这里返回的 pts_feats 是与 voxel_coords 对齐的 learned voxel feature：(M, C)
        pts_feats = self.extract_point_features(
            batch_inputs_dict=batch_inputs_dict,
            voxel_coords=voxel_coords
        )

        raw_predictions = self.extract_3d_boxes(
            det_inputs,
            img=img,
            img_metas=img_metas,
            rescale=rescale
        )

        parsed_predictions = self.parse_3d_predictions(raw_predictions)

        rois, roi_scores, roi_labels, flat_rois, flat_roi_batch_indices = \
            self._standardize_predictions(parsed_predictions)

        return {
            "points": points_list,
            "voxels": voxels,
            "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords,

            # 关键：与 voxel_coords 对齐的 learned voxel feature
            "pts_feats": pts_feats,

            "rois": rois,
            "roi_scores": roi_scores,
            "roi_labels": roi_labels,
            "flat_rois": flat_rois,
            "flat_roi_batch_indices": flat_roi_batch_indices,
            "raw_predictions": raw_predictions
        }

    def extract_point_features(
        self,
        batch_inputs_dict: Dict[str, Any],
        voxel_coords: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        这里返回的不是“任意 detector raw feats”，而是：
            与 voxel_coords 一一对应的 learned voxel feature，shape=(M, C)

        这才是 LoGoNet 全局融合阶段真正应当使用的 lidar_feat。
        """
        if voxel_coords is None:
            raise ValueError("voxel_coords is required for learned voxel feature alignment")

        raw_feats = self.detector.extract_feat(batch_inputs_dict)
        num_voxels = int(voxel_coords.shape[0])

        voxel_feats = self._choose_aligned_voxel_feature(
            raw_feats=raw_feats,
            num_voxels=num_voxels
        )
        return voxel_feats

    # ============================================================
    # 4) 3D 框提取
    # ============================================================
    @torch.no_grad()
    def extract_3d_boxes(
        self,
        det_inputs: DetectorInputs,
        img=None,
        img_metas=None,
        rescale: bool = False
    ):
        """
        使用当前 mmdet3d detector 做 3D 框预测。
        优先适配新版 mmdet3d：predict / test_step。
        """
        batch_inputs_dict = det_inputs.batch_inputs_dict
        batch_data_samples = det_inputs.batch_data_samples
        points_list = det_inputs.points_list
        if hasattr(self.detector, "predict"):
            preds = self.detector.predict(
                batch_inputs_dict,
                batch_data_samples=batch_data_samples
            )
            return preds

        if hasattr(self.detector, "test_step"):
            data = {
                "inputs": batch_inputs_dict,
                "data_samples": batch_data_samples
            }
            preds = self.detector.test_step(data)
            return preds

        if hasattr(self.detector, "simple_test"):
            return self.detector.simple_test(points_list, img_metas, img=img, rescale=rescale)

        raise AttributeError(
            "detector must provide predict(...) or test_step(...) or simple_test(...)"
        )

    # ============================================================
    # 5) 预测结果解析
    # ============================================================
    def parse_3d_predictions(self, raw_predictions):
        """
        统一解析 mmdet3d 新版预测结果，提取：
        - boxes_3d
        - scores_3d
        - labels_3d
        """
        if raw_predictions is None:
            return []

        parsed = []

        for pred in raw_predictions:
            # 新版 mmdet3d 常见返回：Det3DDataSample
            if hasattr(pred, "pred_instances_3d"):
                inst = pred.pred_instances_3d

                boxes = inst.bboxes_3d.tensor if hasattr(inst.bboxes_3d, "tensor") else inst.bboxes_3d
                scores = inst.scores_3d
                labels = inst.labels_3d

                parsed.append({
                    "boxes_3d": boxes,
                    "scores_3d": scores,
                    "labels_3d": labels
                })
                continue

            # 兼容字典形式
            if isinstance(pred, dict):
                if "pred_instances_3d" in pred:
                    inst = pred["pred_instances_3d"]
                    boxes = inst.bboxes_3d.tensor if hasattr(inst.bboxes_3d, "tensor") else inst.bboxes_3d
                    scores = inst.scores_3d
                    labels = inst.labels_3d
                    parsed.append({
                        "boxes_3d": boxes,
                        "scores_3d": scores,
                        "labels_3d": labels
                    })
                    continue

                if "boxes_3d" in pred:
                    parsed.append({
                        "boxes_3d": pred["boxes_3d"],
                        "scores_3d": pred["scores_3d"],
                        "labels_3d": pred["labels_3d"]
                    })
                    continue

            raise TypeError(f"Unsupported prediction format: {type(pred)}")

        return parsed

    # ============================================================
    # 6) 预测结果格式统一
    # ============================================================
    # ============================================================
    def _normalize_points_input(
        self,
        points: Sequence[torch.Tensor] | torch.Tensor
    ) -> List[torch.Tensor]:
        """统一为 list[Tensor]，每个 batch 一个点云。"""
        if torch.is_tensor(points):
            return [points]

        if isinstance(points, (list, tuple)):
            if len(points) == 0:
                raise ValueError("points cannot be empty")
            if not all(torch.is_tensor(p) for p in points):
                raise TypeError("each element in points must be a Tensor")
            return list(points)

        raise TypeError("points must be Tensor or list/tuple of Tensor")

    def _build_batch_data_samples(
        self,
        points_list: List[torch.Tensor],
        img_metas: Optional[List[dict]] = None
    ) -> List[Det3DDataSample]:
        """
        为新版 mmdet3d predict()/test_step() 构造最小可用的 batch_data_samples。
        """
        batch_data_samples: List[Det3DDataSample] = []

        for i in range(len(points_list)):
            ds = Det3DDataSample()

            meta: Dict[str, Any] = {}
            if img_metas is not None and i < len(img_metas) and img_metas[i] is not None:
                meta.update(img_metas[i])

            # mmdet3d 预测阶段后处理常用字段
            if "box_type_3d" not in meta:
                meta["box_type_3d"] = LiDARInstance3DBoxes
            if "box_mode_3d" not in meta:
                meta["box_mode_3d"] = Box3DMode.LIDAR

            ds.set_metainfo(meta)
            batch_data_samples.append(ds)

        return batch_data_samples

    def _prepare_detector_inputs(
        self,
        points: Sequence[torch.Tensor] | torch.Tensor,
        img_metas: Optional[List[dict]] = None
    ) -> DetectorInputs:
        """
        统一构造 detector 需要的：
            points_list
            batch_inputs_dict
            batch_data_samples
        并在可用时走 data_preprocessor，拿到 mmdet3d 真正会用的 inputs。
        """
        points_list = self._normalize_points_input(points)

        batch_inputs_dict: Dict[str, Any] = {
            "points": points_list
        }

        batch_data_samples = self._build_batch_data_samples(
            points_list=points_list,
            img_metas=img_metas
        )

        if hasattr(self.detector, "data_preprocessor"):
            data = {
                "inputs": batch_inputs_dict,
                "data_samples": batch_data_samples
            }
            processed = self.detector.data_preprocessor(data, training=False)
            batch_inputs_dict = processed["inputs"]
            batch_data_samples = processed["data_samples"]

        return DetectorInputs(
            batch_inputs_dict=batch_inputs_dict,
            batch_data_samples=batch_data_samples,
            points_list=points_list
        )

    # ============================================================
    # 2) 体素信息提取
    # ============================================================
    def _extract_voxel_tensors_from_inputs(
        self,
        batch_inputs_dict: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        从 data_preprocessor 处理后的 batch_inputs_dict 中提取：
            voxels, voxel_num_points, voxel_coords

        对 PV-RCNN / mmdet3d 常见结构，batch_inputs_dict["voxels"] 往往是一个 dict：
            {
                "voxels": Tensor(M, T, C),
                "coors": Tensor(M, 4),
                "num_points": Tensor(M,),
                ...
            }

        LoGoNet 后续 voxel_projection_module 只需要体素内点坐标，所以这里输出：
            voxels[..., :3]
        """
        if not self.use_voxelize:
            return None, None, None

        if "voxels" not in batch_inputs_dict:
            return None, None, None

        voxel_dict = batch_inputs_dict["voxels"]

        # mmdet3d 新版常见：dict
        if isinstance(voxel_dict, dict):
            raw_voxels = voxel_dict.get("voxels", None)
            voxel_coords = voxel_dict.get("coors", None)
            voxel_num_points = voxel_dict.get("num_points", None)

            if raw_voxels is None:
                return None, voxel_num_points, voxel_coords

            if raw_voxels.dim() != 3:
                raise ValueError(f"Expected voxel tensor with dim=3, got shape={tuple(raw_voxels.shape)}")

            if raw_voxels.shape[-1] < 3:
                raise ValueError(
                    f"Expected raw_voxels.shape[-1] >= 3, got shape={tuple(raw_voxels.shape)}"
                )

            voxels = raw_voxels[..., :3].contiguous()
            return voxels, voxel_num_points, voxel_coords

        # 兼容极少数旧版直接 tuple/list 输出
        if isinstance(voxel_dict, (list, tuple)) and len(voxel_dict) == 3:
            raw_voxels, voxel_num_points, voxel_coords = voxel_dict
            if raw_voxels is not None:
                if raw_voxels.dim() != 3:
                    raise ValueError(f"Expected voxel tensor with dim=3, got shape={tuple(raw_voxels.shape)}")
                if raw_voxels.shape[-1] < 3:
                    raise ValueError(
                        f"Expected raw_voxels.shape[-1] >= 3, got shape={tuple(raw_voxels.shape)}"
                    )
                raw_voxels = raw_voxels[..., :3].contiguous()
            return raw_voxels, voxel_num_points, voxel_coords

        return None, None, None

    # ============================================================
    # 3) learned voxel feature 提取（追论文精度版本）
    # ============================================================
    def _collect_candidate_tensors(
        self,
        obj: Any,
        prefix: str = ""
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        递归收集 raw feature 里的候选 tensor。
        目标是在 mmdet3d 返回的复杂结构中，找出可能与 voxel_coords 对齐的 (M, C) 特征。
        """
        candidates: List[Tuple[str, torch.Tensor]] = []

        if torch.is_tensor(obj):
            candidates.append((prefix or "tensor", obj))
            return candidates

        # SparseConvTensor / spconv tensor
        if hasattr(obj, "features") and torch.is_tensor(obj.features):
            candidates.append((prefix + ".features" if prefix else "features", obj.features))

        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}.{k}" if prefix else str(k)
                candidates.extend(self._collect_candidate_tensors(v, name))
            return candidates

        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                name = f"{prefix}[{i}]" if prefix else f"[{i}]"
                candidates.extend(self._collect_candidate_tensors(v, name))
            return candidates

        return candidates

    def _choose_aligned_voxel_feature(
        self,
        raw_feats: Any,
        num_voxels: int
    ) -> torch.Tensor:
        """
        从 detector.extract_feat(...) 的原始输出中，自动挑出与 voxel_coords 对齐的 learned feature。

        规则：
            1. 只接受 Tensor，且第一维 == num_voxels
            2. 优先选择二维 Tensor (M, C)
            3. 优先 key/name 中带以下语义的候选：
               voxel / pillar / point / feat / feature / x
            4. 如果 feat_index 被显式指定，则在候选列表中按 index 取
        """
        candidates = self._collect_candidate_tensors(raw_feats)

        matched: List[Tuple[str, torch.Tensor]] = []
        for name, tensor in candidates:
            if not torch.is_tensor(tensor):
                continue
            if tensor.dim() < 2:
                continue
            if tensor.shape[0] != num_voxels:
                continue
            matched.append((name, tensor))

        if len(matched) == 0:
            raise ValueError(
                f"Cannot find learned lidar feature aligned with voxel_coords. "
                f"num_voxels={num_voxels}, raw_feats_type={type(raw_feats)}"
            )

        # 若显式指定 feat_index，则按匹配候选顺序取
        if self.feat_index is not None:
            if self.feat_index >= len(matched):
                raise IndexError(
                    f"feat_index={self.feat_index} out of range for matched voxel features ({len(matched)})"
                )
            feat = matched[self.feat_index][1]
            if feat.dim() > 2:
                feat = feat.reshape(feat.shape[0], -1)
            return feat.contiguous()

        # 否则优先按名字语义选择
        priority_keywords = ["voxel", "pillar", "point", "feat", "feature", "x"]

        def score_name(name: str) -> int:
            name_l = name.lower()
            score = 0
            for i, kw in enumerate(priority_keywords):
                if kw in name_l:
                    score += (len(priority_keywords) - i) * 10
            return score

        matched.sort(key=lambda x: (score_name(x[0]), x[1].dim()), reverse=True)
        feat = matched[0][1]

        if feat.dim() > 2:
            feat = feat.reshape(feat.shape[0], -1)

        return feat.contiguous()

    def _extract_branch_result(self, pred_item: Any) -> Dict[str, Any]:
        """统一从单个 batch 的预测结果中取出 3D 框分支。"""
        if isinstance(pred_item, dict):
            if self.prediction_branch in pred_item and isinstance(pred_item[self.prediction_branch], dict):
                return pred_item[self.prediction_branch]
            return pred_item

        raise TypeError("prediction item must be dict")

    def _boxes_to_tensor(self, boxes_3d: Any) -> torch.Tensor:
        """兼容 BaseInstance3DBoxes 或 Tensor。"""
        if torch.is_tensor(boxes_3d):
            return boxes_3d

        if hasattr(boxes_3d, "tensor") and torch.is_tensor(boxes_3d.tensor):
            return boxes_3d.tensor

        raise TypeError("boxes_3d must be Tensor or object with .tensor")

    def _standardize_predictions(
        self,
        predictions: Any
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        标准化输出为：
            rois, roi_scores, roi_labels, flat_rois, flat_roi_batch_indices
        """
        if not isinstance(predictions, (list, tuple)):
            raise TypeError("predictions must be list/tuple")

        rois: List[torch.Tensor] = []
        roi_scores: List[torch.Tensor] = []
        roi_labels: List[torch.Tensor] = []
        flat_rois_list: List[torch.Tensor] = []
        flat_batch_idx_list: List[torch.Tensor] = []

        for b, item in enumerate(predictions):
            branch = self._extract_branch_result(item)

            boxes_3d = self._boxes_to_tensor(branch["boxes_3d"])
            scores_3d = branch["scores_3d"]
            labels_3d = branch["labels_3d"]

            if self.allowed_labels is not None:
                keep = torch.zeros_like(labels_3d, dtype=torch.bool)
                for cls_id in self.allowed_labels:
                    keep |= (labels_3d == int(cls_id))
                boxes_3d = boxes_3d[keep]
                scores_3d = scores_3d[keep]
                labels_3d = labels_3d[keep]

            rois.append(boxes_3d)
            roi_scores.append(scores_3d)
            roi_labels.append(labels_3d)

            if boxes_3d.numel() > 0:
                flat_rois_list.append(boxes_3d)
                flat_batch_idx_list.append(
                    torch.full(
                        (boxes_3d.shape[0],),
                        b,
                        dtype=torch.long,
                        device=boxes_3d.device
                    )
                )

        if len(flat_rois_list) > 0:
            flat_rois = torch.cat(flat_rois_list, dim=0)
            flat_roi_batch_indices = torch.cat(flat_batch_idx_list, dim=0)
        else:
            device = rois[0].device if len(rois) > 0 else torch.device("cpu")
            flat_rois = torch.empty((0, 7), device=device)
            flat_roi_batch_indices = torch.empty((0,), dtype=torch.long, device=device)

        return rois, roi_scores, roi_labels, flat_rois, flat_roi_batch_indices

    # ============================================================
    # 7) 总接口
    # ============================================================
