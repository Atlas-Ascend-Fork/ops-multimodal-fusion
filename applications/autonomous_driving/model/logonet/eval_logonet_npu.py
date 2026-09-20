# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
LoGoNet NPU 评估脚本.

点云 backbone 为 mmdet3d 官方 PV-RCNN (其稀疏卷积依赖 CUDA C 扩展, NPU 不支持,
由 sparse_conv_npu 在 import 时 patch 为 NPU 实现), 跑通 LoGoNet 融合管线:
voxel projection -> global fusion -> ROI grid pool ->
local fusion -> feature aggregation -> ROI refine.

用法:
  python applications/autonomous_driving/model/logonet/eval_logonet_npu.py \
    --device npu:0 [--max-frames 50]

  需先安装 LoGoNet 官方代码 (utils/common/fusion_unit/backbone/model) 与
  mmdet/mmdet3d/mmcv NPU 环境, 见 applications/autonomous_driving/README.md。
"""
import argparse
import logging
import math
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image  # noqa: F401

AD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AD_ROOT))
sys.path.insert(0, str(AD_ROOT / "common/pcdet_stub"))

import mmdet3d
import mmdet
from mmdet.utils import register_all_modules as register_mmdet
from mmdet3d.utils import register_all_modules as register_mmdet3d
from mmdet.apis import init_detector as init_det2d
from mmdet3d.apis import init_model as init_det3d

import common.ops  # noqa: F401  注册 torch.ops.fusionrepo 自定义算子 (LoGoNet 前向触达)

# Patch: 稀疏卷积 NPU 实现 (须在 mmcv sparse 绑定被使用前生效)
import sparse_conv_npu  # noqa: F401

from backbone.image_feature_interface import ImageFeatureInterface
from backbone.point_cloud_perception_interface import PointCloudPerceptionInterface  # noqa: F401
from model.logonet.logonet_framework import LoGoNetFramework
from model.logonet.initialization.build_logonet_system import (
    LoGoNetPostProcessor,
    test_step,
)
from utils.pcdet_kitti_writer import KittiPredFrame, write_kitti_pred_txt_pcdet

LOG = logging.getLogger(__name__)

# Patch: prevent rotate_iou CUDA kernel import (NPU has no CUDA driver)
_ROTATE_IOU_MOD_PATH = 'mmdet3d.evaluation.functional.kitti_utils.rotate_iou'
if _ROTATE_IOU_MOD_PATH not in sys.modules:
    _stub_module = types.ModuleType(_ROTATE_IOU_MOD_PATH)
    sys.path.insert(0, str(AD_ROOT / "common/kitti_eval"))
    from rotate_iou_cpu import rotate_iou_cpu
    _stub_module.rotate_iou_gpu_eval = rotate_iou_cpu
    sys.modules[_ROTATE_IOU_MOD_PATH] = _stub_module

register_mmdet()
register_mmdet3d()

_PP_CFG = Path(mmdet3d.__file__).parent / \
    ".mim/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
_DET2D_CFG = Path(mmdet.__file__).parent / \
    ".mim/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_1x_coco.py"
PP_CFG = str(_PP_CFG)
PP_CKPT = str(AD_ROOT / "checkpoints/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.pth")

DET2D_CFG = str(_DET2D_CFG)
DET2D_CKPT = str(AD_ROOT / "checkpoints/logonet/"
                 "faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth")
FUSION_CKPT = str(AD_ROOT / "checkpoints/logonet/logonet_fusion_trained_best_loss.pth")

DATA_DIR = AD_ROOT / "data/kitti/training"
RESULTS_DIR = AD_ROOT / "results/logonet"


def read_calib(calib_path):
    data = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = np.array(
                [float(x) for x in v.strip().split()], dtype=np.float32)
    p2_mat = data.get("P2")
    r0_mat = data.get("R0_rect")
    tr_mat = data.get("Tr_velo_to_cam")
    if p2_mat is None or r0_mat is None or tr_mat is None:
        raise KeyError("calib must contain P2/R0_rect/Tr_velo_to_cam")
    return {"P2": p2_mat.reshape(3, 4), "R0_rect": r0_mat.reshape(3, 3),
            "Tr_velo_to_cam": tr_mat.reshape(3, 4)}


def build_lidar_to_image_proj(calib):
    p2_mat = calib["P2"].astype(np.float32)
    r0_mat = calib["R0_rect"].astype(np.float32)
    tr_mat = calib["Tr_velo_to_cam"].astype(np.float32)
    r0_mat_4 = np.eye(4, dtype=np.float32)
    r0_mat_4[:3, :3] = r0_mat
    tr_mat_4 = np.eye(4, dtype=np.float32)
    tr_mat_4[:3, :4] = tr_mat
    return (p2_mat @ r0_mat_4 @ tr_mat_4).astype(np.float32)


def load_image_caffe(img_path, device):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image: {img_path}")
    img = img.astype(np.float32)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    img = (img - mean)
    h, w = img.shape[:2]
    pad_h = int(math.ceil(h / 32) * 32 - h)
    pad_w = int(math.ceil(w / 32) * 32 - w)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    return img_t, (h, w)


class MMDetImageExtractor(torch.nn.Module):
    def __init__(self, cfg_path, ckpt_path, device, feat_index=0):
        super().__init__()
        self.feat_index = feat_index
        from mmengine.registry import init_default_scope
        init_default_scope("mmdet")
        self.model = init_det2d(cfg_path, ckpt_path, device=device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images):
        from mmengine.registry import init_default_scope
        init_default_scope("mmdet")
        feats = self.model.extract_feat(images)
        if isinstance(feats, (list, tuple)):
            return feats[self.feat_index]
        return feats


class PVRcnnAdapter(torch.nn.Module):
    """PV-RCNN 适配器: 提供 LoGoNet 所需的 voxel 特征 + 3D proposals."""

    def __init__(self, detector, allowed_labels=None):
        super().__init__()
        self.detector = detector
        self.allowed_labels = None if allowed_labels is None \
            else [int(x) for x in allowed_labels]

    @torch.no_grad()
    def forward(self, points, img=None, img_metas=None, rescale=False):
        points_list, batch_inputs_dict, ds_list = self._prepare_inputs(
            points, img_metas)
        voxel_dict = batch_inputs_dict.get("voxels", {})
        raw_voxels = voxel_dict.get("voxels")
        voxel_coords = voxel_dict.get("coors")
        voxel_num_points = voxel_dict.get("num_points")
        voxels = raw_voxels[..., :3].contiguous() \
            if raw_voxels is not None else None

        # 3D proposals — RPN only
        if hasattr(self.detector, 'test_cfg'):
            if hasattr(self.detector.test_cfg, 'rpn'):
                self.detector.test_cfg.rpn.score_thr = 0.0
                self.detector.test_cfg.rpn.nms_pre = 1000
        feats_dict = self.detector.extract_feat(batch_inputs_dict)
        preds = self.detector.rpn_head.predict(feats_dict, ds_list)

        fallback_voxel = {"voxels": raw_voxels, "coors": voxel_coords,
                          "num_points": voxel_num_points}
        pts_feats = self._extract_pts_feats(feats_dict, batch_inputs_dict,
                                            fallback_voxel)
        rois, roi_scores, roi_labels = self._collect_rois(preds)

        flat_rois_list, flat_batch_idx_list = [], []
        for b, boxes in enumerate(rois):
            if boxes.numel() > 0:
                flat_rois_list.append(boxes)
                flat_batch_idx_list.append(torch.full(
                    (boxes.shape[0],), b, dtype=torch.long,
                    device=boxes.device))
        if flat_rois_list:
            flat_rois = torch.cat(flat_rois_list, dim=0)
            flat_roi_batch_indices = torch.cat(flat_batch_idx_list, dim=0)
        else:
            dev = points_list[0].device
            flat_rois = torch.empty((0, 7), device=dev)
            flat_roi_batch_indices = torch.empty((0,), dtype=torch.long, device=dev)

        return {
            "points": points_list, "voxels": voxels,
            "voxel_num_points": voxel_num_points, "voxel_coords": voxel_coords,
            "pts_feats": pts_feats,
            "rois": rois, "roi_scores": roi_scores, "roi_labels": roi_labels,
            "flat_rois": flat_rois,
            "flat_roi_batch_indices": flat_roi_batch_indices,
        }

    def _prepare_inputs(self, points, img_metas=None):
        from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
        points_list = [points] if torch.is_tensor(points) else list(points)
        batch_inputs_dict = {"points": points_list}
        ds_list = []
        for i in range(len(points_list)):
            ds = Det3DDataSample()
            meta = {}
            if img_metas and i < len(img_metas) and img_metas[i]:
                meta.update(img_metas[i])
            meta.setdefault("box_type_3d", LiDARInstance3DBoxes)
            meta.setdefault("box_mode_3d", Box3DMode.LIDAR)
            meta.setdefault("ori_shape", (375, 1242))
            meta.setdefault("img_shape", (375, 1242))
            meta.setdefault("pad_shape", (384, 1248))
            meta.setdefault("scale_factor", 1.0)
            meta.setdefault("flip", False)
            ds.set_metainfo(meta)
            ds_list.append(ds)
        if hasattr(self.detector, "data_preprocessor"):
            data = {"inputs": batch_inputs_dict, "data_samples": ds_list}
            processed = self.detector.data_preprocessor(data, training=False)
            batch_inputs_dict = processed["inputs"]
            ds_list = processed["data_samples"]
        # 已知限制: NPU voxelization 与 CPU 存在 ~0.6% coors 差异
        # (0.4ms vs 59.8ms), VFE mean 差异导致 AP=0, 仍走 CPU 体素化
        return points_list, batch_inputs_dict, ds_list

    def _extract_pts_feats(self, feats_dict, batch_inputs_dict, fallback_voxel):
        """pts_feats: 直接取 extract_feat 结果, 避免重复计算 middle_encoder"""
        multi_scale_3d_feats = feats_dict.get('multi_scale_3d_feats', None)
        if multi_scale_3d_feats and len(multi_scale_3d_feats) > 0:
            sp0 = multi_scale_3d_feats[0]
            return sp0.features if hasattr(sp0, 'features') else sp0
        raw_voxels2 = batch_inputs_dict.get("voxels", {}).get(
            "voxels", fallback_voxel["voxels"])
        coors2 = batch_inputs_dict.get("voxels", {}).get(
            "coors", fallback_voxel["coors"])
        npt2 = batch_inputs_dict.get("voxels", {}).get(
            "num_points", fallback_voxel["num_points"])
        vfe_feats = self.detector.voxel_encoder(raw_voxels2, npt2, coors2)
        middle_out = self.detector.middle_encoder(vfe_feats, coors2.int(), 1)
        if isinstance(middle_out, (list, tuple)) and len(middle_out) == 2:
            _, middle_feats_list = middle_out
            sp0 = middle_feats_list[0]
            return sp0.features if hasattr(sp0, 'features') else sp0
        return vfe_feats

    def _collect_rois(self, preds):
        """RPN 预测 → 按 allowed_labels 过滤的 rois/scores/labels"""
        rois, roi_scores, roi_labels = [], [], []
        for pred in preds:
            boxes, scores, labels = _pred_boxes_scores(pred)
            if boxes is None:
                continue
            if self.allowed_labels is not None:
                keep = torch.zeros_like(labels, dtype=torch.bool)
                for cls_id in self.allowed_labels:
                    keep |= (labels == int(cls_id))
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            rois.append(boxes)
            roi_scores.append(scores)
            roi_labels.append(labels)
        return rois, roi_scores, roi_labels


def _pred_boxes_scores(pred):
    """从单个预测项取 (boxes, scores, labels), 无 3D 框时返回 (None,)*3"""
    if hasattr(pred, 'bboxes_3d'):
        boxes = pred.bboxes_3d.tensor if hasattr(pred.bboxes_3d, 'tensor') \
            else pred.bboxes_3d
        scores = pred.scores_3d
        labels = pred.labels_3d if hasattr(pred, 'labels_3d') \
            else torch.zeros(scores.shape, dtype=torch.long, device=scores.device)
        return boxes, scores, labels
    if hasattr(pred, 'pred_instances_3d'):
        inst = pred.pred_instances_3d
        boxes = inst.bboxes_3d.tensor if hasattr(inst.bboxes_3d, 'tensor') \
            else inst.bboxes_3d
        return boxes, inst.scores_3d, inst.labels_3d
    return None, None, None


def _build_detectors(device):
    """构建 2D (Faster R-CNN) + 3D (PV-RCNN) 检测器与适配器"""
    LOG.info("[logonet] 构建 2D 检测器 (Faster R-CNN) ...")
    img_extractor = MMDetImageExtractor(DET2D_CFG, DET2D_CKPT, device, feat_index=0)
    image_iface = ImageFeatureInterface(extractor=img_extractor, feat_index=0)

    LOG.info("[logonet] 构建 3D 检测器 (PV-RCNN) ...")
    det3d = init_det3d(PP_CFG, None, device=device)
    ckpt = torch.load(PP_CKPT, map_location='cpu', weights_only=False)
    missing, unexpected = det3d.load_state_dict(ckpt['state_dict'], strict=False)
    LOG.info("  PV-RCNN checkpoint loaded: %d keys, missing=%d, unexpected=%d",
             len(ckpt['state_dict']), len(missing), len(unexpected))
    det3d.eval()
    pc_iface = PVRcnnAdapter(detector=det3d, allowed_labels=[2])  # Car=2 in 3class
    return image_iface, pc_iface


def _materialize_adapters(model, fusion_sd, device):
    """materialize 延迟初始化的 adapter (权重形状驱动)"""
    gf = getattr(model, "global_fusion_module", None)
    if gf is None:
        return
    lidar_w = fusion_sd.get("global_fusion_module.lidar_input_adapter.weight")
    if lidar_w is not None and getattr(gf, "lidar_input_adapter", None) is None:
        gf.lidar_input_adapter = torch.nn.Linear(
            lidar_w.shape[1], lidar_w.shape[0], bias=True).to(
            device=device, dtype=lidar_w.dtype)
    image_w = fusion_sd.get("global_fusion_module.image_input_adapter.weight")
    if image_w is not None and getattr(gf, "image_input_adapter", None) is None:
        gf.image_input_adapter = torch.nn.Conv2d(
            image_w.shape[1], image_w.shape[0],
            kernel_size=tuple(image_w.shape[2:]), stride=1, padding=0,
            bias=True).to(device=device, dtype=image_w.dtype)


def _load_fusion_weights(model, device):
    """加载融合模块 checkpoint"""
    if not os.path.exists(FUSION_CKPT):
        LOG.info("[WARN] fusion checkpoint not found: %s", FUSION_CKPT)
        return
    LOG.info("[logonet] 加载 fusion checkpoint ...")
    fusion_ckpt = torch.load(FUSION_CKPT, map_location='cpu', weights_only=False)
    fusion_sd = fusion_ckpt.get('state_dict', fusion_ckpt)
    _materialize_adapters(model, fusion_sd, device)
    missing, unexpected = model.load_state_dict(fusion_sd, strict=False)
    LOG.info("  fusion checkpoint loaded: %d keys, missing=%d, unexpected=%d",
             len(fusion_sd), len(missing), len(unexpected))
    if missing:
        LOG.info("  missing sample: %s", missing[:5])
    if unexpected:
        LOG.info("  unexpected sample: %s", unexpected[:5])


def build_logonet_npu(device):
    image_iface, pc_iface = _build_detectors(device)

    LOG.info("[logonet] 构建 LoGoNet 框架 (lidar_channels=128) ...")
    model = LoGoNetFramework(
        image_backbone=image_iface,
        point_cloud_backbone=pc_iface,
        lidar_channels=128,
        image_channels=256,
        pos_channels=128,
        roi_grid_size=6,
        roi_pool_mode="mean",
        num_heads=4,
        ffn_ratio=4,
        num_classes=1,
        box_dim=7,
    ).to(device)
    _load_fusion_weights(model, device)
    model.eval()

    postprocessor = LoGoNetPostProcessor(
        num_classes=1, score_thr=0.0, nms_thr=0.5,
        pre_nms_topk=300, post_nms_topk=100, fuse_roi_score=False,
    ).to(device)
    return types.SimpleNamespace(model=model, postprocessor=postprocessor,
                                 device=device)


def read_kitti_label_as_anno(label_path):
    names, truncated, occluded = [], [], []
    alpha, bbox, dims, loc, ry = [], [], [], [], []
    if not os.path.exists(label_path):
        return dict(
            name=np.array([], dtype=object),
            truncated=np.array([], dtype=np.float32),
            occluded=np.array([], dtype=np.int64),
            alpha=np.array([], dtype=np.float32),
            bbox=np.zeros((0, 4), dtype=np.float32),
            dimensions=np.zeros((0, 3), dtype=np.float32),
            location=np.zeros((0, 3), dtype=np.float32),
            rotation_y=np.array([], dtype=np.float32))
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if not p or p[0] != "Car":
                continue
            names.append(p[0])
            truncated.append(float(p[1]))
            occluded.append(int(p[2]))
            alpha.append(float(p[3]))
            bbox.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
            # KITTI label_2 格式: h w l — kitti_eval 期望 dimensions=[h,w,l]
            dims.append([float(p[8]), float(p[9]), float(p[10])])
            loc.append([float(p[11]), float(p[12]), float(p[13])])
            ry.append(float(p[14]))
    return dict(
        name=np.array(names, dtype=object),
        truncated=np.array(truncated, dtype=np.float32),
        occluded=np.array(occluded, dtype=np.int64),
        alpha=np.array(alpha, dtype=np.float32),
        bbox=np.array(bbox, dtype=np.float32) if bbox
        else np.zeros((0, 4), dtype=np.float32),
        dimensions=np.array(dims, dtype=np.float32) if dims
        else np.zeros((0, 3), dtype=np.float32),
        location=np.array(loc, dtype=np.float32) if loc
        else np.zeros((0, 3), dtype=np.float32),
        rotation_y=np.array(ry, dtype=np.float32))


def run_kitti_eval(gt_annos, dt_annos):
    from mmdet3d.evaluation.functional.kitti_utils.eval import kitti_eval
    for anno in dt_annos:
        if "alpha" not in anno or len(anno.get("alpha", [])) == 0:
            for k in ["name", "truncated", "occluded", "alpha",
                      "rotation_y", "score"]:
                anno[k] = np.array([], dtype=object if k == "name"
                                   else np.float32)
            anno["bbox"] = np.zeros((0, 4), dtype=np.float32)
            anno["dimensions"] = np.zeros((0, 3), dtype=np.float32)
            anno["location"] = np.zeros((0, 3), dtype=np.float32)
    result = kitti_eval(gt_annos, dt_annos, ["Car"],
                        eval_types=["bbox", "bev", "3d"])
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, {}


def _empty_dt_anno():
    return dict(
        name=np.array([], dtype=object),
        truncated=np.array([], dtype=np.float32),
        occluded=np.array([], dtype=np.int64),
        alpha=np.array([], dtype=np.float32),
        bbox=np.zeros((0, 4), dtype=np.float32),
        dimensions=np.zeros((0, 3), dtype=np.float32),
        location=np.zeros((0, 3), dtype=np.float32),
        rotation_y=np.array([], dtype=np.float32),
        score=np.array([], dtype=np.float32))


def _read_dt_anno_with_scores(dt_path):
    """从 pred txt 重建 anno (score 取行尾列)"""
    dt = read_kitti_label_as_anno(dt_path)
    if dt is None:
        return _empty_dt_anno()
    scores_list, names_list = [], []
    with open(dt_path) as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            names_list.append(p[0])
            scores_list.append(float(p[-1]))
    dt["name"] = np.array(names_list, dtype=object)
    dt["score"] = np.array(scores_list, dtype=np.float32)
    return dt


def _infer_one_frame(sys_obj, sid):
    """单帧: 数据加载 → test_step → 底边 z 修正; 返回 (boxes, scores, h, w)"""
    img_path = str(DATA_DIR / "image_2" / f"{sid}.png")
    lidar_path = str(DATA_DIR / "velodyne" / f"{sid}.bin")
    calib_path = str(DATA_DIR / "calib" / f"{sid}.txt")

    calib = read_calib(calib_path)
    img_t, (h, w) = load_image_caffe(img_path, sys_obj.device)
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    pts_t = torch.from_numpy(pts).to(sys_obj.device)
    proj_matrix = torch.from_numpy(
        build_lidar_to_image_proj(calib)).float().to(sys_obj.device)

    batch = {
        "images": img_t, "image": img_t,
        "points": [pts_t], "point": [pts_t],
        "proj_matrix": proj_matrix,
        "img_metas": [{
            "ori_shape": (h, w), "img_shape": (h, w),
            "pad_shape": (img_t.shape[-2], img_t.shape[-1]),
            "scale_factor": 1.0, "flip": False,
        }],
    }
    out = test_step(sys_obj, batch)
    if os.environ.get("EVAL_MEM_DEBUG"):
        LOG.info("  [mem] %s alloc=%.2fGB reserved=%.2fGB", sid,
                 torch.npu.memory_allocated() / 2**30,
                 torch.npu.memory_reserved() / 2**30)
    final_results = out["final_results"]
    if len(final_results) == 0:
        return torch.zeros((0, 7), dtype=torch.float32), \
            torch.zeros((0,), dtype=torch.float32), h, w
    pred0 = final_results[0]
    boxes_lidar = pred0["boxes_3d"].detach().cpu()
    scores = pred0["scores_3d"].detach().cpu()
    # LoGoNet 输出 [x,y,z,dx,dy,dz,yaw] 中心 z, writer 需要 bottom z
    if boxes_lidar.numel() > 0:
        boxes_lidar[:, 2] = boxes_lidar[:, 2] - boxes_lidar[:, 5] / 2.0
    return boxes_lidar, scores, h, w


def _run_inference(sys_obj, val_ids, max_frames):
    pred_dir = str(RESULTS_DIR / "pred")
    os.makedirs(pred_dir + "/data", exist_ok=True)
    n_det = 0
    t0 = time.time()

    for i, sid in enumerate(val_ids):
        try:
            boxes_lidar, scores, h, w = _infer_one_frame(sys_obj, sid)
            n_det += boxes_lidar.shape[0]
            write_kitti_pred_txt_pcdet(
                pred_dir=pred_dir,
                frame=KittiPredFrame(
                    frame_id=sid, boxes_lidar=boxes_lidar,
                    scores=scores, calib_path=str(
                        DATA_DIR / "calib" / f"{sid}.txt"),
                    image_shape=(h, w),
                ),
                cls_name="Car", score_thr=0.0,
            )
            if (i + 1) % 100 == 0 or i == 0 or i == len(val_ids) - 1:
                elapsed = time.time() - t0
                avg_ms = elapsed * 1000 / (i + 1)
                eta_s = avg_ms * (len(val_ids) - i - 1) / 1000
                LOG.info("  [%d/%d] %s  det=%d  avg=%.0fms  ETA=%.0fs",
                         i + 1, len(val_ids), sid, n_det, avg_ms, eta_s)
        except Exception as e:
            LOG.info(f"[eval] frame {sid} error: {e}")

    elapsed = time.time() - t0
    LOG.info("[eval] 推理完成: %.1fs (%.0fms/帧), 检测 %d 框",
             elapsed, elapsed * 1000 / len(val_ids), n_det)
    return pred_dir


def _run_kitti_eval_flow(val_ids, pred_dir, max_frames):
    """陈旧 pred 防护 + 事后从 pred 文件重建 annos 评估; 返回是否成功"""
    # 陈旧 pred 防护: pred 文件缺失 = 帧级错误被静默掩盖,
    # AP 会读到上一轮运行的旧文件 (曾因此产生假 "无回归" 结论)
    stale = [
        sid
        for sid in val_ids
        if not os.path.exists(os.path.join(pred_dir, "data", f"{sid}.txt"))
    ]
    if stale:
        LOG.error("[eval][FATAL] %d/%d 帧 pred 文件缺失 "
                  "(推理期错误被静默跳过), AP 不可信, 中止。首缺: %s",
                  len(stale), len(val_ids), stale[:5])
        return False
    LOG.info("[eval] 事后从 pred 文件重建 annos (避免推理进程内联构建的未知污染) ...")
    gt_annos, dt_annos = [], []
    for sid in val_ids:
        gt_annos.append(read_kitti_label_as_anno(
            str(DATA_DIR / "label_2" / f"{sid}.txt")))
        dt_path = os.path.join(pred_dir, "data", f"{sid}.txt")
        dt_annos.append(_read_dt_anno_with_scores(dt_path))
    LOG.info("[eval] 运行 kitti_eval ...")

    result_str, _result_dict = run_kitti_eval(gt_annos, dt_annos)
    LOG.info("\n" + "=" * 60)
    LOG.info("[eval] KITTI Car 3D/BEV AP (IoU>=0.7)")
    LOG.info("=" * 60)
    LOG.info("%s", result_str)

    tag = f"_{max_frames}frames" if max_frames > 0 else "_full3769"
    out_txt = RESULTS_DIR / f"logonet_eval_result{tag}.txt"
    with open(out_txt, "w") as f:
        f.write(result_str)
    LOG.info("[eval] 结果已保存: %s", out_txt)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-frames", type=int, default=0, help="0=全量 3769 帧")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sys_obj = build_logonet_npu(args.device)
    LOG.info("[logonet] 系统就绪\n")

    val_ids = [f"{i:06d}" for i in range(3712, 7481)]
    if args.max_frames > 0:
        val_ids = val_ids[:args.max_frames]

    pred_dir = _run_inference(sys_obj, val_ids, args.max_frames)
    if not _run_kitti_eval_flow(val_ids, pred_dir, args.max_frames):
        sys.exit(2)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
