# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pcdet.utils.calibration_kitti import Calibration
from pcdet.utils.box_utils import (
    boxes3d_lidar_to_kitti_camera,
    boxes3d_kitti_camera_to_imageboxes
)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_box_to_image(box_cam, calib):
    x, y, z, box_len, box_wid, box_hgt, ry = box_cam

    corners = np.array([
        [box_len / 2, box_wid / 2, 0],
        [box_len / 2, -box_wid / 2, 0],
        [-box_len / 2, -box_wid / 2, 0],
        [-box_len / 2, box_wid / 2, 0],
        [box_len / 2, box_wid / 2, -box_hgt],
        [box_len / 2, -box_wid / 2, -box_hgt],
        [-box_len / 2, -box_wid / 2, -box_hgt],
        [-box_len / 2, box_wid / 2, -box_hgt],
    ])

    rot_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    corners = corners @ rot_mat.T
    corners += np.array([x, y, z])

    pts_img = calib.project_rect_to_image(corners)

    x1, y1 = pts_img.min(axis=0)
    x2, y2 = pts_img.max(axis=0)

    return [x1, y1, x2, y2]


class CalibAdapter:
    """
    适配 OpenPCDet box_utils 所需的 calibration 接口
    输入 calib 为 dict:
        {
            "P2": (3,4),
            "R0_rect": (3,3),
            "Tr_velo_to_cam": (3,4)
        }
    """

    def __init__(self, calib: dict):
        self.p2_mat = _to_numpy(calib["P2"]).astype(np.float32)
        self.r0_mat = _to_numpy(calib["R0_rect"]).astype(np.float32)
        self.tr_mat = _to_numpy(calib["Tr_velo_to_cam"]).astype(np.float32)

        self.r0_ext_mat = np.eye(4, dtype=np.float32)
        self.r0_ext_mat[:3, :3] = self.r0_mat

        self.tr_ext_mat = np.eye(4, dtype=np.float32)
        self.tr_ext_mat[:3, :4] = self.tr_mat

    def lidar_to_rect(self, pts_lidar: np.ndarray) -> np.ndarray:
        pts_lidar = np.asarray(pts_lidar, dtype=np.float32)
        pts_hom = np.concatenate(
            [pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)],
            axis=1
        )
        pts_rect = (self.r0_ext_mat @ (self.tr_ext_mat @ pts_hom.T)).T
        return pts_rect[:, :3]

    def rect_to_img(self, pts_rect: np.ndarray):
        pts_rect = np.asarray(pts_rect, dtype=np.float32)
        pts_hom = np.concatenate(
            [pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)],
            axis=1
        )
        pts_2d_hom = (self.p2_mat @ pts_hom.T).T
        pts_img = pts_2d_hom[:, 0:2] / np.clip(pts_2d_hom[:, 2:3], a_min=1e-6, a_max=None)
        pts_depth = pts_2d_hom[:, 2]
        return pts_img, pts_depth

    def corners3d_to_img_boxes(self, corners3d: np.ndarray):
        """
        OpenPCDet 的 boxes3d_kitti_camera_to_imageboxes 会调用这个接口
        corners3d: (N,8,3)
        """
        sample_num = corners3d.shape[0]
        corners_hom = np.concatenate(
            [corners3d, np.ones((sample_num, 8, 1), dtype=np.float32)],
            axis=2
        )  # (N,8,4)

        img_pts = corners_hom @ self.p2_mat.T  # (N,8,3)
        x = img_pts[:, :, 0] / np.clip(img_pts[:, :, 2], a_min=1e-6, a_max=None)
        y = img_pts[:, :, 1] / np.clip(img_pts[:, :, 2], a_min=1e-6, a_max=None)

        x1 = np.min(x, axis=1)
        y1 = np.min(y, axis=1)
        x2 = np.max(x, axis=1)
        y2 = np.max(y, axis=1)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        boxes_corner = np.stack([x, y], axis=2).astype(np.float32)
        return boxes, boxes_corner


@dataclass
class KittiPredFrame:
    """单帧预测数据: frame_id/boxes_lidar/scores/calib_path/image_shape"""

    frame_id: str
    boxes_lidar: Any
    scores: Any
    calib_path: str
    image_shape: Any


def _kitti_box_line(boxes_camera, boxes_img, scores, i, line_ctx):
    """单框 → KITTI label_2 行 (无效框返回 None)

    line_ctx: (cls_name, img_h, img_w)
    """
    cls_name, img_h, img_w = line_ctx
    x1, y1, x2, y2 = boxes_img[i]
    x1 = np.clip(x1, 0, img_w - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    x, y, z = boxes_camera[i, 0:3]
    box_len, box_hgt, box_wid = boxes_camera[i, 3:6]
    ry = boxes_camera[i, 6]
    score = scores[i]
    # 跳过无效框
    if z <= 0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    # 关键修正：OpenPCDet 这里的 y 是 box center，
    # KITTI label_2 需要的是 bottom center

    return (
        f"{cls_name} 0 0 {boxes_camera[i, 6] - np.arctan2(x, z):.6f} "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{box_hgt:.6f} {box_wid:.6f} {box_len:.6f} "
        f"{x:.6f} {y:.6f} {z:.6f} "
        f"{ry:.6f} {score:.6f}"
    )


def write_kitti_pred_txt_pcdet(
    pred_dir: str,
    frame: KittiPredFrame,
    cls_name: str = "Car",
    score_thr: float = 0.0
):
    """
    直接使用 OpenPCDet 的 KITTI 转换逻辑
    约定输入 boxes_lidar 当前来自你的工程:
        [x, y, z, w, l, h, yaw]
    OpenPCDet 期望:
        [x, y, z, l, w, h, yaw]
    所以这里会自动交换第4/5维
    """
    frame_id = frame.frame_id
    boxes_lidar = frame.boxes_lidar
    scores = frame.scores
    calib_path = frame.calib_path
    image_shape = frame.image_shape

    os.makedirs(pred_dir, exist_ok=True)
    data_dir = os.path.join(pred_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{frame_id}.txt")

    boxes_lidar = _to_numpy(boxes_lidar).astype(np.float32)
    scores = _to_numpy(scores).astype(np.float32).reshape(-1)
    if boxes_lidar.shape[0] > 0:
        keep = scores >= float(score_thr)
        boxes_lidar = boxes_lidar[keep]
        scores = scores[keep]

    if boxes_lidar.shape[0] == 0:
        open(out_path, "w").close()
        return out_path

    boxes_pcdet = boxes_lidar.copy()
    boxes_pcdet[:, 2] = boxes_pcdet[:, 2] + boxes_pcdet[:, 5] / 2.0
    calib_obj = Calibration(calib_path)

    # OpenPCDet 逻辑：LiDAR -> KITTI camera
    boxes_camera = boxes3d_lidar_to_kitti_camera(boxes_pcdet, calib_obj)

    # OpenPCDet 逻辑：camera box -> image box
    boxes_img = boxes3d_kitti_camera_to_imageboxes(
        boxes_camera,
        calib_obj,
        image_shape=image_shape
    )
    img_h, img_w = int(image_shape[0]), int(image_shape[1])

    line_ctx = (cls_name, img_h, img_w)
    lines = []
    for i in range(boxes_camera.shape[0]):
        line = _kitti_box_line(boxes_camera, boxes_img, scores, i, line_ctx)
        if line is not None:
            lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path