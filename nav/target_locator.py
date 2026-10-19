"""YOLO 偵測框 + SLAM 相機位姿 -> 目標在地面座標的位置。

做法：把偵測框底部中心的像素反投影成射線，與地面平面 (h=0) 求交。
"""
import numpy as np
from config import CameraIntrinsics
from map_builder import GroundFrame


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]])


def load_tum_poses(path):
    """TUM 格式：t tx ty tz qx qy qz qw -> [(t, R_wc, t_wc)]。"""
    out = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        t, x, y, z, qx, qy, qz, qw = map(float, line.split()[:8])
        out.append((t, quat_to_R(qx, qy, qz, qw), np.array([x, y, z])))
    return out


def pixel_to_ground(px, py, R_wc, t_wc, K: CameraIntrinsics, gf: GroundFrame):
    """回傳地面座標 (u, v)（公尺）；射線不與地面相交則回傳 None。"""
    d_cam = np.array([(px - K.cx) / K.fx, (py - K.cy) / K.fy, 1.0])
    d_w = R_wc @ d_cam                                  # SLAM 座標下射線方向
    o_g = gf.to_ground(t_wc)[0]                         # 相機原點（地面座標）
    d_g = (d_w @ gf.R.T) * gf.scale                     # 方向（地面座標）
    if abs(d_g[2]) < 1e-9:
        return None
    s = -o_g[2] / d_g[2]
    if s <= 0:
        return None
    return (o_g + s * d_g)[:2]


def locate_targets(detections, R_wc, t_wc, K, gf, label='target'):
    """detections: [(label, x1, y1, x2, y2), ...]（同一幀）。"""
    res = []
    for lab, x1, y1, x2, y2 in detections:
        if lab != label:
            continue
        p = pixel_to_ground((x1 + x2) / 2, y2, R_wc, t_wc, K, gf)
        if p is not None:
            res.append(p)
    return res


def merge_observations(points, radius=0.3, min_obs=1):
    """多幀觀測合併（簡單分群取平均）。min_obs：至少被觀測幾次才採用，用來濾掉誤偵測。"""
    clusters = []
    for p in points:
        for c in clusters:
            if np.linalg.norm(np.mean(c, 0) - p) < radius:
                c.append(p)
                break
        else:
            clusters.append([p])
    return [np.mean(c, 0) for c in clusters if len(c) >= min_obs]
