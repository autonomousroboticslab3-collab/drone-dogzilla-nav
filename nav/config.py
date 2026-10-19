"""集中管理參數。路徑一律用相對路徑或參數傳入，不要寫死。"""
from dataclasses import dataclass


@dataclass
class MapConfig:
    resolution: float = 0.05       # 每格幾公尺（尺度校正後）
    obstacle_min_h: float = 0.08   # 高於地面多少公尺視為障礙
    obstacle_max_h: float = 3.0    # 高於此視為空中物（忽略）。空拍機只看得到物體頂部，不能設太低
    min_points_per_cell: int = 2   # 鄰域內至少幾個點才算障礙（濾雜訊）
    count_window: int = 1          # 鄰域半徑（格）；1 = 3x3。點雲越稀疏要越大，但障礙會被略為加粗
    robot_radius: float = 0.20     # 機器人半徑，用於障礙膨脹
    known_fill: float = 0.30       # 稀疏點雲補洞半徑 (m)，用來判定哪些區域有被觀測
    min_density_frac: float = 0.35 # 局部點密度低於地圖典型密度的這個比例 -> 視為未知（0 = 不檢查）
    density_window: float = 0.30   # 計算局部密度的視窗半徑 (m)


@dataclass
class CameraIntrinsics:
    # 請換成 dji.yaml 內的值
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 640.0
    cy: float = 360.0


@dataclass
class FollowerConfig:
    lookahead: float = 0.30        # pure pursuit 前視距離 (m)
    v_max: float = 0.20            # 線速度上限 (m/s)
    w_max: float = 1.0             # 角速度上限 (rad/s)
    goal_tol: float = 0.15         # 抵達容許誤差 (m)
    rate_hz: float = 10.0
