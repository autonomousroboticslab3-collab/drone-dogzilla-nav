"""端到端 demo：合成地圖 -> 偵測目標 -> 點選 -> A* -> 追蹤。

接上真實資料時替換 synthetic_scene() 為：
  points = np.loadtxt('map_points.csv', delimiter=',')   # 需在 ORB-SLAM3 端匯出
  poses  = load_tum_poses('slam_poses.txt')
  dets   = YOLO 對關鍵幀輸出的偵測結果
"""
import numpy as np
from config import MapConfig, CameraIntrinsics, FollowerConfig
from map_builder import fit_ground, build_grid
from target_locator import locate_targets, merge_observations
from planner import plan
from follower import follow
from robot import SimRobot
from ui_select import pick_goal


def synthetic_scene(rng):
    """10x10 單位的地面 + 兩面牆 + 一個箱子；SLAM 座標刻意旋轉並縮放（尺度 0.5）。"""
    ground = np.c_[rng.uniform(0, 10, (4000, 2)), rng.normal(0, 0.003, 4000)]
    wall1 = np.c_[np.full(300, 5.0) + rng.normal(0, .02, 300), rng.uniform(0, 6, 300), rng.uniform(.1, .8, 300)]
    wall2 = np.c_[rng.uniform(2, 8, 300), np.full(300, 7.0) + rng.normal(0, .02, 300), rng.uniform(.1, .8, 300)]
    box = np.c_[rng.uniform(7, 8, 200), rng.uniform(2, 3, 200), rng.uniform(.1, .5, 200)]
    pts = np.vstack([ground, wall1, wall2, box]) * 0.5   # 真實 1 單位 = 0.5 SLAM 單位
    a = np.deg2rad(30)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ Rz.T + [3, -2, 1]


def scene_to_ground(xy, gf):
    """合成場景座標 -> 地面座標（僅 demo 用；實際流程目標座標直接來自 target_locator）。"""
    a = np.deg2rad(30)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    slam = (np.array([xy[0], xy[1], 0.0]) * 0.5) @ Rz.T + [3, -2, 1]
    return gf.to_ground(slam)[0][:2]


def main():
    rng = np.random.default_rng(1)
    mcfg, fcfg = MapConfig(), FollowerConfig()

    # 1) 地圖 -> 地面座標 -> 尺度校正
    pts = synthetic_scene(rng)
    gf = fit_ground(pts)
    # 使用者在地圖上量一段真實 1 m 的距離（此處用相距 0.5 SLAM 單位的兩點代替）
    a, b = gf.from_ground([0, 0, 0])[0], gf.from_ground([0.5, 0, 0])[0]
    gf.set_scale_from_pair(a, b, real_dist_m=1.0)
    print(f'[map] 尺度 = {gf.scale:.3f}（SLAM 單位 -> 公尺）')
    grid = build_grid(gf.to_ground(pts), mcfg)
    print(f'[map] 柵格 {grid.data.shape}, 障礙格 {int(grid.data.sum())}')

    # 2) 目標（此處直接給地面座標；實際用 locate_targets + merge_observations）
    targets = [scene_to_ground([8.0, 8.0], gf)]
    print('[target]', targets)

    # 3) 使用者點選（無顯示環境時用 fallback）
    start = scene_to_ground([1.0, 1.0], gf)
    goal = pick_goal(grid, targets, start, fallback=targets[0])

    # 4) 規劃 + 追蹤
    path = plan(grid, mcfg.robot_radius, start, goal)
    if path is None:
        print('[plan] 無可行路徑')
        return
    print(f'[plan] {len(path)} 個路徑點')
    robot = SimRobot(*start, yaw=0.0)
    ok = follow(robot, path, fcfg)
    print('[follow]', '抵達' if ok else '逾時', '最終位置', np.round(robot.get_pose()[:2], 2))


if __name__ == '__main__':
    main()
