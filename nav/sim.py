"""完整模擬：原廠 Gazebo 場景 + 飛行的空拍機 + 帶雜訊的 SLAM 輸出 -> 導航流程 -> 與真值比較。

  .venv/bin/python sim.py --scene corridor              # 目標 = .world 裡的綠色標記
  .venv/bin/python sim.py --scene corridor --all-goals  # 依序跑原廠 benchmark 的全部目標點
  .venv/bin/python sim.py --scene clutter --click 3.4,-1.5   # 模擬使用者點選此世界座標
  .venv/bin/python sim.py --interactive                 # 真的用滑鼠點

模擬的「假設」（真實系統要靠前半段達成）：
  * SLAM 世界座標與真實世界差一個未知的旋轉、平移、尺度（單目）
  * 地圖點、相機位姿、偵測框都有雜訊；偵測會漏偵、會誤偵
  * 使用者量一段已知距離來校正尺度；機器人起點與朝向由使用者在地圖上標（以真值 + 雜訊代替）
  * 機器人速度特性取自原廠實測：指令 0.15 m/s 實際約 0.084 m/s，轉彎時自動降速
"""
import argparse
import math
import numpy as np
import matplotlib
from config import MapConfig, CameraIntrinsics, FollowerConfig
from map_builder import fit_ground, orient_ground, build_grid
from target_locator import locate_targets, merge_observations
from planner import plan, reachable_area
from follower import follow
from robot import SimRobot
from ui_select import pick_goal
from worlds import load_scene, DEFAULT_SIM_ENV

IMG_W, IMG_H = 1280, 720
K = CameraIntrinsics(fx=900.0, fy=900.0, cx=IMG_W / 2, cy=IMG_H / 2)
TARGET_SIZE = 0.3                 # 平貼地面的目標標記邊長（原廠的 goal_marker 是平的綠色圓墊）
R_NADIR = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])   # 下視：影像右 = +x，下 = -y，光軸 = -z


# ---------------- 場景表面取樣 ----------------
def sample_prim(p, rng, density):
    """物體表面點 (N,6)：x y z 與法向量。在物體自己的座標（原點在中心）取樣，再旋轉（傾斜 + 偏航）並平移。"""
    co, si, cp, sp = math.cos(p.yaw), math.sin(p.yaw), math.cos(p.pitch), math.sin(p.pitch)
    R = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    c3 = np.array([p.c[0], p.c[1], p.z0 + p.h / 2])
    loc, nrm = [], []

    def add(pts, n):
        loc.append(pts); nrm.append(np.tile(n, (len(pts), 1)))

    hz = p.h / 2
    if p.kind == 'box':
        sx, sy = np.array(p.size) / 2
        n = max(int(4 * sx * sy * density), 1)
        add(np.c_[rng.uniform(-sx, sx, n), rng.uniform(-sy, sy, n), np.full(n, hz)], [0, 0, 1.0])
        for sgn in (1, -1):
            n = max(int(4 * sy * hz * density), 1)
            add(np.c_[np.full(n, sgn * sx), rng.uniform(-sy, sy, n), rng.uniform(-hz, hz, n)], [sgn, 0, 0.0])
            n = max(int(4 * sx * hz * density), 1)
            add(np.c_[rng.uniform(-sx, sx, n), np.full(n, sgn * sy), rng.uniform(-hz, hz, n)], [0, sgn, 0.0])
    else:
        r = p.size[0]
        n = max(int(math.pi * r * r * density), 1)
        rr, tt = r * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
        add(np.c_[rr * np.cos(tt), rr * np.sin(tt), np.full(n, hz)], [0, 0, 1.0])
        n = max(int(2 * math.pi * r * p.h * density), 1)
        tt = rng.uniform(0, 2 * np.pi, n)
        loc.append(np.c_[r * np.cos(tt), r * np.sin(tt), rng.uniform(-hz, hz, n)])
        nrm.append(np.c_[np.cos(tt), np.sin(tt), np.zeros(n)])
    loc, nrm = np.vstack(loc), np.vstack(nrm)
    return np.c_[loc @ R.T + c3, nrm @ R.T]


def sample_world(scene, rng, g_density=60, o_density=400, margin=1.0):
    lo, hi = scene.bounds()
    lo, hi = lo - margin, hi + margin
    n = int(np.prod(hi - lo) * g_density)
    g = rng.uniform(lo, hi, (n, 2))
    g = g[scene.clearance(g, physical=False) > 0]      # 物體（含懸空的板）底下，空拍機看不到地面
    out = [np.c_[g, np.zeros(len(g)), np.tile([0, 0, 1.0], (len(g), 1))]]
    out += [sample_prim(p, rng, o_density) for p in scene.prims]
    return np.vstack(out)


def centers(lo, hi, half, max_step):
    if hi - lo <= 2 * half:
        return [(lo + hi) / 2]
    n = int(math.ceil((hi - lo - 2 * half) / max_step)) + 1
    return list(np.linspace(lo + half, hi - half, n))


def drone_keyframes(scene, alt):
    hw, hh = alt * (IMG_W / 2) / K.fx, alt * (IMG_H / 2) / K.fy
    lo, hi = scene.bounds()
    # 飛出邊界半個畫面：否則場地邊緣只被最外圈的 1 個關鍵幀看到，邊緣的目標偵測不到
    lo, hi = lo - [0.5 * hw, 0.5 * hh], hi + [0.5 * hw, 0.5 * hh]
    xs, ys = centers(lo[0], hi[0], hw, 0.6 * hw), centers(lo[1], hi[1], hh, 0.8 * hh)   # 重疊多，同一目標才會被看到 3 次以上
    kfs = []
    for i, y in enumerate(ys):
        for x in (xs if i % 2 == 0 else xs[::-1]):
            kfs.append(np.array([x, y, alt]))
    return kfs


def project(pts, R_wc, t_wc):
    pc = (pts - t_wc) @ R_wc
    z = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return K.fx * pc[:, 0] / z + K.cx, K.fy * pc[:, 1] / z + K.cy, z


def small_rot(rng, deg):
    w = np.deg2rad(rng.normal(0, deg, 3))
    return np.eye(3) + np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


class DogzillaLikeRobot(SimRobot):
    """模擬原廠量到的速度特性：0.15 -> 約 0.084 m/s；轉彎降速 v*max(0.15, 1-|w|)；加雜訊。"""

    def __init__(self, rng, nz, *a, **k):
        super().__init__(*a, **k)
        self.rng, self.nz, self.steps = rng, nz, 0

    def send_cmd(self, v, w):
        self.steps += 1
        v = v * max(0.15, 1 - abs(w)) * 0.56
        super().send_cmd(max(0.0, v + self.rng.normal(0, 0.01 * self.nz)), w + self.rng.normal(0, 0.05 * self.nz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='corridor', help='project3 / corridor / paper_fig4 / clutter / quake[:seed]（震後）')
    ap.add_argument('--sim-env', default=str(DEFAULT_SIM_ENV), help='原廠 DOGZILLA/sim_env 路徑')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=1.0, help='雜訊倍率')
    ap.add_argument('--alt', type=float, default=5.0, help='空拍機高度 (m)')
    ap.add_argument('--click', help='模擬使用者點選的世界座標 x,y')
    ap.add_argument('--all-goals', action='store_true', help='依序跑原廠 benchmark 的全部目標點')
    ap.add_argument('--interactive', action='store_true', help='用滑鼠點選目的地')
    ap.add_argument('--no-dust', action='store_true', help='關掉煙塵（用來分離煙塵的影響）')
    ap.add_argument('--world-out', help='把場景存成 Gazebo .world（震後場景可直接放進原廠模擬）')
    ap.add_argument('--start', help='機器人當下的世界位姿 x,y,yaw（使用者在空拍機地圖上標的起點；整合 Gazebo 時用）')
    ap.add_argument('--block', help='重規劃用：地面座標的封鎖點 "x,y;x,y"（機器人在那裡走不過去），以 --block=... 傳入')
    ap.add_argument('--block-radius', type=float, default=0.6, help='封鎖點半徑 (m)')
    ap.add_argument('--export', help='只規劃不模擬機器人，把目標（機器人座標系偏移量）輸出成 JSON 給 Gazebo 橋接')
    ap.add_argument('--overlay', help='讀 Gazebo 回傳的軌跡 JSON，疊到地圖上並用真值評分')
    ap.add_argument('--out', default='sim_result.png')
    a = ap.parse_args()
    if not a.interactive:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    rng = np.random.default_rng(a.seed)
    nz = a.noise
    scene = load_scene(a.scene, a.sim_env)
    if a.no_dust:
        scene.dust = []
    if a.start:
        sx_, sy_, syaw_ = [float(v) for v in a.start.split(',')]
        scene.start, scene.start_yaw = np.array([sx_, sy_]), syaw_
    mcfg = MapConfig(resolution=0.05, robot_radius=0.15)
    fcfg = FollowerConfig(v_max=0.15, goal_tol=0.15)
    if a.world_out:
        from worlds import write_world
        write_world(scene, a.world_out)
        print(f'[scene] 已輸出 {a.world_out}')
    lo, hi = scene.bounds()
    print(f'[scene] {scene.name}：{len(scene.prims)} 個障礙物，範圍 {lo.round(2)} ~ {hi.round(2)}')

    # ---- 1) 未知的 SLAM 座標系 ----
    yaw = rng.uniform(0, 2 * np.pi)
    Rs = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    s_true, t0 = 0.37, np.array([5.0, -3.0, 2.0])
    w2s = lambda p: s_true * (np.atleast_2d(p) @ Rs.T) + t0
    s2w = lambda p: ((np.atleast_2d(p) - t0) / s_true) @ Rs

    # ---- 2) 空拍機飛行 ----
    world = sample_world(scene, rng)
    marker = scene.goal_marker
    h = TARGET_SIZE / 2
    cube = np.array([[marker[0] + sx * h, marker[1] + sy * h, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
    seen = np.zeros(len(world), bool)
    poses, dets_per_frame, kfs = [], [], drone_keyframes(scene, a.alt)
    for c in kfs:
        R_wc = R_NADIR @ small_rot(rng, 1.0 * nz)
        u, v, z = project(world[:, :3], R_wc, c)
        facing = np.einsum('ij,ij->i', world[:, 3:], c - world[:, :3]) > 0
        seen |= (z > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H) & facing
        poses.append((Rs @ R_wc @ small_rot(rng, 0.3 * nz), w2s(c + rng.normal(0, 0.05 * nz, 3))[0]))
        dets = []
        cu, cv, cz = project(cube, R_wc, c)
        if np.all(cz > 0) and cu.min() > 0 and cu.max() < IMG_W and cv.min() > 0 and cv.max() < IMG_H \
                and rng.random() > (0.85 if scene.in_dust(marker)[0] else 0.10):
            dets.append(('target', *(np.array([cu.min(), cv.min(), cu.max(), cv.max()]) + rng.normal(0, 3 * nz, 4))))
        if rng.random() < 0.05:
            x, y = rng.uniform(100, IMG_W - 100), rng.uniform(100, IMG_H - 100)
            dets.append(('target', x - 20, y - 20, x + 20, y + 20))
        dets_per_frame.append(dets)
    if scene.dust:                                                   # 煙塵區：80% 的特徵點追蹤不到
        dusty = scene.in_dust(world[:, :2])
        seen &= ~(dusty & (rng.random(len(world)) < 0.8))
    slam_pts = w2s(world[seen, :3] + rng.normal(0, 0.03 * nz, (seen.sum(), 3)))
    print(f'[sim] 關鍵幀 {len(kfs)}（高度 {a.alt} m），地圖點 {len(slam_pts)}，真實尺度 {s_true}')

    # ---- 3) 地面擬合 + 尺度校正 + 柵格 ----
    # 下視空拍機：地面法向量 ≈ 相機光軸的反方向（光軸 = 旋轉矩陣第 3 欄）
    up_hint = -np.mean([p[0][:, 2] for p in poses], axis=0)
    gf = fit_ground(slam_pts, up_hint=up_hint)
    orient_ground(gf, np.array([p[1] for p in poses]))
    ctr = (lo + hi) / 2
    pA, pB = np.r_[ctr - [1.5, 0], 0.0], np.r_[ctr + [1.5, 0], 0.0]     # 使用者量地面上一段 3 m
    gf.set_scale_from_pair(w2s(pA)[0] + rng.normal(0, 0.01 * nz, 3) * s_true,
                           w2s(pB)[0] + rng.normal(0, 0.01 * nz, 3) * s_true, real_dist_m=3.0)
    print(f'[map] 估計尺度 {gf.scale:.3f}（真值 {1 / s_true:.3f}）')
    gp = gf.to_ground(slam_pts)
    hq = np.percentile(gp[:, 2], [5, 50, 90, 99])
    print(f'[map] 高度分位數 5/50/90/99% = {hq.round(2)} m（5% 分位數應接近 0 = 地面；中位數會被牆面的點拉高，不用看）')
    grid = build_grid(gp, mcfg)
    print(f'[map] 柵格 {grid.data.shape}，障礙格 {int(grid.data.sum())}')
    blocks = [np.array([float(v) for v in b.split(',')]) for b in a.block.split(';')] if a.block else []
    for b_ in blocks:                                   # 機器人在這裡走不過去：地圖上標成障礙，重規劃會繞開
        rr, cc = grid.world_to_cell(b_)
        k = int(np.ceil(a.block_radius / grid.res))
        R_, C_ = np.ogrid[:grid.data.shape[0], :grid.data.shape[1]]
        grid.data[(R_ - rr) ** 2 + (C_ - cc) ** 2 <= k * k] = 1
    if blocks:
        print(f'[map] 已封鎖 {len(blocks)} 處（機器人實際走不過去的地方）')
    # 地圖品質檢查：每個真實物體的俯視範圍中，有多少比例在地圖上被標成障礙
    w2g_ = lambda xy: gf.to_ground(w2s(np.r_[xy, 0.0]))[0][:2]
    missed = []
    for p in [q for q in scene.prims if q.blocks]:
        g = np.stack(np.meshgrid(*[np.linspace(-1, 1, 7)] * 2), -1).reshape(-1, 2)
        loc = g * (p.half() if p.kind == 'box' else p.size[0] * 0.7)
        co, si = math.cos(p.yaw), math.sin(p.yaw)
        pts = loc @ np.array([[co, si], [-si, co]]) + p.c
        hit = np.mean([grid.data[grid.world_to_cell(w2g_(q))] if
                       0 <= grid.world_to_cell(w2g_(q))[0] < grid.data.shape[0] and
                       0 <= grid.world_to_cell(w2g_(q))[1] < grid.data.shape[1] else 0 for q in pts])
        if hit < 0.5:
            missed.append((p.name, p.h, hit))
    print(f'[map] 漏檢物體 {len(missed)}/{sum(q.blocks for q in scene.prims)}：'
          + ', '.join(f'{n}(高{h:.2f}m,{r:.0%})' for n, h, r in missed))
    w2g = lambda xy: gf.to_ground(w2s(np.r_[xy, 0.0]))[0][:2]
    g2w = lambda g: s2w(gf.from_ground(np.c_[np.atleast_2d(g), np.zeros(len(np.atleast_2d(g)))]))[:, :2]

    # ---- 4) 目標偵測 ----
    obs = []
    for (R_wc, t_wc), dets in zip(poses, dets_per_frame):
        obs += locate_targets(dets, R_wc, t_wc, K, gf)
    found = merge_observations(obs, radius=0.5, min_obs=3)
    print(f'[target] 原始觀測 {len(obs)} 筆 -> 有效目標 {len(found)} 個'
          + (f'，定位誤差 {min(np.linalg.norm(g2w(f)[0] - marker) for f in found):.2f} m' if found else ''))

    # ---- 5) 起點（使用者在地圖上標位置與朝向）----
    start = w2g(scene.start) + rng.normal(0, 0.05 * nz, 2)
    d = w2g(scene.start + [math.cos(scene.start_yaw), math.sin(scene.start_yaw)]) - w2g(scene.start)
    start_yaw = math.atan2(d[1], d[0]) + math.radians(rng.normal(0, 3 * nz))

    # ---- 6) 決定要跑的目標清單（世界座標）----
    if a.interactive:
        g = pick_goal(grid, found, start, fallback=found[0] if found else None)
        goals_w = [g2w(g)[0]]
    elif a.click:
        goals_w = [np.array([float(v) for v in a.click.split(',')])]
    elif a.all_goals:
        goals_w = [np.array(g) for g in scene.goals]
    elif found:
        goals_w = [g2w(found[0])[0]]            # 使用者點選自動偵測到的目標
    else:
        print('[target] 沒偵測到目標，且未指定 --click'); return
    if a.interactive:
        goals_g = [g]
    else:
        goals_g = [w2g(g) if (a.click or a.all_goals) else found[0] for g in goals_w]

    # ---- 7) 逐目標：規劃 -> 追蹤 -> 評估 ----
    results = []
    for k, (gw, gg) in enumerate(zip(goals_w, goals_g)):
        path = plan(grid, mcfg.robot_radius, start, gg, clear_radius=0.1, start_clear=0.4)   # 機器人實際就站在起點，周圍一定走得出去
        if path is None:
            ra, ta = reachable_area(grid, mcfg.robot_radius, start, 0.4)
            if ra < 0.2 * ta:
                # 脫困模式：起點被困在小空間。機器人既然在裡面，就代表過得了這些縫，逐步縮小膨脹半徑再試
                for rad in (0.10, 0.05, 0.0):
                    path = plan(grid, rad, start, gg, clear_radius=0.1, start_clear=0.4)
                    if path is not None:
                        print(f'   [脫困模式] 起點被困，膨脹半徑縮小到 {rad:.2f} m 後找到路徑（會擠過窄縫，Nav2 不一定肯走）')
                        break
        if path is None:
            rc = grid.world_to_cell(gg)
            in_map = 0 <= rc[0] < grid.data.shape[0] and 0 <= rc[1] < grid.data.shape[1]
            why = '目標落在空拍機沒看清楚的區域（煙塵／點太少），建議降低高度重拍' if (not in_map or not grid.known[rc]) \
                else '地圖上目標被障礙包圍或沒有通路'
            print(f'[goal {k}] ({gw[0]:.2f},{gw[1]:.2f}) 規劃失敗：{why}')
            ra, ta = reachable_area(grid, mcfg.robot_radius, start, 0.4)
            print(f'   [診斷] 從起點可達 {ra:.1f} m²，整張地圖可通行 {ta:.1f} m²'
                  + ('（起點被困在小區域裡）' if ra < 0.2 * ta else ''))
            results.append(dict(gw=gw, path=None, traj=None, ok=False))
            continue
        plen = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        if a.export:
            # 只規劃：把目標與途經點轉成「機器人座標系」（x 朝前、y 朝左）的偏移量。
            # 這是真實系統唯一需要的對齊資訊：使用者標的機器人位置與朝向（含標註誤差）。
            c_, s_ = math.cos(-start_yaw), math.sin(-start_yaw)
            to_body = lambda p: [float(c_ * (p[0] - start[0]) - s_ * (p[1] - start[1])),
                                 float(s_ * (p[0] - start[0]) + c_ * (p[1] - start[1]))]
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            idx = sorted({int(np.searchsorted(cum, t)) for t in np.arange(1.0, cum[-1], 1.0)} | {len(path) - 1})
            results.append(dict(gw=gw, path=path, traj=None, ok=False, clear=1.0, plen=plen,
                                export=dict(goal_body=to_body(gg), waypoints_body=[to_body(path[i]) for i in idx], waypoints_ground=[[float(path[i][0]), float(path[i][1])] for i in idx],
                                            goal_world_true=[float(gw[0]), float(gw[1])], path_len=plen,
                                            start_marked=[float(start[0]), float(start[1]), float(start_yaw)])))
            print(f'[goal {k}] 規劃完成：路徑 {plen:.1f} m，目標在機器人座標系 (前 {to_body(gg)[0]:.2f}, 左 {to_body(gg)[1]:.2f}) m')
            continue
        if a.overlay:
            import json
            ov = json.load(open(a.overlay))
            tw = np.array(ov['trace'])
            traj = np.array([w2g(p) for p in tw])
            ok = ov['status'] in ('SUCCEEDED', 'REACHED') and not ov.get('flipped')
            steps_dt = float(ov.get('duration', 0))
        else:
            robot = DogzillaLikeRobot(rng, nz, *start, yaw=start_yaw, dt=0.1)
            traj = []
            orig = robot.send_cmd
            robot.send_cmd = lambda v, w, r=robot, o=orig, t=traj: (t.append(r.get_pose()[:2]), o(v, w))[1]
            ok = follow(robot, path, fcfg, max_steps=8000)
            traj = np.array(traj)
            tw = g2w(traj)
            steps_dt = robot.steps * robot.dt
        clear = float(np.min(scene.clearance(tw)))
        hit = ''
        if clear <= 0:
            phys = [p for p in scene.prims if p.blocks]
            cl = np.array([p.clearance(tw) for p in phys])      # (物體數, 軌跡點數)
            i, j = np.unravel_index(np.argmin(cl), cl.shape)
            p = phys[i]
            hit = f' 撞到 {p.name}（高 {p.h:.2f} m，{"在地圖上" if grid.data[grid.world_to_cell(w2g(p.c))] else "不在地圖上"}）'
        dist = float(np.linalg.norm(tw[-1] - gw))
        results.append(dict(gw=gw, path=path, traj=traj, tw=tw, ok=ok, dist=dist, clear=clear, plen=plen,
                            t=steps_dt))
        print(f'[goal {k}] ({gw[0]:.2f},{gw[1]:.2f}) 路徑 {plen:.1f} m，'
              f'{"抵達" if ok else "逾時"} 用時 {steps_dt:.0f}s，離真實目標 {dist:.2f} m，'
              f'離障礙最近 {clear:.2f} m{"（碰撞！）" + hit if clear <= 0 else ""}')
    if a.export:
        import json
        ex = [r['export'] for r in results if 'export' in r]
        if not ex:
            print('[export] 沒有可匯出的目標（規劃失敗）'); return
        json.dump(ex[0], open(a.export, 'w'), indent=1)
        print(f'[export] 已輸出 {a.export}')
    n_ok = sum(1 for r in results if r['ok'] and r['clear'] > 0)
    print(f'[summary] {scene.name}：抵達且無碰撞 {n_ok}/{len(results)}')

    # ---- 8) 繪圖 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for p in scene.prims:
        if not p.blocks:                                            # 懸空樓板：淡色斜線
            ax1.add_patch(Polygon(p.corners(), fc='none', ec='tab:blue', hatch='//', lw=0.8))
        else:
            ax1.add_patch(Polygon(p.corners(), color='sienna' if abs(p.pitch) > 0 else 'gray'))
    for x, y, r in scene.dust:
        ax1.add_patch(plt.Circle((x, y), r, color='orange', alpha=0.18, lw=0))
    ax1.plot(*zip(*[k[:2] for k in kfs]), 'c.-', alpha=.4, label='drone path')
    ax1.plot(*marker, 'g*', ms=14, label='goal marker')
    ax1.plot(*scene.start, 'bo', label='robot start')
    cols = plt.cm.tab10(np.arange(10))
    for i, r in enumerate(results):
        ax1.plot(*r['gw'], 'x', color=cols[i % 10], ms=10)
        if r['traj'] is not None:
            ax1.plot(r['tw'][:, 0], r['tw'][:, 1], '-', color=cols[i % 10], lw=1.5)
    ax1.set_xlim(lo[0] - .3, hi[0] + .3); ax1.set_ylim(lo[1] - .3, hi[1] + .3); ax1.set_aspect('equal')
    ax1.set_title(f'Ground truth: {scene.name}'); ax1.legend(loc='upper right', fontsize=7)
    ext = [grid.origin[0], grid.origin[0] + grid.data.shape[1] * grid.res,
           grid.origin[1], grid.origin[1] + grid.data.shape[0] * grid.res]
    ax2.imshow(1 - grid.blocked(mcfg.robot_radius) * 0.4 - grid.data * 0.6, cmap='gray', origin='lower',
               extent=ext, vmin=0, vmax=1)
    for i, r in enumerate(results):
        if r['path'] is not None:
            ax2.plot(r['path'][:, 0], r['path'][:, 1], '--', color=cols[i % 10], lw=1)
            if r['traj'] is not None:
                ax2.plot(r['traj'][:, 0], r['traj'][:, 1], '-', color=cols[i % 10], lw=1.5)
        ax2.plot(*goals_g[i], 'x', color=cols[i % 10], ms=10)
    for f in found:
        ax2.plot(*f, 'g*', ms=14)
    ax2.plot(*start, 'bo')
    ax2.set_title(f'Estimated map + paths: reached {n_ok}/{len(results)}')
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print(f'[plot] 已存 {a.out}')
    if a.interactive:
        plt.show()


if __name__ == '__main__':
    main()
