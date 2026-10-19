"""SLAM 地圖點 -> 地面座標系 + 2D 佔據柵格。

輸入：N x 3 的 SLAM 地圖點（SLAM 世界座標，單位任意）。
流程：RANSAC 擬合地面 -> 建立地面座標系 (u,v,h) -> 尺度校正 -> 柵格化。
"""
import numpy as np
from config import MapConfig


class GroundFrame:
    """SLAM 世界座標 <-> 地面座標 (u, v, h)，單位公尺（需先設 scale）。"""

    def __init__(self, origin, R, scale=1.0):
        self.origin = origin   # 地面上一點（SLAM 座標）
        self.R = R             # 3x3，列為 u,v,n 軸（SLAM 座標表示）
        self.scale = scale

    def to_ground(self, pts):
        return (np.atleast_2d(pts) - self.origin) @ self.R.T * self.scale

    def from_ground(self, g):
        return (np.atleast_2d(g) / self.scale) @ self.R + self.origin

    def set_scale_from_pair(self, p1, p2, real_dist_m):
        """用地面上已知實際距離的兩點（SLAM 座標）校正尺度。"""
        self.scale = real_dist_m / np.linalg.norm(np.asarray(p1) - np.asarray(p2))


def orient_ground(gf, cam_positions_slam):
    """法向量正負號由 RANSAC 決定，不固定。相機（空拍機）必須在地面上方，
    否則高度為負、障礙會全部被忽略。"""
    if gf.to_ground(cam_positions_slam)[:, 2].mean() < 0:
        gf.R[1] *= -1   # v 與 n 同時反向，維持右手系
        gf.R[2] *= -1
    return gf


def fit_ground(points, iters=500, thresh=None, seed=0, thresh_ratio=0.006, up_hint=None, max_angle_deg=15.0):
    """RANSAC 擬合地面。

    up_hint：地面法向量大致朝哪（SLAM 座標，通常取「下視相機光軸的反方向」）。
    只有加這個限制才可靠：災區裡牆面的取樣點遠多於地面，無限制的 RANSAC 會選到牆。
    """
    rng = np.random.default_rng(seed)
    if thresh is None:
        thresh = thresh_ratio * np.ptp(points, axis=0).max()   # 尺度未知，只能用點雲範圍的相對值
    cos_min = np.cos(np.deg2rad(max_angle_deg))
    hint = None if up_hint is None else np.asarray(up_hint, float) / np.linalg.norm(up_hint)
    cands = []                                     # (內點數, 沿 hint 的高度, 內點遮罩)
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if hint is not None and abs(n @ hint) < cos_min:
            continue
        inl = np.abs((points - s[0]) @ n) < thresh
        # 用內點群的平均高度比較，不能用取樣點的高度（傾斜的假平面會穿過不同高度）
        cands.append((int(inl.sum()), float((points[inl] @ hint).mean()) if hint is not None else 0.0, inl))
    if not cands:
        raise RuntimeError('找不到符合方向限制的地面平面')
    top = max(c[0] for c in cands)
    if hint is not None:
        # 點數最多的水平面不一定是地面：高障礙的頂面取樣很密，合起來常是「點最多」的面。
        # 地面一定在所有障礙頂面下方，所以在點數夠多的候選裡選最低的。
        good = [c for c in cands if c[0] >= 0.25 * top]
        best_inl = min(good, key=lambda c: c[1])[2]
    else:
        best_inl = max(cands, key=lambda c: c[0])[2]
    g = points[best_inl]
    origin = g.mean(0)
    n = np.linalg.eigh(np.cov((g - origin).T))[1][:, 0]   # 共變異數最小特徵值的特徵向量 = 法向量（不會建 N x N 矩陣）
    if hint is not None and n @ hint < 0:
        n = -n
    for _ in range(2):                              # 用擬合出的平面重新找內點、再擬合，去掉取樣三點帶來的偏差
        inl = np.abs((points - origin) @ n) < thresh
        g = points[inl]
        origin = g.mean(0)
        n = np.linalg.eigh(np.cov((g - origin).T))[1][:, 0]
        if hint is not None and n @ hint < 0:
            n = -n
    u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return GroundFrame(origin, np.stack([u, v, n]))
def _dilate(a, k):
    out = a.copy()
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr * dr + dc * dc <= k * k:
                out |= np.roll(np.roll(a, dr, 0), dc, 1)
    return out


class OccupancyGrid:
    def __init__(self, shape, resolution, origin_xy):
        self.data = np.zeros(shape, dtype=np.uint8)   # 0 free, 1 obstacle
        self.known = np.ones(shape, dtype=bool)       # 有被地圖點觀測到的區域；其餘是未知
        self.res = resolution
        self.rough_radius, self.rough_weight = 0.6, 40.0   # 窗口半徑(m)、密度 1.0 時的額外成本
        self.edge_margin, self.edge_weight = 0.8, 6.0                        # 已知區域邊界內縮多少公尺才可走（機器人偏離一點就會走出空拍機地圖）
        self.origin = np.asarray(origin_xy)           # 格 (0,0) 左下角的 (x,y) 公尺

    def world_to_cell(self, xy):
        c = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        return int(c[1]), int(c[0])                   # (row, col)

    def cell_to_world(self, rc):
        return self.origin + (np.array([rc[1], rc[0]]) + 0.5) * self.res

    def inflate(self, radius):
        return _dilate(self.data, int(np.ceil(radius / self.res)))

    def proximity_penalty(self, radius, safety, weight):
        """離膨脹後障礙越近成本越高（類似 Nav2 的 inflation layer）：讓路徑盡量走通道中央。
        從膨脹邊界往外每一圈（1 格）遞減，共 safety 公尺。"""
        occ = self.blocked(radius)
        pen = np.zeros(occ.shape)
        rings = max(int(np.ceil(safety / self.res)), 1)
        prev = occ
        for i in range(1, rings + 1):
            cur = _dilate(occ, i)
            pen[cur & ~prev] = weight * (1 - (i - 1) / rings)
            prev = cur
        # 地圖邊緣：離未知區域越近成本越高（機器人偏離一點就會走出空拍機地圖），但不封死
        unk = ~self.known
        erings = max(int(np.ceil(self.edge_margin / self.res / 2)), 1)
        for i in range(1, erings + 1):
            ring = _dilate(unk, 2 * i)
            pen[ring & ~unk] += self.edge_weight * (1 - (i - 1) / erings) / erings * 2
        # 粗糙度：局部障礙密度（碎石區地圖漏檢多，單格看起來可走，但整片都不好走）
        k = int(round(self.rough_radius / self.res))
        d = self.data.astype(float)
        ii = np.pad(d, ((k + 1, k), (k + 1, k))).cumsum(0).cumsum(1)
        H, W = d.shape
        box = ii[2*k+1:2*k+1+H, 2*k+1:2*k+1+W] - ii[:H, 2*k+1:2*k+1+W] - ii[2*k+1:2*k+1+H, :W] + ii[:H, :W]
        pen += self.rough_weight * box / (2 * k + 1) ** 2
        return pen

    def blocked(self, radius):
        """膨脹後的障礙 + 未知區域。未觀測到的地方不能假設可通行。"""
        return self.inflate(radius) | ~self.known


def build_grid(points_ground, cfg: MapConfig, margin=0.5):
    """points_ground: N x 3 (u, v, h) 公尺。"""
    uv, h = points_ground[:, :2], points_ground[:, 2]
    lo, hi = uv.min(0) - margin, uv.max(0) + margin
    shape = tuple(np.ceil((hi - lo)[::-1] / cfg.resolution).astype(int))
    grid = OccupancyGrid(shape, cfg.resolution, lo)
    sel = (h > cfg.obstacle_min_h) & (h < cfg.obstacle_max_h)
    counts = np.zeros(shape, dtype=int)
    for p in uv[sel]:
        r, c = grid.world_to_cell(p)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            counts[r, c] += 1
    # 稀疏點雲：一格內的點太少，改用鄰域累計。低矮物體只有頂面被看到，每格平均不到 1 個點
    k = cfg.count_window
    agg = sum(np.roll(np.roll(counts, dr, 0), dc, 1)
              for dr in range(-k, k + 1) for dc in range(-k, k + 1))
    # 該格自己要有點、鄰域累計夠多才算障礙：鄰域只用來濾掉孤立雜點，不會把障礙加粗（0.6 m 的窄門才過得去）
    grid.data = ((counts >= 1) & (agg >= cfg.min_points_per_cell)).astype(np.uint8)
    # 已知區域：有任何點落入的格子，先膨脹補洞（稀疏點雲）再侵蝕還原邊界（閉運算）
    seen = np.zeros(shape, dtype=bool)
    for p in uv:
        r, c = grid.world_to_cell(p)
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            seen[r, c] = True
    k = int(np.ceil(cfg.known_fill / cfg.resolution))
    grid.known = ~_dilate(~_dilate(seen, k), k)
    if cfg.min_density_frac > 0:
        # 點密度太低的地方（煙塵、陰影、無紋理）空拍機其實沒看清楚，不能當成空地：視為未知、禁止通行。
        # 門檻是相對於這張地圖自己的典型密度（不同場景、不同 SLAM 的絕對密度差很多）。
        pc = np.zeros(shape)
        for p in uv:
            r, c = grid.world_to_cell(p)
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                pc[r, c] += 1
        kd = int(np.ceil(cfg.density_window / cfg.resolution))
        win = sum(np.roll(np.roll(pc, dr, 0), dc, 1) for dr in range(-kd, kd + 1) for dc in range(-kd, kd + 1))
        med = np.median(win[win > 0])
        grid.known &= win >= cfg.min_density_frac * med
    return grid
