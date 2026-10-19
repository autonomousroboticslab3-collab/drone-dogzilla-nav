"""解析原廠 Gazebo .world 檔（只含 box / cylinder 的靜態障礙物）。

原廠專案：DOGZILLA/sim_env。座標：機器狗從世界原點朝 +x 出發，單位公尺。
"""
import importlib.util
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SIM_ENV = Path.home() / 'Downloads' / 'DOGZILLA' / 'sim_env'

# project3 的目標點不在 rl/scenes.py，而是寫在 benchmark_nav.py（它 import rclpy 無法直接載入）
PROJECT3_GOALS = [(5.9, 1.1), (0.0, 0.0), (5.6, -1.5), (0.3, 1.5), (3.1, 0.6), (1.5, -1.5), (5.9, 1.1)]


@dataclass
class Prim:
    kind: str                 # 'box' | 'cyl'
    c: np.ndarray             # 中心 (x, y)
    size: tuple               # box: (sx, sy)；cyl: (r,)
    yaw: float
    z0: float                 # 底部高度
    h: float                  # 高度
    name: str = ''
    pitch: float = 0.0        # 繞自身 y 軸傾斜（傾倒的牆板）；只用於 box
    blocks: bool = True       # 是否擋住機器人。懸空的樓板殘骸下方機器狗過得去（空拍機卻只看得到頂面）

    def half(self):
        """俯視投影的半邊長。傾斜時投影變長（保守估計）。"""
        if self.kind == 'cyl':
            return np.array([self.size[0]] * 2)
        sx, sy = np.array(self.size) / 2
        return np.array([sx * abs(math.cos(self.pitch)) + self.h / 2 * abs(math.sin(self.pitch)), sy])

    def to_local(self, xy):
        d = np.asarray(xy, float) - self.c
        co, si = math.cos(-self.yaw), math.sin(-self.yaw)
        return np.stack([d[..., 0] * co - d[..., 1] * si, d[..., 0] * si + d[..., 1] * co], -1)

    def clearance(self, xy):
        """到物體邊緣的距離（公尺），在內部為負。"""
        p = self.to_local(xy)
        if self.kind == 'cyl':
            return np.linalg.norm(p, axis=-1) - self.size[0]
        q = np.abs(p) - self.half()
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, -1), 0)

    def corners(self):
        """俯視輪廓（box 四角；cyl 用 24 邊形近似），世界座標。"""
        if self.kind == 'cyl':
            t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            loc = np.c_[self.size[0] * np.cos(t), self.size[0] * np.sin(t)]
        else:
            sx, sy = self.half()
            loc = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]])
        co, si = math.cos(self.yaw), math.sin(self.yaw)
        return loc @ np.array([[co, si], [-si, co]]) + self.c


@dataclass
class Scene:
    name: str
    prims: list = field(default_factory=list)
    goal_marker: np.ndarray = None      # .world 裡綠色目標標記的位置
    goals: list = field(default_factory=list)   # benchmark 用的目標點清單
    start: np.ndarray = field(default_factory=lambda: np.zeros(2))
    start_yaw: float = 0.0
    dust: list = field(default_factory=list)    # 煙塵區 [(x, y, 半徑)]：特徵點變少、目標偵測會漏

    def bounds(self):
        pts = np.vstack([p.corners() for p in self.prims])
        return pts.min(0), pts.max(0)

    def clearance(self, xy, physical=True):
        """到最近物體的距離。physical=True 只算會擋住機器人的；False 算全部（空拍機看得到的）。"""
        return np.min([p.clearance(xy) for p in self.prims if p.blocks or not physical], axis=0)

    def in_dust(self, xy):
        xy = np.atleast_2d(xy)
        m = np.zeros(len(xy), bool)
        for x, y, r in self.dust:
            m |= np.hypot(xy[:, 0] - x, xy[:, 1] - y) < r
        return m


def _pose(el):
    if el is None or not (el.text or '').strip():
        return np.zeros(6)
    return np.array([float(v) for v in el.text.split()])


def _floats(el):
    return [float(v) for v in el.text.split()]


def parse_world(path):
    root = ET.parse(path).getroot()
    prims, marker = [], None
    for m in root.iter('model'):
        name = m.get('name', '')
        if name == 'goal_marker':
            marker = _pose(m.find('pose'))[:2]
            continue
        geom_el = m.find('./link/collision/geometry')
        if name == 'ground_plane' or geom_el is None:
            continue
        mp = _pose(m.find('pose'))
        lp = _pose(m.find('./link/pose'))
        cp = _pose(m.find('./link/collision/pose'))
        # 只處理 yaw + 平移（原廠場景沒有 roll/pitch）
        yaw = mp[5] + lp[5] + cp[5]
        loc = lp[:2] + cp[:2]
        co, si = math.cos(mp[5]), math.sin(mp[5])
        c = mp[:2] + np.array([co * loc[0] - si * loc[1], si * loc[0] + co * loc[1]])
        zc = mp[2] + lp[2] + cp[2]
        if geom_el.find('box') is not None:
            sx, sy, sz = _floats(geom_el.find('box/size'))
            prims.append(Prim('box', c, (sx, sy), yaw, zc - sz / 2, sz, name, pitch=mp[4] + lp[4] + cp[4]))
        elif geom_el.find('cylinder') is not None:
            r = float(geom_el.find('cylinder/radius').text)
            ln = float(geom_el.find('cylinder/length').text)
            prims.append(Prim('cyl', c, (r,), yaw, zc - ln / 2, ln, name))
    return prims, marker


def _load_scenes_py(sim_env):
    spec = importlib.util.spec_from_file_location('dz_scenes', Path(sim_env) / 'rl' / 'scenes.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCENES


def load_scene(name, sim_env=DEFAULT_SIM_ENV):
    if name.startswith('site'):                        # site 或 site:<seed>：較真實的災區
        return make_site(int(name.split(':')[1]) if ':' in name else 0)
    if name.startswith('quake'):                       # quake 或 quake:<seed>
        return make_quake(int(name.split(':')[1]) if ':' in name else 0)
    sim_env = Path(sim_env)
    prims, marker = parse_world(sim_env / 'worlds' / f'{name}.world')
    if name == 'project3':
        goals = PROJECT3_GOALS
    else:
        try:
            goals = _load_scenes_py(sim_env)[name]['goals']
        except Exception:
            goals = [tuple(marker)] if marker is not None else []
    return Scene(name, prims, marker, [tuple(g) for g in goals])


# ======================= 震後場景（程序化生成） =======================
def _box(c, size, yaw, h, name):
    return Prim('box', np.asarray(c, float), tuple(size), yaw, 0.0, h, name)


def _wall_pieces(p0, p1, gaps, thick, h, name):
    """沿線段 p0->p1 放牆，gaps 是 [(起點比例 s0, 終點比例 s1)] 的缺口。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L, yaw = np.linalg.norm(p1 - p0), math.atan2(*(p1 - p0)[::-1])
    cuts = [0.0]
    for s0, s1 in sorted(gaps):
        cuts += [s0, s1]
    cuts.append(1.0)
    out = []
    for i in range(0, len(cuts), 2):
        a, b = cuts[i], cuts[i + 1]
        if (b - a) * L > 0.15:
            c = p0 + (p1 - p0) * (a + b) / 2
            out.append(_box(c, ((b - a) * L, thick), yaw, h, f'{name}_{i // 2}'))
    return out


def _free_mask(scene, lo, hi, res, radius):
    xs, ys = np.arange(lo[0], hi[0], res) + res / 2, np.arange(lo[1], hi[1], res) + res / 2
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X, Y], -1).reshape(-1, 2)
    return (scene.clearance(pts) > radius).reshape(X.shape), xs, ys


def _reachable(free, start_rc):
    from collections import deque
    seen = np.zeros_like(free)
    if not free[start_rc]:
        return seen
    q = deque([start_rc]); seen[start_rc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n] and not seen[n]:
                seen[n] = True; q.append(n)
    return seen


def make_quake(seed=0, robot_radius=0.15, n_goals=5):
    """倒塌的建築：外牆與隔間牆有缺口，缺口堆滿碎石，另有傾倒的牆板與橫樑。
    10 m x 8 m，機器狗從 (0,0) 朝 +x，生還者標記放在最深處的房間。
    會一直重抽直到目標可達（以機器人半徑膨脹後的連通性判斷）。"""
    for attempt in range(200):
        rng = np.random.default_rng(seed * 1000 + attempt)
        prims, T, H = [], 0.2, 2.0
        X0, X1, Y0, Y1 = -1.0, 9.0, -4.0, 4.0

        def gaps(n, wmin, wmax, L):
            out = []
            for _ in range(n):
                w = rng.uniform(wmin, wmax) / L
                s = rng.uniform(0.1, 0.9 - w)
                if all(s > g1 + 0.05 or s + w < g0 - 0.05 for g0, g1 in out):
                    out.append((s, s + w))
            return out

        # 外牆：西牆保留 1.6 m 入口（機器狗起點在入口內側），其餘各有 1~2 個坍塌缺口
        west = [(0.5 - 0.8 / 8, 0.5 + 0.8 / 8)]
        sides = [((X0, Y0), (X0, Y1), west), ((X1, Y0), (X1, Y1), gaps(1, 1.0, 2.0, 8)),
                 ((X0, Y0), (X1, Y0), gaps(2, 1.0, 2.0, 10)), ((X0, Y1), (X1, Y1), gaps(2, 1.0, 2.0, 10))]
        for i, (a, b, g) in enumerate(sides):
            prims += _wall_pieces(a, b, g, T, H, f'outer{i}')
        # 兩道隔間牆，各有 1~2 個缺口（0.7~1.4 m，可能被碎石擋住）
        part_gaps = []
        for i, x in enumerate((3.0, 6.0)):
            g = gaps(int(rng.integers(1, 3)), 0.7, 1.4, 8)
            part_gaps.append((x, g))
            prims += _wall_pieces((x, Y0), (x, Y1), g, T, H, f'part{i}')
        # 碎石：落在缺口附近（有些缺口幾乎被堵死，逼路徑繞行）
        for x, g in part_gaps:
            for s0, s1 in g:
                cy = Y0 + 8 * (s0 + s1) / 2
                for k in range(int(rng.integers(3, 9))):
                    sz = rng.uniform(0.12, 0.45, 2)
                    c = [x + rng.normal(0, 0.35), cy + rng.uniform(-0.5, 0.5) * 8 * (s1 - s0) * 1.1]
                    prims.append(_box(c, sz, rng.uniform(0, math.pi), rng.uniform(0.12, 0.5), f'rubble_{len(prims)}'))
        # 傾倒的牆板（躺在地上，斜著）與橫樑
        for k in range(int(rng.integers(2, 4))):
            c = [rng.uniform(0.5, 8.5), rng.uniform(-3.3, 3.3)]
            prims.append(_box(c, (rng.uniform(1.0, 1.8), rng.uniform(0.6, 1.0)), rng.uniform(0, math.pi),
                              rng.uniform(0.15, 0.3), f'slab_{k}'))
        for k in range(int(rng.integers(2, 4))):
            c = [rng.uniform(0.5, 8.5), rng.uniform(-3.3, 3.3)]
            prims.append(_box(c, (rng.uniform(1.5, 2.5), 0.2), rng.uniform(0, math.pi), 0.25, f'beam_{k}'))
        # 零星碎石散佈在房間裡
        for k in range(int(rng.integers(10, 20))):
            c = [rng.uniform(-0.5, 8.5), rng.uniform(-3.5, 3.5)]
            if np.hypot(*c) < 0.8:
                continue                                   # 起點周圍留空
            if rng.random() < 0.5:
                prims.append(_box(c, rng.uniform(0.1, 0.35, 2), rng.uniform(0, math.pi),
                                  rng.uniform(0.12, 0.4), f'debris_{k}'))
            else:
                prims.append(Prim('cyl', np.array(c), (rng.uniform(0.05, 0.15),), 0.0, 0.0,
                                  rng.uniform(0.12, 0.4), f'debris_{k}'))

        scene = Scene(f'quake:{seed}', prims)
        lo, hi = np.array([X0 - 0.5, Y0 - 0.5]), np.array([X1 + 0.5, Y1 + 0.5])
        res = 0.05
        free, xs, ys = _free_mask(scene, lo, hi, res, robot_radius)
        sr, sc = int((0 - lo[1]) / res), int((0 - lo[0]) / res)
        reach = _reachable(free, (sr, sc))
        # 生還者：最深處房間 (x > 6.5) 中可達且離障礙 >= 0.4 m 的點
        cand = np.argwhere(reach)
        cand = [(r, c) for r, c in cand if xs[c] > 6.5 and xs[c] < X1 - 0.5 and abs(ys[r]) < Y1 - 0.5]
        if not cand:
            continue
        r, c = cand[int(rng.integers(len(cand)))]
        scene.goal_marker = np.array([xs[c], ys[r]])
        # 其他測試目標：可達區域內隨機點
        allc = np.argwhere(reach)
        picks = allc[rng.choice(len(allc), n_goals - 1, replace=False)]
        scene.goals = [tuple(scene.goal_marker)] + [(float(xs[c_]), float(ys[r_])) for r_, c_ in picks]
        return scene
    raise RuntimeError('找不到可達的震後場景，請換 seed')


def _slab(c, size, yaw, h, pitch, name, blocks=True, lift=0.0):
    """傾斜的板：低的一端貼地，另一端被抬高（斜靠在牆或碎石上）。lift = 整塊再抬高多少（懸空樓板）。"""
    sx, hz = size[0] / 2, h / 2
    zc = sx * abs(math.sin(pitch)) + hz * abs(math.cos(pitch)) + lift
    return Prim('box', np.asarray(c, float), tuple(size), yaw, zc - hz, h, name, pitch=pitch, blocks=blocks)


def make_site(seed=0, robot_radius=0.15, n_goals=6):
    """較真實的災區（約 17 x 15 m）：街道 + 一棟部分倒塌的建築。
      * 建築外牆、隔間牆有大小不一的坍塌缺口，缺口堆滿碎石
      * 斜靠的牆板（傾斜、機器狗爬不上去）、懸空的樓板殘骸（機器狗可從下方過，空拍機卻只看得到頂面）
      * 街道上有車輛、傾倒的電線桿、碎石堆
      * 煙塵區：空拍機特徵點變少、目標偵測會漏
    機器狗從街道 (0,0) 朝 +x 出發，生還者標記在建築最深處。重抽直到目標可達。"""
    for attempt in range(300):
        rng = np.random.default_rng(seed * 1000 + attempt)
        prims, T, H = [], 0.25, 2.8
        X0, X1, Y0, Y1 = 3.0, 13.0, -5.0, 5.0

        def gaps(n, wmin, wmax, L):
            out = []
            for _ in range(n):
                w_ = rng.uniform(wmin, wmax) / L
                s = rng.uniform(0.08, 0.92 - w_)
                if all(s > g1 + 0.05 or s + w_ < g0 - 0.05 for g0, g1 in out):
                    out.append((s, s + w_))
            return out

        def rubble_at(cx, cy, n, spread, big=False):
            for _ in range(n):
                lo_, hi_ = (0.25, 0.7) if big and rng.random() < 0.3 else (0.1, 0.4)
                p = _box([cx + rng.normal(0, spread), cy + rng.normal(0, spread)], rng.uniform(lo_, hi_, 2),
                         rng.uniform(0, math.pi), rng.uniform(0.12, 0.6), f'rubble_{len(prims)}')
                if np.hypot(*p.c) > 1.0:
                    prims.append(p)

        ent = [(0.5 - 1.0 / 10, 0.5 + 1.0 / 10)]                       # 西牆入口 2 m
        collapse = gaps(1, 2.0, 3.5, 10)
        west = ent + [g for g in collapse if g[0] > ent[0][1] + 0.05 or g[1] < ent[0][0] - 0.05]
        sides = [((X0, Y0), (X0, Y1), west), ((X1, Y0), (X1, Y1), gaps(1, 1.5, 3.0, 10)),
                 ((X0, Y0), (X1, Y0), gaps(2, 1.5, 3.0, 10)), ((X0, Y1), (X1, Y1), gaps(2, 1.5, 3.0, 10))]
        cand_gaps = []
        for i, ((ax, ay), (bx, by), g) in enumerate(sides):
            prims += _wall_pieces((ax, ay), (bx, by), g, T, H, f'outer{i}')
            L = math.hypot(bx - ax, by - ay)
            for s0, s1 in g:
                m = (s0 + s1) / 2
                cand_gaps.append((ax + (bx - ax) * m, ay + (by - ay) * m, (s1 - s0) * L))
        for i, x in enumerate((6.3, 9.7)):                            # 隔間牆
            g = gaps(int(rng.integers(1, 3)), 0.8, 1.5, 10)
            prims += _wall_pieces((x, Y0), (x, Y1), g, T, H, f'part{i}')
            for s0, s1 in g:
                cand_gaps.append((x, Y0 + 10 * (s0 + s1) / 2, 10 * (s1 - s0)))
        g = gaps(1, 0.9, 1.4, 3.3)                                    # 最深處房間的橫隔牆
        prims += _wall_pieces((9.7, 0.0), (X1, 0.0), g, T, H, 'back')
        for cx, cy, wd in cand_gaps:                                  # 缺口堆碎石（有的堵死）
            rubble_at(cx, cy, int(rng.integers(3, 11)), 0.3 + 0.1 * wd, big=True)
        # 斜靠的牆板：倒在建築外兩個坍塌處
        outer = [c for c in cand_gaps if c[0] in (X0, X1) or c[1] in (Y0, Y1)]
        for k in range(min(2, len(outer))):
            cx, cy, _ = outer[int(rng.integers(len(outer)))]
            prims.append(_slab([cx + rng.normal(0, .6), cy + rng.normal(0, .6)],
                               (rng.uniform(2.2, 3.4), rng.uniform(1.0, 1.6)), rng.uniform(0, math.pi), 0.2,
                               rng.choice([-1, 1]) * rng.uniform(0.25, 0.5), f'leaning_{k}'))
        # 懸空的樓板殘骸：下方 >= 0.45 m，機器狗過得去，但空拍機只看到頂面
        for k in range(int(rng.integers(1, 3))):
            prims.append(_slab([rng.uniform(4.0, 12.0), rng.uniform(-3.5, 3.5)], (rng.uniform(2.0, 3.0), rng.uniform(1.6, 2.4)),
                               rng.uniform(0, math.pi), 0.2, 0.0, f'floor_{k}', blocks=False, lift=0.5))
        # 街道：車輛、電線桿、碎石堆
        for k in range(int(rng.integers(1, 3))):
            c = [rng.uniform(0.0, 2.0), rng.choice([-1, 1]) * rng.uniform(3.0, 6.0)]
            prims.append(_box(c, (4.2, 1.8), rng.normal(0, 0.4) + rng.choice([0, math.pi / 2]), 1.4, f'car_{k}'))
        prims.append(_box([rng.uniform(1.0, 2.5), rng.uniform(-2.0, 2.0)], (rng.uniform(4.0, 6.0), 0.22),
                          rng.uniform(0.4, 1.2), 0.25, 'pole_0'))
        for k in range(2):
            rubble_at(rng.uniform(1.5, 2.8), rng.choice([-1, 1]) * rng.uniform(3.0, 5.0), int(rng.integers(8, 18)), 0.7)
        # 零星碎石
        for k in range(int(rng.integers(25, 45))):
            c = [rng.uniform(-1.0, 13.5), rng.uniform(-6.5, 6.5)]
            if np.hypot(*c) < 1.0:
                continue
            if rng.random() < 0.6:
                prims.append(_box(c, rng.uniform(0.1, 0.4, 2), rng.uniform(0, math.pi), rng.uniform(0.12, 0.45), f'debris_{k}'))
            else:
                prims.append(Prim('cyl', np.array(c), (rng.uniform(0.05, 0.15),), 0.0, 0.0, rng.uniform(0.12, 0.45), f'debris_{k}'))

        scene = Scene(f'site:{seed}', prims)
        scene.dust = [(cx + rng.normal(0, 0.5), cy + rng.normal(0, 0.5), rng.uniform(1.5, 2.5))
                      for cx, cy, _ in [cand_gaps[int(rng.integers(len(cand_gaps)))] for _ in range(3)]]
        lo, hi = scene.bounds()
        res = 0.05
        free, xs, ys = _free_mask(scene, lo - 0.3, hi + 0.3, res, robot_radius)
        sr, sc = int((0 - (lo[1] - 0.3)) / res), int((0 - (lo[0] - 0.3)) / res)
        reach = _reachable(free, (sr, sc))
        cand = [(r, c) for r, c in np.argwhere(reach) if xs[c] > 10.3 and xs[c] < X1 - 0.6 and abs(ys[r]) < Y1 - 0.6]
        if len(cand) < 30:
            continue
        r, c = cand[int(rng.integers(len(cand)))]
        scene.goal_marker = np.array([xs[c], ys[r]])
        allc = np.argwhere(reach)
        picks = allc[rng.choice(len(allc), n_goals - 1, replace=False)]
        scene.goals = [tuple(scene.goal_marker)] + [(float(xs[c_]), float(ys[r_])) for r_, c_ in picks]
        return scene
    raise RuntimeError('找不到可達的災區場景，請換 seed')


def write_world(scene, path, sim_env=DEFAULT_SIM_ENV):
    """輸出成 Gazebo .world。檔頭（地面、光源、物理設定）直接沿用原廠 corridor.world，
    只換掉障礙物與目標標記，所以和原廠 tools/make_world.py 的產物格式一致。"""
    src = (Path(sim_env) / 'worlds' / 'corridor.world').read_text()
    head = src[:src.index("<!-- ")]                      # 原廠檔頭到第一個註解為止（障礙物從這裡開始）

    def model(p):
        zc = p.z0 + p.h / 2
        if p.kind == 'box':
            g = f'<box><size>{p.size[0]:.3f} {p.size[1]:.3f} {p.h:.3f}</size></box>'
        else:
            g = f'<cylinder><radius>{p.size[0]:.3f}</radius><length>{p.h:.3f}</length></cylinder>'
        mat = '<ambient>0.55 0.5 0.45 1</ambient><diffuse>0.55 0.5 0.45 1</diffuse>'
        return (f"<model name='{p.name}'><static>1</static><pose>{p.c[0]:.3f} {p.c[1]:.3f} {zc:.3f} 0 {p.pitch:.4f} {p.yaw:.4f}</pose>"
                f"<link name='l'>\n<collision name='c'><geometry>{g}</geometry></collision>\n"
                f"<visual name='v'><geometry>{g}</geometry><material>{mat}</material></visual></link></model>")
    gm = scene.goal_marker
    marker = (f"<model name='goal_marker'><static>1</static><pose>{gm[0]:.3f} {gm[1]:.3f} 0 0 0 0</pose><link name='l'>\n"
              "<visual name='pad'><pose>0 0 0.005 0 0 0</pose><geometry><cylinder><radius>0.2</radius><length>0.01</length>"
              "</cylinder></geometry><material><ambient>0.15 0.6 0.25 1</ambient><diffuse>0.15 0.6 0.25 1</diffuse>"
              "</material></visual>\n"
              "<visual name='pole'><pose>0 0 0.3 0 0 0</pose><geometry><cylinder><radius>0.03</radius><length>0.6</length>"
              "</cylinder></geometry><material><ambient>0.15 0.6 0.25 1</ambient><diffuse>0.15 0.6 0.25 1</diffuse>"
              "</material></visual></link></model>")
    body = '\n'.join(model(p) for p in scene.prims)
    Path(path).write_text(f"{head}<!-- 由 nav/worlds.py 的 make_quake 產生（震後場景） -->\n{body}\n{marker}\n</world>\n</sdf>\n")
