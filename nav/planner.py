"""8 鄰域 A*，輸入膨脹後的柵格，回傳世界座標路徑。"""
import heapq
import numpy as np


def astar(occ, start_rc, goal_rc, penalty=None):
    H, W = occ.shape
    if occ[start_rc] or occ[goal_rc]:
        return None
    nbrs = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    h = lambda a: np.hypot(a[0] - goal_rc[0], a[1] - goal_rc[1])
    pq, g, parent = [(h(start_rc), start_rc)], {start_rc: 0.0}, {}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal_rc:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        for dr, dc, w in nbrs:
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or occ[n]:
                continue
            if dr and dc and (occ[cur[0] + dr, cur[1]] or occ[cur[0], cur[1] + dc]):
                continue                                  # 不切角
            ng = g[cur] + w + (penalty[n] if penalty is not None else 0.0)
            if ng < g.get(n, 1e18):
                g[n], parent[n] = ng, cur
                heapq.heappush(pq, (ng + h(n), n))
    return None


def plan(grid, inflate_radius, start_xy, goal_xy, clear_radius=0.5, safety=0.30, weight=4.0, start_clear=None):
    """clear_radius：清空起點、終點周圍的障礙。目標物本身（箱子、人）會被建進地圖，
    若不清空會導致終點落在障礙內而無解。"""
    occ = grid.blocked(inflate_radius).astype(np.uint8)
    s, e = grid.world_to_cell(start_xy), grid.world_to_cell(goal_xy)
    for (r0, c0), rad in ((s, start_clear if start_clear is not None else clear_radius), (e, clear_radius)):
        k = int(np.ceil(rad / grid.res))
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = r0 + dr, c0 + dc
                if dr * dr + dc * dc <= k * k and 0 <= r < occ.shape[0] and 0 <= c < occ.shape[1]:
                    occ[r, c] = 0
    H, W = occ.shape
    if not all(0 <= r < H and 0 <= c < W for r, c in (s, e)):
        return None
    path = astar(occ, s, e, grid.proximity_penalty(inflate_radius, safety, weight))
    if path is None:
        return None
    return np.array([grid.cell_to_world(rc) for rc in path])


def reachable_area(grid, inflate_radius, start_xy, start_clear=0.4):
    """診斷用：從起點（清掉周圍 start_clear）出發，用 4 鄰域能到的面積 (m^2)，以及整張地圖可通行的面積。"""
    from collections import deque
    occ = grid.blocked(inflate_radius).astype(np.uint8)
    r0, c0 = grid.world_to_cell(start_xy)
    H, W = occ.shape
    k = int(np.ceil(start_clear / grid.res))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            r, c = r0 + dr, c0 + dc
            if dr * dr + dc * dc <= k * k and 0 <= r < H and 0 <= c < W:
                occ[r, c] = 0
    if not (0 <= r0 < H and 0 <= c0 < W):
        return 0.0, float((occ == 0).sum() * grid.res ** 2)
    seen = np.zeros_like(occ, bool)
    q = deque([(r0, c0)]); seen[r0, c0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and not occ[n] and not seen[n]:
                seen[n] = True; q.append(n)
    return float(seen.sum() * grid.res ** 2), float((occ == 0).sum() * grid.res ** 2)
