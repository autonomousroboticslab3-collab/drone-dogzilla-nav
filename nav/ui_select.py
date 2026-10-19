"""使用者點選目的地。matplotlib 為選用依賴；無法顯示時用 fallback 座標。"""
import numpy as np


def pick_goal(grid, targets, start_xy=None, fallback=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[ui] 無 matplotlib，使用 fallback 目的地', fallback)
        return np.asarray(fallback)
    fig, ax = plt.subplots()
    ext = [grid.origin[0], grid.origin[0] + grid.data.shape[1] * grid.res,
           grid.origin[1], grid.origin[1] + grid.data.shape[0] * grid.res]
    ax.imshow(1 - grid.data, cmap='gray', origin='lower', extent=ext)
    for t in targets:
        ax.plot(*t, 'r*', ms=15)
    if start_xy is not None:
        ax.plot(*start_xy, 'bo')
    ax.set_title('點選目的地（紅星 = 自動偵測目標）')
    pt = plt.ginput(1, timeout=0)
    plt.close(fig)
    return np.array(pt[0]) if pt else None
