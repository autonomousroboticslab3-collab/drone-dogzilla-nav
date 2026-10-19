"""Pure pursuit 路徑追蹤。"""
import time
import numpy as np
from config import FollowerConfig


def follow(robot, path, cfg: FollowerConfig, realtime=False, max_steps=5000):
    goal = path[-1]
    for _ in range(max_steps):
        x, y, yaw = robot.get_pose()
        if np.hypot(*(goal - [x, y])) < cfg.goal_tol:
            robot.stop()
            return True
        # 找離機器人最近的路徑點，再往前取前視點
        i = int(np.argmin(np.linalg.norm(path - [x, y], axis=1)))
        j = i
        while j < len(path) - 1 and np.linalg.norm(path[j] - [x, y]) < cfg.lookahead:
            j += 1
        dx, dy = path[j] - [x, y]
        err = np.arctan2(np.sin(np.arctan2(dy, dx) - yaw),
                         np.cos(np.arctan2(dy, dx) - yaw))
        v = cfg.v_max * max(0.0, np.cos(err))        # 偏差大時先原地轉
        w = np.clip(2.0 * err, -cfg.w_max, cfg.w_max)
        robot.send_cmd(v, w)
        if realtime:
            time.sleep(1.0 / cfg.rate_hz)
    robot.stop()
    return False
