#!/usr/bin/env python3
"""Gazebo 橋接：在 dogzilla-sim 容器內執行（ROS 2 Humble）。

  pose                    印出機器人目前位姿：Gazebo 真值（世界座標）與 SLAM 的 map->base_link
  go --plan plan.json     把空拍機端算好的目標（機器人座標系偏移量）送給機器人，等它走完，回報軌跡

目標一律是「相對於出發當下機器人」的偏移量（x 朝前、y 朝左），這是真實系統唯一需要的對齊資訊：
  * Nav2 模式：用機器人自己的 SLAM 位姿（map->base_link）換成 map 座標，不用任何真值。
  * RL 模式：原廠 RL 控制器吃世界座標（用 Gazebo 真值定位，是模擬捷徑），所以改用真值位姿換算。
"""
import argparse
import json
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Bridge(Node):
    def __init__(self):
        super().__init__('gz_bridge')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.gt, self.flipped, self.trace, self.tilt_cos = None, False, [], 1.0
        self.tf_log, self._tf_t = [], -1e9       # 診斷：(模擬時間, 真值xyyaw, map→odom xyyaw, map→base xyyaw)
        self.create_subscription(Odometry, '/demo/odom/ground_truth', self.on_gt, 10)
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.rl_goal = self.create_publisher(PoseStamped, '/rl_goal', 10)
        self.scan = None
        self._scan_t = {}
        for topic in ('/scan', '/scan_filtered'):              # 優先用去掉地面點的 /scan_filtered
            self.create_subscription(LaserScan, topic, lambda m, t=topic: self.on_scan(t, m), qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.rl_status = 'IDLE'
        self.create_subscription(String, '/rl_status', lambda m: setattr(self, 'rl_status', m.data), 10)

    def on_gt(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.gt = (p.x, p.y, yaw_of(q))
        self.tilt_cos = 1 - 2 * (q.x * q.x + q.y * q.y)    # 機身 z 軸與世界 z 軸夾角的 cos
        self.log_tf()
        if self.tilt_cos < 0.5:                             # 夾角 > 60° = 翻倒
            self.flipped = True
        if not self.trace or math.hypot(p.x - self.trace[-1][0], p.y - self.trace[-1][1]) > 0.02:
            self.trace.append((p.x, p.y))

    def log_tf(self):
        t = self.sim_now()
        if t - self._tf_t < 0.5 or self.gt is None:
            return
        self._tf_t = t
        try:
            a = self.tfbuf.lookup_transform('map', 'odom', rclpy.time.Time()).transform
            b_ = self.tfbuf.lookup_transform('map', 'base_link', rclpy.time.Time()).transform
        except Exception:
            return
        self.tf_log.append([t, *self.gt, a.translation.x, a.translation.y, yaw_of(a.rotation),
                            b_.translation.x, b_.translation.y, yaw_of(b_.rotation)])

    def on_scan(self, topic, m):
        self._scan_t[topic] = time.time()
        if topic == '/scan_filtered' or time.time() - self._scan_t.get('/scan_filtered', 0) > 2.0:
            self.scan = m

    def cmd_pub_ready(self):
        self.spin_for(0.5)

    def send_vel(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait_ready(self, need_tf):
        while self.gt is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        while need_tf and not self.tfbuf.can_transform('map', 'base_link', rclpy.time.Time()):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.spin_for(0.5)

    def map_pose(self):
        t = self.tfbuf.lookup_transform('map', 'base_link', rclpy.time.Time())
        p = t.transform.translation
        return p.x, p.y, yaw_of(t.transform.rotation)

    def detect_mode(self):
        return 'rl' if self.get_publishers_info_by_topic('/rl_status') else 'nav2'


def to_frame(base, d):
    """機器人座標系偏移量 d（前, 左）-> 以 base=(x,y,yaw) 為原點的外部座標。"""
    c, s = math.cos(base[2]), math.sin(base[2])
    return base[0] + c * d[0] - s * d[1], base[1] + s * d[0] + c * d[1]


def leg_rl(b, gx, gy, timeout):
    def send(x, y):
        g = PoseStamped(); g.header.frame_id = 'world'
        g.pose.position.x, g.pose.position.y = x, y
        b.rl_goal.publish(g)
    b.rl_status = 'IDLE'
    t0 = time.time()
    send(gx, gy)
    b.spin_for(0.5)
    while b.rl_status != 'REACHED' and time.time() - t0 < timeout and not b.flipped:
        rclpy.spin_once(b, timeout_sec=0.05)
    ok = b.rl_status == 'REACHED'
    send(float('nan'), float('nan'))
    b.spin_for(1.0)
    return 'REACHED' if ok else ('FLIPPED' if b.flipped else 'TIMEOUT')


def leg_nav2(b, mx, my, timeout):
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = mx, my
    goal.pose.pose.orientation.w = 1.0
    t0 = time.time()
    fut = b.ac.send_goal_async(goal)
    while not fut.done():
        rclpy.spin_once(b, timeout_sec=0.05)
    h = fut.result()
    if not h.accepted:
        return 'REJECTED'
    res = h.get_result_async()
    while not res.done() and time.time() - t0 < timeout and not b.flipped:
        rclpy.spin_once(b, timeout_sec=0.05)
    if not res.done():
        h.cancel_goal_async()
        b.spin_for(2.0)
        return 'FLIPPED' if b.flipped else 'TIMEOUT'
    return {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(res.result().status, str(res.result().status))


def heading_clearance(ranges, a0, da, half_w=0.2, cap=3.0):
    """各個方向的「走廊」淨空：沿該方向、寬 2*half_w 的帶狀範圍內，最近障礙的前向距離。
    無回波（inf）當作一路暢通。回傳 [(角度 rad, 淨空 m)]。"""
    r = np.asarray(ranges, float)
    ang = a0 + da * np.arange(len(r))
    r = np.where(np.isfinite(r) & (r > 0.12), np.minimum(r, cap), np.where(np.isfinite(r), 0.0, cap))
    x, y = r * np.cos(ang), r * np.sin(ang)
    out = []
    for th in np.deg2rad(np.arange(-180, 180, 10)):
        c, s = math.cos(th), math.sin(th)
        fwd, lat = x * c + y * s, -x * s + y * c
        m = (np.abs(lat) < half_w) & (fwd > 0.0)
        out.append((float(th), float(fwd[m].min()) if m.any() else cap))
    return out


def escape_reactive(b, max_sim_s=240.0, need_move=1.0, max_move=3.0):
    """不依賴任何地圖的反應式脫困：每 0.1 s 看一次 LiDAR，往最遠的走廊走，直到周圍夠開闊。
    原因：機器人在碎石區繞久了，自己的 SLAM / costmap 常已壞掉（認為被障礙包圍），Nav2 規劃不出路。"""
    b.cmd_pub_ready()
    x0, y0 = b.gt[0], b.gt[1]
    t0 = b.sim_now()
    prev_th, log = 0.0, []
    while b.sim_now() - t0 < max_sim_s and not b.flipped:
        b.spin_for(0.1)
        if b.scan is None:
            continue
        if b.tilt_cos < 0.82:                    # 傾斜超過約 35°：再硬闖就會翻倒，立刻停下
            b.send_vel(0.0, 0.0)
            return 'UNSTABLE', math.hypot(b.gt[0] - x0, b.gt[1] - y0), 0.0
        s = b.scan
        cl = heading_clearance(s.ranges, s.angle_min, s.angle_increment)
        # 偏好少轉彎、也偏好延續上一個方向（避免左右猶豫）
        th, c = max(cl, key=lambda tc: tc[1] - 0.12 * abs(tc[0]) - 0.10 * abs(tc[0] - prev_th))
        rr = np.asarray(s.ranges, float)
        openness = float(np.mean(~np.isfinite(rr) | (rr > 1.5)))
        moved = math.hypot(b.gt[0] - x0, b.gt[1] - y0)
        if (openness >= 0.5 and moved >= need_move) or moved >= max_move:
            b.send_vel(0.0, 0.0)
            return 'ESCAPED', moved, openness
        if c < 0.30:
            v, w = 0.0, 0.6                      # 四面都近：原地轉著找縫
        elif abs(th) > 0.4:
            v, w = 0.0, 0.6 * (1 if th > 0 else -1)
        else:
            # 原廠安全濾波會把線速度乘 max(0.15, 1-|w|)，而指令低於約 0.06 m/s 機器狗根本不前進，
            # 所以線速度要給足、轉彎幅度要小，濾波後才還走得動
            v, w = 0.15, max(-0.3, min(0.3, 1.0 * th))
        prev_th = th
        b.send_vel(v, w)
        log.append((round(b.sim_now() - t0, 1), round(th, 2), round(c, 2), round(openness, 2)))
    b.send_vel(0.0, 0.0)
    return 'FAILED', math.hypot(b.gt[0] - x0, b.gt[1] - y0), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pose', 'go', 'escape'])
    ap.add_argument('--plan')
    ap.add_argument('--mode', default='auto', choices=['auto', 'rl', 'nav2'])
    ap.add_argument('--waypoints', action='store_true', help='依序送途經點，不只送最終目標')
    ap.add_argument('--drone-fix', action='store_true',
                    help='每段出發前用「空拍機看到的機器人位置」修正：機器人自己的 SLAM 座標會漂移，不能只在起點對齊一次')
    ap.add_argument('--fix-sigma', type=float, default=0.10, help='空拍機觀測機器人的位置雜訊 (m)；朝向雜訊固定 3°')
    ap.add_argument('--stop-on-fail', action='store_true', help='某段失敗就停下回報，由空拍機端重新規劃')
    ap.add_argument('--timeout', type=float, default=600.0, help='整趟逾時（秒）')
    ap.add_argument('--final-tol', type=float, default=0.35, help='最後一點：空拍機觀測誤差超過此值就要求校正 (m)')
    ap.add_argument('--leg-timeout', type=float, default=240.0)
    ap.add_argument('--tries', type=int, default=6, help='escape：最多試幾個脫困點')
    ap.add_argument('--out', default='/tmp/gz_result.json')
    a = ap.parse_args()
    rclpy.init()
    b = Bridge()
    if a.cmd == 'pose':
        b.wait_ready(need_tf=True)
        print('RESULT:' + json.dumps(dict(gt=b.gt, map=b.map_pose(), mode=b.detect_mode())))
        return
    if a.cmd == 'escape':
        b.wait_ready(need_tf=False)
        t_end = time.time() + 15
        while b.scan is None and time.time() < t_end:
            rclpy.spin_once(b, timeout_sec=0.1)
        if b.scan is None:
            print('RESULT:' + json.dumps(dict(status='NO_SCAN'))); return
        b.trace, t0s = [(b.gt[0], b.gt[1])], b.sim_now()
        status, moved, openness = escape_reactive(b)
        print(f'[bridge] 反應式脫困：{status}，移動 {moved:.2f} m，周圍開闊度 {openness:.2f}', flush=True)
        out = dict(status=status, moved=moved, openness=openness, trace=b.trace, duration=b.sim_now() - t0s,
                   flipped=b.flipped, final_gt=b.gt)
        json.dump(out, open(a.out, 'w'))
        print('RESULT:' + json.dumps(dict(status=status, moved=round(moved, 2), flipped=b.flipped)))
        return
    plan = json.load(open(a.plan))
    mode = b.detect_mode() if a.mode == 'auto' else a.mode
    if mode == 'nav2':
        b.ac.wait_for_server()
    b.wait_ready(need_tf=True)
    # 用「使用者標記當下」的位姿換算（plan 內附）。機器人在標記後可能又移動了，
    # 目標在 map / 世界座標系是固定的，不該跟著機器人現在的位置跑。沒附就退回用現在的位姿。
    base_gt = tuple(plan['base_gt']) if 'base_gt' in plan else b.gt
    base_map = tuple(plan['base_map']) if 'base_map' in plan else b.map_pose()
    wps = plan['waypoints_body'] if a.waypoints else [plan['goal_body']]
    b.trace, t0w, t0s = [(base_gt[0], base_gt[1])], time.time(), b.sim_now()
    legs, status, failed_wp = [], 'SUCCEEDED', None
    n_retry = n_fix = 0
    for i, d in enumerate(wps):
        last = i == len(wps) - 1
        tmo = min(a.leg_timeout, max(10.0, a.timeout - (time.time() - t0w)))
        if mode == 'rl':
            gx, gy = to_frame(base_gt, d)
            st = leg_rl(b, gx, gy, tmo)
            ok = st == 'REACHED'
        else:
            def fixed_goal(d):
                if not a.drone_fix:
                    return to_frame(base_map, d)
                # 空拍機此刻看到的機器人位姿（模擬：真值 + 雜訊）。途經點的世界座標由「標記當下的位姿 + 目標偏移」決定，
                # 只算「從現在的位置還要走多遠」，再交給機器人的局部 SLAM 座標去執行，所以 SLAM 全域漂移不會累積。
                import random
                ox, oy = b.gt[0] + random.gauss(0, a.fix_sigma), b.gt[1] + random.gauss(0, a.fix_sigma)
                oyaw = b.gt[2] + math.radians(random.gauss(0, 3.0))
                wx, wy = to_frame(base_gt, d)
                dx, dy = wx - ox, wy - oy
                c_, s_ = math.cos(-oyaw), math.sin(-oyaw)
                return to_frame(b.map_pose(), (c_ * dx - s_ * dy, s_ * dx + c_ * dy))
            gx, gy = fixed_goal(d)
            st = leg_nav2(b, gx, gy, tmo)
            if st == 'ABORTED' and not b.flipped and a.drone_fix:
                # 暫時性卡住（Nav2 "Failed to make progress"）：用空拍機重新量一次位置再試一次，再失敗才算失敗
                print('   [retry] 該段被中止，重新定位後重試一次', flush=True)
                n_retry += 1
                gx, gy = fixed_goal(d)
                st = leg_nav2(b, gx, gy, max(10.0, tmo / 2))
            # 最後一點：Nav2 以為到了，但 SLAM 可能滑掉；用空拍機觀測再校正，最多 2 次
            for _ in range(2 if (last and a.drone_fix and st == 'SUCCEEDED') else 0):
                wgt = to_frame(base_gt, d)
                if math.hypot(b.gt[0] - wgt[0], b.gt[1] - wgt[1]) < a.final_tol:
                    break
                print('   [校正] 最後一點離期望仍有誤差，空拍機要求再修正', flush=True)
                n_fix += 1
                gx, gy = fixed_goal(d)
                st = leg_nav2(b, gx, gy, 60.0)
                if st != 'SUCCEEDED':
                    break
            ok = st == 'SUCCEEDED'
        mp = b.map_pose()
        wgt = to_frame(base_gt, d)          # 該途經點的期望真值座標
        err = math.hypot(b.gt[0] - wgt[0], b.gt[1] - wgt[1])
        legs.append(dict(wp=i, goal=[gx, gy], status=st, gt=[b.gt[0], b.gt[1]], map=[mp[0], mp[1]],
                         want_gt=list(wgt), err=err, yaw_gt=b.gt[2], yaw_map=mp[2]))
        print(f'   [diag] 結束時真值 ({b.gt[0]:.2f},{b.gt[1]:.2f}) 期望 ({wgt[0]:.2f},{wgt[1]:.2f}) 誤差 {err:.2f} m；'
              f'yaw 真值 {math.degrees(b.gt[2]):.0f}° map {math.degrees(mp[2]):.0f}°', flush=True)
        print(f'[bridge] 第 {i + 1}/{len(wps)} 段 -> ({gx:.2f},{gy:.2f}) {st}', flush=True)
        if not ok and a.stop_on_fail:
            status, failed_wp = st, i      # 停下來：往下送只會一路連鎖失敗
            break
        if not ok and not last:
            continue                       # 途經點沒到不致命，往下一個走
        if not ok:
            status = st
    if any(l['status'] in ('FLIPPED',) for l in legs):
        status = 'FLIPPED'
    out = dict(status=status, mode=mode, flipped=b.flipped, duration=b.sim_now() - t0s, trace=b.trace,
               base_gt=base_gt, base_map=base_map, legs=legs, failed_wp=failed_wp, final_gt=b.gt, n_retry=n_retry, n_fix=n_fix, tf_log=b.tf_log)
    json.dump(out, open(a.out, 'w'))
    print('RESULT:' + json.dumps(dict(status=status, mode=mode, flipped=b.flipped, duration=round(out['duration'], 1))))


if __name__ == '__main__':
    main()
