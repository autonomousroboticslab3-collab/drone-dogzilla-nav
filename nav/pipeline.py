#!/usr/bin/env python3
"""整條架構的端到端模擬：

  空拍機模擬（建圖 / 偵測目標 / 使用者點選 / A* 預覽）
        │  目標 = 相對於機器人的偏移量（空拍機地圖上標的機器人位置與朝向）
        ▼
  橋接（容器內 ROS 2）→ Nav2 或原廠 RL 控制器 → Gazebo 裡的 DOGZILLA 實際行走
        │  軌跡
        ▼
  疊回空拍機地圖，用真值評分（是否抵達 / 碰撞 / 翻倒 / 用時）

前提：原廠模擬已啟動，且載入的世界與 --scene 相同，例如
  cd ~/Downloads/DOGZILLA/sim_env && ./run.sh sim rl world:=/worlds/site3.world
用法：
  .venv/bin/python pipeline.py --scene site:3 --seed 3                 # 使用者點選自動偵測到的目標
  .venv/bin/python pipeline.py --scene site:3 --seed 3 --click 6,-3    # 使用者點選此世界座標
  .venv/bin/python pipeline.py ... --waypoints                         # 沿著 A* 路徑逐段送途經點
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
CONTAINER = 'dogzilla-sim'
ROS = 'source /opt/ros/humble/setup.bash'


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def bridge(args, timeout):
    r = sh(['docker', 'exec', CONTAINER, 'bash', '-lc', f'{ROS}; python3 /tmp/gz_bridge.py {args}'], timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith('[bridge]') or any(k in line for k in ('[retry]', '[校正]', '[diag]')):
            print(line, flush=True)
    res = [l for l in r.stdout.splitlines() if l.startswith('RESULT:')]
    if not res:
        sys.exit(f'橋接失敗：\n{r.stdout[-800:]}\n{r.stderr[-800:]}')
    return json.loads(res[-1][7:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=1.0)
    ap.add_argument('--click')
    ap.add_argument('--no-dust', action='store_true')
    ap.add_argument('--waypoints', action='store_true')
    ap.add_argument('--replan', type=int, default=0, help='某段失敗時最多重新規劃幾次（0 = 不重規劃）')
    ap.add_argument('--block-radius', type=float, default=0.6)
    ap.add_argument('--drone-fix', action='store_true', help='每段用空拍機觀測到的機器人位置修正 SLAM 漂移')
    ap.add_argument('--near', type=float, default=1.5, help='失敗的途經點離機器人小於此距離 (m)：視為被困，先脫困、不封鎖')
    ap.add_argument('--max-escapes', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=900.0, help='Gazebo 裡每一輪的逾時（秒）')
    ap.add_argument('--tag', default='run')
    a = ap.parse_args()
    if a.replan:
        a.waypoints = True                      # 要知道是哪一段失敗，必須逐段送
    out = HERE / 'runs'
    out.mkdir(exist_ok=True)
    png, merged_f = out / f'{a.tag}.png', out / f'{a.tag}_result.json'

    if CONTAINER not in sh(['docker', 'ps', '--format', '{{.Names}}']).stdout.split():
        sys.exit('原廠模擬沒有在跑。先執行：cd ~/Downloads/DOGZILLA/sim_env && ./run.sh sim dwb world:=/worlds/<世界>.world')
    sh(['docker', 'cp', str(HERE / 'bridge' / 'gz_bridge.py'), f'{CONTAINER}:/tmp/gz_bridge.py'])

    blocks, trace, dur, flipped, status, history, n_esc = [], [], 0.0, False, 'NOT_RUN', [], 0
    for it in range(a.replan + 1):
        tag = f'{a.tag}_it{it}'
        plan_f, res_f = out / f'{tag}_plan.json', out / f'{tag}_result.json'
        plan_f.unlink(missing_ok=True)
        print(f'\n######## 第 {it + 1}/{a.replan + 1} 輪' + (f'（已封鎖 {len(blocks)} 處）' if blocks else ''))
        print('== 1/3 讀取機器人目前位姿（模擬「空拍機看到機器人，使用者在地圖上標出位置與朝向」）')
        pose = bridge('pose', 90)
        gx, gy, gyaw = pose['gt']
        print(f'   機器人世界位姿 ({gx:.2f}, {gy:.2f}, {math.degrees(gyaw):.0f}°)，控制模式 = {pose["mode"]}')

        common = ['--scene', a.scene, '--seed', str(a.seed), '--noise', str(a.noise), f'--start={gx},{gy},{gyaw}']
        if a.click:
            common += [f'--click={a.click}']
        if a.no_dust:
            common += ['--no-dust']
        if blocks:
            common += [f'--block={";".join(f"{x},{y}" for x, y in blocks)}', f'--block-radius={a.block_radius}']

        print('== 2/3 空拍機端：規劃路徑')
        r = sh([PY, str(HERE / 'sim.py'), *common, '--export', str(plan_f), '--out', str(out / f'{tag}_plan.png')])
        print('\n'.join('   ' + l for l in r.stdout.splitlines() if l.startswith(('[map] 已封鎖', '[goal', '[export]'))))
        if not plan_f.exists():
            status = 'NO_PLAN'
            print('   規劃失敗：地圖上已經沒有可行路徑（可能被封鎖點堵死，或目標在未知區域）')
            break

        print('== 3/3 送進 Gazebo' + ('（逐段途經點，失敗即停）' if a.waypoints else ''))
        plan = json.load(open(plan_f))
        plan['base_gt'], plan['base_map'] = pose['gt'], pose['map']
        json.dump(plan, open(plan_f, 'w'), indent=1)
        sh(['docker', 'cp', str(plan_f), f'{CONTAINER}:/tmp/gz_plan.json'])
        res = bridge(f'go --plan /tmp/gz_plan.json --out /tmp/gz_result.json --timeout {a.timeout}'
                     + (' --waypoints' if a.waypoints else '') + (' --stop-on-fail' if a.replan else '') + (' --drone-fix' if a.drone_fix else ''), a.timeout + 120)
        sh(['docker', 'cp', f'{CONTAINER}:/tmp/gz_result.json', str(res_f)])
        full = json.load(open(res_f))
        trace += full['trace']; dur += full['duration']; flipped |= full['flipped']; status = full['status']
        history.append(dict(round=it + 1, status=full['status'], failed_wp=full.get('failed_wp'), n_wp=len(plan['waypoints_body']),
                            duration=round(full['duration'], 1)))
        print(f'   本輪結果：{full["status"]}，模擬時間 {full["duration"]:.0f} s，翻倒 = {full["flipped"]}')
        if status in ('SUCCEEDED', 'REACHED') or flipped:
            break
        if full.get('failed_wp') is None:
            break
        if it < a.replan:
            fi = full['failed_wp']
            dist = math.hypot(*plan['waypoints_body'][fi])
            if dist < a.near:
                # 失敗點就在機器人身邊：不是「那條路不通」，而是機器人被困住了。封鎖只會把它周圍越封越死。
                # 改讓機器人用自己的 LiDAR costmap 找附近開闊處自行脫困，再從新位置重規劃。
                if n_esc >= a.max_escapes:
                    print(f'   已脫困 {n_esc} 次仍被困，放棄'); status = 'STUCK'; break
                n_esc += 1
                print(f'   第 {fi + 1} 個途經點只在 {dist:.1f} m 外 -> 判定被困，不封鎖；機器人脫困（第 {n_esc} 次）')
                er = bridge('escape --out /tmp/gz_escape.json', 400)
                sh(['docker', 'cp', f'{CONTAINER}:/tmp/gz_escape.json', str(out / f'{tag}_escape.json')])
                ef = json.load(open(out / f'{tag}_escape.json'))
                trace += ef.get('trace', []); dur += ef.get('duration', 0.0); flipped |= ef.get('flipped', False)
                history.append(dict(round=f'{it + 1}-脫困', status=ef['status'], failed_wp=None, n_wp=0, duration=round(ef.get('duration', 0), 1)))
                print(f'   脫困結果：{ef["status"]}（移動 {ef.get("moved", 0):.2f} m，周圍開闊度 {ef.get("openness", 0):.2f}）')
                if ef['status'] != 'ESCAPED':
                    status = 'STUCK'; break
            else:
                fw = plan['waypoints_ground'][fi]
                blocks.append(tuple(fw))
                print(f'   第 {fi + 1} 個途經點（{dist:.1f} m 外）走不過去 -> 在空拍機地圖上封鎖 ({fw[0]:.2f}, {fw[1]:.2f})，重新規劃')

    merged = dict(status=status, flipped=flipped, duration=dur, trace=trace, rounds=history, blocks=blocks)
    json.dump(merged, open(merged_f, 'w'))
    print('\n== 疊回空拍機地圖並用真值評分')
    if trace:
        r = sh([PY, str(HERE / 'sim.py'), *common, '--overlay', str(merged_f), '--out', str(png)])
        print('\n'.join('   ' + l for l in r.stdout.splitlines() if l.startswith(('[goal', '[summary]'))))
        print(f'圖：{png}')
    print('各輪：', ' -> '.join(f'[{h["round"]}]{h["status"]}' + (f'（第{h["failed_wp"] + 1}/{h["n_wp"]}段失敗）' if h['failed_wp'] is not None else '') for h in history))


if __name__ == '__main__':
    main()
