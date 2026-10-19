#!/bin/zsh
# 6 個短目標 × 3 次重複；每次重啟 Gazebo，從乾淨狀態出發
cd /home/t1204/thesis/nav
OUT=runs/cnt_batch.txt; : > $OUT
for rep in 1 2 3; do
for c in -1.5,1.5 1.0,2.5 0.0,4.5 -1.0,6.0 0.5,6.5 3.5,6.5; do
  (cd ~/Downloads/DOGZILLA/sim_env && ./run.sh stop >/dev/null 2>&1; (./run.sh sim dwb world:=/worlds/site3.world >/dev/null 2>&1 &))
  sleep 20
  until docker logs dogzilla-sim 2>&1 | grep -qi "Managed nodes are active\|Nav2 active"; do sleep 5; done; sleep 15
  tag=cnt${rep}_${c/,/_}
  .venv/bin/python pipeline.py --scene site:3 --seed 3 --click=$c --replan 2 --drone-fix --timeout 400 --tag $tag > runs/$tag.log 2>&1
  echo "rep$rep click $c | $(grep -E '\[goal 0\].*(離真實|逾時|失敗)' runs/$tag.log | tail -1 | sed 's/.*\] //') | $(grep -E '各輪' runs/$tag.log) | $(.venv/bin/python -c "import json,glob;r=[json.load(open(f)) for f in sorted(glob.glob(\"runs/${tag}_it*_result.json\"))];print(\"retry=%d 校正=%d\"%(sum(x.get(\"n_retry\",0) for x in r),sum(x.get(\"n_fix\",0) for x in r)))")" >> $OUT
done; done
echo DONE >> $OUT
