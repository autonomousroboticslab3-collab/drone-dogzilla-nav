#!/bin/zsh
# 逐一短目標：每次重啟 Gazebo，確保機器狗從乾淨狀態出發
cd /home/t1204/thesis/nav
OUT=runs/shortB_batch.txt; : > $OUT
for c in 1.0,2.5 0.0,4.5 -1.5,1.5 0.5,6.5; do
  (cd ~/Downloads/DOGZILLA/sim_env && ./run.sh stop >/dev/null 2>&1; (./run.sh sim dwb world:=/worlds/site3.world >/dev/null 2>&1 &))
  sleep 20
  until docker logs dogzilla-sim 2>&1 | grep -qi "Managed nodes are active\|Nav2 active"; do sleep 5; done; sleep 15
  tag=shortB_${c/,/_}
  .venv/bin/python pipeline.py --scene site:3 --seed 3 --click=$c --replan 2 --drone-fix --timeout 400 --tag $tag > runs/$tag.log 2>&1
  echo "== click $c" >> $OUT; grep -E "各輪|summary" runs/$tag.log >> $OUT
done
echo DONE >> $OUT
