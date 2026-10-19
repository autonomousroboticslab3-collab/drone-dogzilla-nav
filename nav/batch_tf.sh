#!/bin/zsh
cd /home/t1204/thesis/nav
OUT=runs/tf_batch.txt; : > $OUT
for rep in 1 2 3 4 5 6; do
  (cd ~/Downloads/DOGZILLA/sim_env && ./run.sh stop >/dev/null 2>&1; (./run.sh sim dwb world:=/worlds/site3.world >/dev/null 2>&1 &))
  sleep 20
  until docker logs dogzilla-sim 2>&1 | grep -qi "Managed nodes are active\|Nav2 active"; do sleep 5; done; sleep 15
  tag=tf${rep}_-1.0_6.0
  .venv/bin/python pipeline.py --scene site:3 --seed 3 --click=-1.0,6.0 --replan 2 --drone-fix --timeout 400 --tag $tag > runs/$tag.log 2>&1
  echo "$tag | $(grep -E '各輪' runs/$tag.log)" >> $OUT
done
echo DONE >> $OUT
