#!/bin/bash

constants="-t --train-data data/radar/train_3seen --test-data data/radar/test_seen2_n0"
nepochs="--milestones 40 10 20 30"
outprefix="tune_"

cac=0

for ca in 0.0 0.1 0.5 0.9 1.0
do
    cac=$((cac+1))
    cwc=0
    for cw in 0.01 0.1 1.0
    do
        cwc=$((cwc+1))
        sbatch run_radar_model.sbatch $constants $nepochs -ca $ca -cw $cw -st ckpts/${outprefix}ca${cac}_cw${cwc}
    done
done