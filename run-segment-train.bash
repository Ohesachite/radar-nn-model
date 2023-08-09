#!/bin/bash

constants="--train-data data/radar/train_sd_big --test-data data/radar/test_sd_big -t"
nepochs="--milestones 200 50 100 150"
sbatch run_radar_model.sbatch $constants --mode 3 --store ckpts/sg-nodeci --output $HOME/p-ks207-0/mode3_big.txt
sbatch run_radar_model.sbatch $constants --mode 5 --store ckpts/sg-deci --output $HOME/p-ks207-0/mode5_big.txt
sbatch run_radar_model.sbatch $constants --mode 6 --store ckpts/sg-sw --output $HOME/p-ks207-0/mode6_big.txt