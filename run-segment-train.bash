#!/bin/bash

constants="--test-data data/radar/test_seen3 -t"
nepochs="--milestones 50 10 20 30"
sbatch run_p4_model.sbatch $constants --store ckpts/p43r-3seen --train-data data/radar/train_3seen
sbatch run_p4_model.sbatch $constants --store ckpts/p41r-3seen --train-data data/radar/train_3seen_1r