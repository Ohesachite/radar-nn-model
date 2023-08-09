#!/bin/bash

for lr in 0.005 0.01 0.05
do
    for momentum in 0.5 0.9 0.95
    do
        for nintlayers in 0 1 2
        do
            file_name="$HOME/p-ks207-0/results-lr-$lr-momentum-$clip_len-nil-$nintlayers.txt"
            if [ ! -f ${file_name} ]; then
                sbatch run_output_combinator.sbatch -m ckpts/sw/ckpt_39.pth ckpts/sg-deci/ckpt_199.pth -h $momentum $nintlayers $lr -trd data/radar/train_sd_big -ted data/radar/test_sd_big -o $file_name
            fi
        done
    done
done