#!/bin/bash
lr=0.01

for clip_len in 16
do
    for ca in 0.1 0.3 0.5 0.7 0.9
    do
        for cw in 0.1
        do
            file_name="results/results-lr-$lr-clip-len-$clip_len-ca-$ca-cw-$cw.txt"
            if [ ! -f ${file_name} ]; then
                python train-radar.py --lr=$lr --clip-len=$clip_len --contrastive-alpha=$ca --contrastive-weight=$cw  --result-file=$file_name
            fi
        done
    done
done