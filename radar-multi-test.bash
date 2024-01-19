#!/bin/bash

constants="--train-data data/radar/train_3seen"
ckpt="--load ckpts/sw-3seen-new/ckpt_39.pth"
nepochs="--milestones 40 10 20 30"
mode="--mode 0"
outprefix="3s_sw"

# # Whole environments
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_2seen --output $HOME/p-ks207-0/${outprefix}_457.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen3 --output $HOME/p-ks207-0/${outprefix}_5158.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1 --output $HOME/p-ks207-0/${outprefix}_935.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3 --output $HOME/p-ks207-0/${outprefix}_131o.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue4 --output $HOME/p-ks207-0/${outprefix}_131i.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue --output $HOME/p-ks207-0/${outprefix}_ue.txt

# # Front-nonfront split (Front)
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_0 --output $HOME/p-ks207-0/${outprefix}_5152_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen2_0 --output $HOME/p-ks207-0/${outprefix}_457_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_s3_0 --output $HOME/p-ks207-0/${outprefix}_5158_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1_0 --output $HOME/p-ks207-0/${outprefix}_935_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3_0 --output $HOME/p-ks207-0/${outprefix}_131o_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue_0 --output $HOME/p-ks207-0/${outprefix}_ue_0.txt

# # Front-nonfront split (Not Front)
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_old --output $HOME/p-ks207-0/${outprefix}_5152_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen2_n0 --output $HOME/p-ks207-0/${outprefix}_457_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_s3_n0 --output $HOME/p-ks207-0/${outprefix}_5158_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1_n0 --output $HOME/p-ks207-0/${outprefix}_935_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3_n0 --output $HOME/p-ks207-0/${outprefix}_131o_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue_n0 --output $HOME/p-ks207-0/${outprefix}_ue_n0.txt

# # Split view
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_60 --output $HOME/p-ks207-0/${outprefix}_5152_60.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_120 --output $HOME/p-ks207-0/${outprefix}_5152_120.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_180 --output $HOME/p-ks207-0/${outprefix}_5152_180.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_240 --output $HOME/p-ks207-0/${outprefix}_5152_240.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_300 --output $HOME/p-ks207-0/${outprefix}_5152_300.txt

# # Unseen targets (Whole, Front, and Not Front)
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut --output $HOME/p-ks207-0/${outprefix}_ut.txt
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut_0 --output $HOME/p-ks207-0/${outprefix}_ut_0.txt
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut_n0 --output $HOME/p-ks207-0/${outprefix}_ut_n0.txt

# NLOS
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_NLOS --output $HOME/p-ks207-0/${outprefix}_nlos.txt


constants="--train-data data/radar/train_3seen"
ckpt="--load ckpts/sw-3seen-new/ckpt_39.pth"
nepochs="--milestones 40 10 20 30"
mode="--mode 6"
outprefix="3s_sg"

# # Whole environments
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_2seen --output $HOME/p-ks207-0/${outprefix}_457.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen3 --output $HOME/p-ks207-0/${outprefix}_5158.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1 --output $HOME/p-ks207-0/${outprefix}_935.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3 --output $HOME/p-ks207-0/${outprefix}_131o.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue4 --output $HOME/p-ks207-0/${outprefix}_131i.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue --output $HOME/p-ks207-0/${outprefix}_ue.txt

# # Front-nonfront split (Front)
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_0 --output $HOME/p-ks207-0/${outprefix}_5152_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen2_0 --output $HOME/p-ks207-0/${outprefix}_457_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_s3_0 --output $HOME/p-ks207-0/${outprefix}_5158_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1_0 --output $HOME/p-ks207-0/${outprefix}_935_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3_0 --output $HOME/p-ks207-0/${outprefix}_131o_0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue_0 --output $HOME/p-ks207-0/${outprefix}_ue_0.txt

# # Front-nonfront split (Not Front)
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_old --output $HOME/p-ks207-0/${outprefix}_5152_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_seen2_n0 --output $HOME/p-ks207-0/${outprefix}_457_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_s3_n0 --output $HOME/p-ks207-0/${outprefix}_5158_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue1_n0 --output $HOME/p-ks207-0/${outprefix}_935_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue3_n0 --output $HOME/p-ks207-0/${outprefix}_131o_n0.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ue_n0 --output $HOME/p-ks207-0/${outprefix}_ue_n0.txt

# # Split view
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_60 --output $HOME/p-ks207-0/${outprefix}_5152_60.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_120 --output $HOME/p-ks207-0/${outprefix}_5152_120.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_180 --output $HOME/p-ks207-0/${outprefix}_5152_180.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_240 --output $HOME/p-ks207-0/${outprefix}_5152_240.txt
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/rtest_300 --output $HOME/p-ks207-0/${outprefix}_5152_300.txt

# # Unseen targets (Whole, Front, and Not Front)
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut --output $HOME/p-ks207-0/${outprefix}_ut.txt
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut_0 --output $HOME/p-ks207-0/${outprefix}_ut_0.txt
sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_ut_n0 --output $HOME/p-ks207-0/${outprefix}_ut_n0.txt

# NLOS
# sbatch run_radar_model.sbatch $constants $mode $ckpt $nepochs --test-data data/radar/test_NLOS --output $HOME/p-ks207-0/${outprefix}_nlos.txt


constants="--train-data data/radar/train_3seen"
ckpt="--load ckpts/p43r-3seen/ckpt_49.pth"
nepochs="--milestones 50 10 20 30"
outprefix="3s_p43r"

# Whole environments
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_2seen --output $HOME/p-ks207-0/${outprefix}_457.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen3 --output $HOME/p-ks207-0/${outprefix}_5158.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1 --output $HOME/p-ks207-0/${outprefix}_935.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3 --output $HOME/p-ks207-0/${outprefix}_131o.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue4 --output $HOME/p-ks207-0/${outprefix}_131i.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue --output $HOME/p-ks207-0/${outprefix}_ue.txt

# Front-nonfront split (Front)
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_0 --output $HOME/p-ks207-0/${outprefix}_5152_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen2_0 --output $HOME/p-ks207-0/${outprefix}_457_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_s3_0 --output $HOME/p-ks207-0/${outprefix}_5158_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1_0 --output $HOME/p-ks207-0/${outprefix}_935_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3_0 --output $HOME/p-ks207-0/${outprefix}_131o_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue_0 --output $HOME/p-ks207-0/${outprefix}_ue_0.txt

# Front-nonfront split (Not Front)
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_old --output $HOME/p-ks207-0/${outprefix}_5152_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen2_n0 --output $HOME/p-ks207-0/${outprefix}_457_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_s3_n0 --output $HOME/p-ks207-0/${outprefix}_5158_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1_n0 --output $HOME/p-ks207-0/${outprefix}_935_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3_n0 --output $HOME/p-ks207-0/${outprefix}_131o_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue_n0 --output $HOME/p-ks207-0/${outprefix}_ue_n0.txt

# Split view
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_60 --output $HOME/p-ks207-0/${outprefix}_5152_60.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_120 --output $HOME/p-ks207-0/${outprefix}_5152_120.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_180 --output $HOME/p-ks207-0/${outprefix}_5152_180.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_240 --output $HOME/p-ks207-0/${outprefix}_5152_240.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_300 --output $HOME/p-ks207-0/${outprefix}_5152_300.txt

# Unseen targets (Whole, Front, and Not Front)
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut --output $HOME/p-ks207-0/${outprefix}_ut.txt
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut_0 --output $HOME/p-ks207-0/${outprefix}_ut_0.txt
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut_n0 --output $HOME/p-ks207-0/${outprefix}_ut_n0.txt

# NLOS
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_NLOS --output $HOME/p-ks207-0/${outprefix}_nlos.txt


constants="--train-data data/radar/train_3seen"
ckpt="--load ckpts/p41r-3seen/ckpt_49.pth"
nepochs="--milestones 50 10 20 30"
outprefix="3s_p41r"

# Whole environments
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_2seen --output $HOME/p-ks207-0/${outprefix}_457.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen3 --output $HOME/p-ks207-0/${outprefix}_5158.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1 --output $HOME/p-ks207-0/${outprefix}_935.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3 --output $HOME/p-ks207-0/${outprefix}_131o.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue4 --output $HOME/p-ks207-0/${outprefix}_131i.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue --output $HOME/p-ks207-0/${outprefix}_ue.txt

# Front-nonfront split (Front)
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_0 --output $HOME/p-ks207-0/${outprefix}_5152_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen2_0 --output $HOME/p-ks207-0/${outprefix}_457_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_s3_0 --output $HOME/p-ks207-0/${outprefix}_5158_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1_0 --output $HOME/p-ks207-0/${outprefix}_935_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3_0 --output $HOME/p-ks207-0/${outprefix}_131o_0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue_0 --output $HOME/p-ks207-0/${outprefix}_ue_0.txt

# Front-nonfront split (Not Front)
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_old --output $HOME/p-ks207-0/${outprefix}_5152_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_seen2_n0 --output $HOME/p-ks207-0/${outprefix}_457_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_s3_n0 --output $HOME/p-ks207-0/${outprefix}_5158_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue1_n0 --output $HOME/p-ks207-0/${outprefix}_935_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue3_n0 --output $HOME/p-ks207-0/${outprefix}_131o_n0.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ue_n0 --output $HOME/p-ks207-0/${outprefix}_ue_n0.txt

# Split view
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_60 --output $HOME/p-ks207-0/${outprefix}_5152_60.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_120 --output $HOME/p-ks207-0/${outprefix}_5152_120.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_180 --output $HOME/p-ks207-0/${outprefix}_5152_180.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_240 --output $HOME/p-ks207-0/${outprefix}_5152_240.txt
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/rtest_300 --output $HOME/p-ks207-0/${outprefix}_5152_300.txt

# Unseen targets (Whole, Front, and Not Front)
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut --output $HOME/p-ks207-0/${outprefix}_ut.txt
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut_0 --output $HOME/p-ks207-0/${outprefix}_ut_0.txt
sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_ut_n0 --output $HOME/p-ks207-0/${outprefix}_ut_n0.txt

# NLOS
# sbatch run_p4_model.sbatch $constants $ckpt $nepochs --test-data data/radar/test_NLOS --output $HOME/p-ks207-0/${outprefix}_nlos.txt