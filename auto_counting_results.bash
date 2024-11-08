#!/bin/bash

# prefix="results/decay_002_window_8_"
# decay=0.002
# window=8

# dwarg="--decay=$decay --min-window-size=$window"

# python get-count-accs.py --base-count=3 --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' $dwarg &
# python get-count-accs.py --base-count=10 --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue1.json" --ov-results-path="${prefix}accs_ue1.json" --sets 'setenv-935s_0' 'setenv-935s_60' 'setenv-935u_0' 'setenv-935u_60' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue2.json" --ov-results-path="${prefix}accs_ue2.json" --sets 'setenv-5158s_0' 'setenv-5158s_270' 'setenv-5158u_0' 'setenv-5158u_270' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue3.json" --ov-results-path="${prefix}accs_ue3.json" --sets 'setenv-457_0' 'setenv-457_seentar' 'setenv-457_unseentar' $dwarg &

prefix="results/decay_01_window_8_"
decay=0.01
window=8

dwarg="--decay=$decay --min-window-size=$window"

# python get-count-accs.py --base-count=3 --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' $dwarg &
# python get-count-accs.py --base-count=10 --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue1.json" --ov-results-path="${prefix}accs_ue1.json" --sets 'setenv-935s_0' 'setenv-935s_60' 'setenv-935u_0' 'setenv-935u_60' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue2.json" --ov-results-path="${prefix}accs_ue2.json" --sets 'setenv-5158s_0' 'setenv-5158s_270' 'setenv-5158u_0' 'setenv-5158u_270' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue3.json" --ov-results-path="${prefix}accs_ue3.json" --sets 'setenv-457_0' 'setenv-457_seentar' 'setenv-457_unseentar' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ut.json" --ov-results-path="${prefix}accs_ut.json" --sets 'setenv-457_unseentar' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_nlos.json" --ov-results-path="${prefix}accs_nlos.json" --sets 'set_10' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_se3.json" --ov-results-path="${prefix}accs_se3.json" --sets 'setenv-5158-2_n0' &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_se3_0.json" --ov-results-path="${prefix}accs_se3.json" --sets 'setenv-5158-2_0' &
# python get-count-accs.py --base-count=6 --br-results-path="${prefix}counts_ue4.json" --ov-results-path="${prefix}accs_ue4.json" --sets 'setenv-TSRB-conf' $dwarg &
# python get-count-accs.py --base-count=8 --br-results-path="${prefix}counts_ue5_ne.json" --ov-results-path="${prefix}accs_ue5_ne.json" --sets 'setenv-131_0' 'setenv-131_n0' $dwarg &
python get-count-accs.py --base-count=8 --br-results-path="${prefix}counts_ue5_fa.json" --ov-results-path="${prefix}accs_ue5_fa.json" --sets 'setenv-131_far' $dwarg &

# prefix="results/decay_02_window_8_"
# decay=0.02
# window=8

# dwarg="--decay=$decay --min-window-size=$window"

# python get-count-accs.py --base-count=3 --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' $dwarg &
# python get-count-accs.py --base-count=10 --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue1.json" --ov-results-path="${prefix}accs_ue1.json" --sets 'setenv-935s_0' 'setenv-935s_60' 'setenv-935u_0' 'setenv-935u_60' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue2.json" --ov-results-path="${prefix}accs_ue2.json" --sets 'setenv-5158s_0' 'setenv-5158s_270' 'setenv-5158u_0' 'setenv-5158u_270' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue3.json" --ov-results-path="${prefix}accs_ue3.json" --sets 'setenv-457_0' 'setenv-457_seentar' 'setenv-457_unseentar' $dwarg &

# prefix="results/decay_05_window_8_"
# decay=0.05
# window=8

# dwarg="--decay=$decay --min-window-size=$window"

# python get-count-accs.py --base-count=3 --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' $dwarg &
# python get-count-accs.py --base-count=10 --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue1.json" --ov-results-path="${prefix}accs_ue1.json" --sets 'setenv-935s_0' 'setenv-935s_60' 'setenv-935u_0' 'setenv-935u_60' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue2.json" --ov-results-path="${prefix}accs_ue2.json" --sets 'setenv-5158s_0' 'setenv-5158s_270' 'setenv-5158u_0' 'setenv-5158u_270' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}counts_ue3.json" --ov-results-path="${prefix}accs_ue3.json" --sets 'setenv-457_0' 'setenv-457_seentar' 'setenv-457_unseentar' $dwarg &

# prefix="results/view_"

# dwarg="--decay=0.01 --min-window-size=8"

# python get-count-accs.py --base-count=5 --br-results-path="${prefix}0_counts_5.json" --ov-results-path="${prefix}0_accs_5.json" --sets 'set9_0' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}60_counts_5.json" --ov-results-path="${prefix}60_accs_5.json" --sets 'set9_60' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}120_counts_5.json" --ov-results-path="${prefix}120_accs_5.json" --sets 'set9_120' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}180_counts_5.json" --ov-results-path="${prefix}180_accs_5.json" --sets 'set9_180' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}240_counts_5.json" --ov-results-path="${prefix}240_accs_5.json" --sets 'set9_240' $dwarg &
# python get-count-accs.py --base-count=5 --br-results-path="${prefix}300_counts_5.json" --ov-results-path="${prefix}300_accs_5.json" --sets 'set9_300' $dwarg &

wait
