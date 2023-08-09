#!/bin/bash

prefix="results/decay_002_"

python get-count-accs.py --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' --decay=0.002

python get-count-accs.py --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' --decay=0.002

python get-count-accs.py --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' --decay=0.002

prefix="results/decay_02_"

python get-count-accs.py --br-results-path="${prefix}counts_3.json" --ov-results-path="${prefix}accs_3.json" --sets 'set8_0' 'set8_60' 'set8_120' 'set8_180' 'set8_240' 'set8_300' --decay=0.02

python get-count-accs.py --br-results-path="${prefix}counts_5.json" --ov-results-path="${prefix}accs_5.json" --sets 'set9_0' 'set9_60' 'set9_120' 'set9_180' 'set9_240' 'set9_300' --decay=0.02

python get-count-accs.py --br-results-path="${prefix}counts_10.json" --ov-results-path="${prefix}accs_10.json" --sets 'set11' 'set11_180' --decay=0.02
