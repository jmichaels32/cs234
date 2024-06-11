#!/bin/bash

set -ex
# Run the code
flags="--geom-friction --geom-margin --body-mass --body-gravcomp"
seeds="0 1"

for flag in $flags; do
    for seed in $seeds; do
        python3 test.py --seed $seed $flag=True
    done
done