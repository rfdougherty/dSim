#!/bin/bash
#
# Reproduce the corpus callosum density simulations from
# Stikov et al., NeuroImage 2011 (https://pmc.ncbi.nlm.nih.gov/articles/PMC2997845/).
#
# Runs every pre-generated callosal fiber set in fibers/ (5 segments x a range
# of densities) with the paper's settings (sim_paper.cfg / sim_paper.grads).
# Results are appended, one run per block, to $OUT as a MATLAB script; load
# them with analysis/analyzeDensity.m.
#
# Usage: ./runSimDensity.sh [gpu|cpu] [output.m]

BIN=./bin/release/diffusion
GPU="${1:-gpu}"
OUT="${2:-data/densityTest_paper.m}"
CFG=sim_paper.cfg
SPACE=100       # voxel half-width in um (200 um voxel)
NUMSPINS=100000

echo "OUT = $OUT"

for FF in fibers/*.fibers
do
  echo "*** RUNNING $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numSpins=$NUMSPINS -fibersFile=$FF ***"
  $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numSpins=$NUMSPINS -fibersFile=$FF
done
