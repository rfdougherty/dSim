#!/bin/bash

BIN=./bin/release/diffusion
GPU="gpu"
SPACE=100
NUMP=50000
FR=0.5
FS=0.5

CFG=sim.cfg
OUT="data/crossSim.m"

echo "OUT = $OUT"
   
for FS in 0.5 0.4 0.3 0.2 0.1
do
  for CROSS in 0.0 0.1 0.2 0.3 0.4 0.5
  do
    echo "*** RUNNING: $BIN -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS ***"
    $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS
  done
done               

