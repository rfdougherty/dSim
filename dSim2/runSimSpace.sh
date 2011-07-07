#!/bin/bash

BIN=./bin/release/diffusion
GPU="gpu"
SPACE=50
NUMP=50000
FR=0.5
FS=0.5
CROSS=0

CFG=sim.cfg
OUT="data/spaceSim.m"

echo "OUT = $OUT"
   
for FS in 10.0 9.0 8.0 7.0 6.0 5.0 4.5 4.0 3.5 3.0 2.5 2.0 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.15 0.1 0.05 0.0
do
   echo "*** RUNNING: $BIN -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS ***"
   $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS
done               

