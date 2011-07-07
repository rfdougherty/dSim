#!/bin/bash

BIN=./bin/release/diffusion
GPU="gpu"
SPACE=50
NUMP=50000
FR=0.5
FS=0.5
CROSS=0

CFG=sim.cfg
OUT="data/sizeSpaceSimNew.m"

echo "OUT = $OUT"
   
for FS in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.5 3.0 3.5 4.0 4.5 5.0
do
  for FR in 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.8 5.0 
  do
    echo "*** RUNNING: $BIN -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS ***"
    $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -radiusMean=$FR -spaceMean=$FS -crossFraction=$CROSS
  done
done               

