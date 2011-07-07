#!/bin/bash

BIN=./bin/release/diffusion
GPU="gpu"
SPACE=100
NUMP=100000

if [ "$GPU" == "gpu" ]; then
   GPU_STR=`/sbin/lspci | grep VGA`
   PROC=`expr "$GPU_STR" : '.* \(G[0-9][0-9]\) '`
else
   PROC="core2_2660"
fi

CFG=sim.cfg
OUT="data/densityTest_new.m"
echo "OUT = $OUT"
   
for FF in `ls fibers_new/*`
  do
  echo "*** RUNNING $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -fibersFile=$FF ***"
  $BIN -nodisp -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP -fibersFile=$FF
done

