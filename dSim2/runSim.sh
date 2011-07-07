#!/bin/bash

BIN=./bin/release/diffusion
GPU="cpu"

if [ "$GPU" == "gpu" ]; then
   GPU_STR=`/sbin/lspci | grep VGA`
   PROC=`expr "$GPU_STR" : '.* \(G[0-9][0-9]\) '`
else
   PROC="core2_2660"
fi

CFG=sim_gpuTest.cfg
OUT="data/gpuTest_${PROC}.m"
echo "OUT = $OUT"
   
for SPACE in 10 18 32 56 100
do
  for NUMP in 100 316 1000 3162 10000 31623 100000 316228 1000000
  do
      echo "*** RUNNING: $BIN -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP ***"
      $BIN -config=$CFG -out=$OUT -$GPU -spaceScale=$SPACE -numParticles=$NUMP
  done
done               

