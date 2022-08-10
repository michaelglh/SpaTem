#!/bin/bash
W=100
K=20
L=10
T=100

epoch=8
COUNTER=1
THREAD=8

dpath="./data"
fpath="./fig"
rm -r $dpath
rm -r $fpath

sim=1

if [ $sim -gt 0 ]
then
    for trial in `seq 1 1 ${epoch}`
    do
        echo $COUNTER
        if (( "${COUNTER}" == "${THREAD}" ))
        then
            python randnet.py --W $W --K $K --L $L --T $T --trial $trial
            COUNTER=1
        else
            python randnet.py --W $W --K $K --L $L --T $T --trial $trial &
            COUNTER=$(( COUNTER + 1 ))
        fi
    done
fi
wait
