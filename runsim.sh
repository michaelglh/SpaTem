#!/bin/bash
W=20
K=5
L=5
T=50

epoch=1000
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
