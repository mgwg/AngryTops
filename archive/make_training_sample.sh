#!/bin/bash

nevts=3000000
mc_file=csv/topreco.mc.410501.nominal.csv
target=csv/training.csv

wc -l $mc_file

cat $mc_file | shuf | head -n $nevts > $target
wc -l $target
