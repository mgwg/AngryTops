#!/usr/bin/env bash
source /home/fsyed/.bashrc
source /home/fsyed/tf/bin/activate

python train_simple_model.py $1 $2 $3 $4 $5

/usr/bin/python fit.py CheckPoints/$2
/usr/bin/python histograms.py CheckPoints/$2
/usr/bin/python MakeContourPlots.py CheckPoints/$2
/usr/bin/python plot_observables.py CheckPoints/$2
