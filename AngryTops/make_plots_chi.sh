#!/bin/bash
python2 Plotting/fit.py $1 pxpypzEM filler
python2 Plotting/histograms.py $1
python2 Plotting/plot_observables.py $1 None
