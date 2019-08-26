#!/bin/bash
python2 Plotting/fit_tquark.py $1 $2 anything
python2 Plotting/histograms_tquark.py $1
python2 Plotting/plot_observables.py $1 $3 filler
