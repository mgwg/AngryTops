#!/bin/bash
python2 ChiSquared/ChiSquaredAlgo.py
python2 Plotting/fit.py ../CheckPoints/Chi2Model pxpypzEM filler
python2 Plotting/histograms.py ../CheckPoints/Chi2Model
python2 Plotting/plot_observables.py ../CheckPoints/Chi2Model None
