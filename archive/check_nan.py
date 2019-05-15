#!/usr/bin/env python

import os, sys
import numpy as np
np.set_printoptions( precision=2, suppress=True, linewidth=200 )

import pandas as pd

from features import *

############

training_filename = "csv/training.csv"

if len(sys.argv) > 1: training_filename  = sys.argv[1]

# read in input file
data = pd.read_csv( training_filename, delimiter=',', names=header )
print "INFO: dataset loaded into memory"
print "INFO: header:"
print header

print "INFO: input events:", len(data)

nan_rows = data[data.isnull().T.any().T]

print "INFO: invalid rows:"
print nan_rows
