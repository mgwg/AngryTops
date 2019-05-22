#!/usr/bin/env python
import os, sys, time
import argparse
import tensorflow as tf
from tensorflow import keras
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
import pandas as pd
from FormatInputOutput import get_input_output

from features import *
import models
################################################################################
# CONSTANTS
GeV = 1e3
TeV = 1e6
m_t = 172.5
m_W = 80.4
m_b = 4.95
max_momentum = 1000.
np.set_printoptions( precision=3, suppress=True, linewidth=250 )
infilename = "csv/topreco_augmented1.csv"
model_filename  = "{}/simple_model.h5".format(training_dir)
if len(sys.argv) > 1: infilename = sys.argv[1]

################################################################################
# HELPER FUNCTIONS

def PrintOut( p4_true, p4_fitted, event_info, label ):
  print("rn=%-10i en=%-10i ) %s :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
               ( event_info[0], event_info[1], label,
                p4_true.Pt(),   p4_true.Rapidity(),   p4_true.Phi(),   p4_true.E(),   p4_true.M(), \
                p4_fitted.Pt(), p4_fitted.Rapidity(), p4_fitted.Phi(), p4_fitted.E(), p4_fitted.M() ))


def MakeP4( y, m=0., sf=1.0 ):
  p4 = TLorentzVector()

  px = y[0] * sf
  py = y[1] * sf
  pz = y[2] * sf
  P2 = px*px + py*py + pz*pz
  E  = TMath.Sqrt( P2 + m*m )

  p4.SetPxPyPzE( px, py, pz, E )
  return p4

################################################################################
# LOAD KERAS MODEL AND CSV FILES
X_scaler = None
y_scaler = None
dnn      = None

dnn = keras.models.load_model(model_filename)
print(dnn.summary())

# read in input file
data = pd.read_csv(infilename, names=column_names)

X_jets = data[input_features_jets].values
X_lept = data[input_features_lep].values

y_true_W_lep = data[target_features_W_lep].values
y_true_W_had = data[target_features_W_had].values
y_true_b_lep = data[target_features_b_lep].values
y_true_b_had = data[target_features_b_had].values
y_true_t_lep = data[target_features_t_lep].values
y_true_t_had = data[target_features_t_had].values

event_info = data[features_event_info].values
n_events   = len(event_info)
(training_input, training_output), (testing_input, testing_output) = get_input_output()

################################################################################
# TRAINING MODEL
print("INFO: fitting ttbar decay chain...")
y_fitted = dnn.predict(testing_input)
#y_fitted = y_scaler.inverse_transform( y_fitted )
print("INFO ...done")

################################################################################
# CREATE OUTPUT TREE/FILE
ofilename = "{}/fitted.root".format(training_dir)
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

# Create output tree
b_eventNumber = array( 'l', [ 0 ] )
b_runNumber   = array( 'i', [ 0 ] )
b_mcChannelNumber = array( 'i', [ 0 ] )
b_weight_mc   = array( 'f', [ 0.] )

b_W_had_px_true   = array( 'f', [ -1.] )
b_W_had_py_true   = array( 'f', [ -1.] )
b_W_had_pz_true   = array( 'f', [ -1.] )
b_W_had_E_true    = array( 'f', [ -1.] )
b_W_had_m_true    = array( 'f', [ -1.] )
b_W_had_pt_true   = array( 'f', [ -1.] )
b_W_had_y_true    = array( 'f', [ -1.] )
b_W_had_phi_true  = array( 'f', [ -1.] )
b_b_had_px_true   = array( 'f', [ -1.] )
b_b_had_py_true   = array( 'f', [ -1.] )
b_b_had_pz_true   = array( 'f', [ -1.] )
b_b_had_E_true    = array( 'f', [ -1.] )
b_b_had_m_true    = array( 'f', [ -1.] )
b_b_had_pt_true   = array( 'f', [ -1.] )
b_b_had_y_true    = array( 'f', [ -1.] )
b_b_had_phi_true  = array( 'f', [ -1.] )
b_t_had_px_true   = array( 'f', [ -1.] )
b_t_had_py_true   = array( 'f', [ -1.] )
b_t_had_pz_true   = array( 'f', [ -1.] )
b_t_had_E_true    = array( 'f', [ -1.] )
b_t_had_m_true    = array( 'f', [ -1.] )
b_t_had_pt_true   = array( 'f', [ -1.] )
b_t_had_y_true    = array( 'f', [ -1.] )
b_t_had_phi_true  = array( 'f', [ -1.] )
b_W_lep_px_true   = array( 'f', [ -1.] )
b_W_lep_py_true   = array( 'f', [ -1.] )
b_W_lep_pz_true   = array( 'f', [ -1.] )
b_W_lep_E_true    = array( 'f', [ -1.] )
b_W_lep_m_true    = array( 'f', [ -1.] )
b_W_lep_pt_true   = array( 'f', [ -1.] )
b_W_lep_y_true    = array( 'f', [ -1.] )
b_W_lep_phi_true  = array( 'f', [ -1.] )
b_b_lep_px_true   = array( 'f', [ -1.] )
b_b_lep_py_true   = array( 'f', [ -1.] )
b_b_lep_pz_true   = array( 'f', [ -1.] )
b_b_lep_E_true    = array( 'f', [ -1.] )
b_b_lep_m_true    = array( 'f', [ -1.] )
b_b_lep_pt_true   = array( 'f', [ -1.] )
b_b_lep_y_true    = array( 'f', [ -1.] )
b_b_lep_phi_true  = array( 'f', [ -1.] )
b_t_lep_px_true   = array( 'f', [ -1.] )
b_t_lep_py_true   = array( 'f', [ -1.] )
b_t_lep_pz_true   = array( 'f', [ -1.] )
b_t_lep_E_true    = array( 'f', [ -1.] )
b_t_lep_m_true    = array( 'f', [ -1.] )
b_t_lep_pt_true   = array( 'f', [ -1.] )
b_t_lep_y_true    = array( 'f', [ -1.] )
b_t_lep_phi_true  = array( 'f', [ -1.] )

b_W_had_px_fitted   = array( 'f', [ -1.] )
b_W_had_py_fitted   = array( 'f', [ -1.] )
b_W_had_pz_fitted   = array( 'f', [ -1.] )
b_W_had_E_fitted    = array( 'f', [ -1.] )
b_W_had_m_fitted    = array( 'f', [ -1.] )
b_W_had_pt_fitted   = array( 'f', [ -1.] )
b_W_had_y_fitted    = array( 'f', [ -1.] )
b_W_had_phi_fitted  = array( 'f', [ -1.] )
b_b_had_px_fitted   = array( 'f', [ -1.] )
b_b_had_py_fitted   = array( 'f', [ -1.] )
b_b_had_pz_fitted   = array( 'f', [ -1.] )
b_b_had_E_fitted    = array( 'f', [ -1.] )
b_b_had_m_fitted    = array( 'f', [ -1.] )
b_b_had_pt_fitted   = array( 'f', [ -1.] )
b_b_had_y_fitted    = array( 'f', [ -1.] )
b_b_had_phi_fitted  = array( 'f', [ -1.] )
b_t_had_px_fitted   = array( 'f', [ -1.] )
b_t_had_py_fitted   = array( 'f', [ -1.] )
b_t_had_pz_fitted   = array( 'f', [ -1.] )
b_t_had_E_fitted    = array( 'f', [ -1.] )
b_t_had_m_fitted    = array( 'f', [ -1.] )
b_t_had_pt_fitted   = array( 'f', [ -1.] )
b_t_had_y_fitted    = array( 'f', [ -1.] )
b_t_had_phi_fitted  = array( 'f', [ -1.] )
b_W_lep_px_fitted   = array( 'f', [ -1.] )
b_W_lep_py_fitted   = array( 'f', [ -1.] )
b_W_lep_pz_fitted   = array( 'f', [ -1.] )
b_W_lep_E_fitted    = array( 'f', [ -1.] )
b_W_lep_m_fitted    = array( 'f', [ -1.] )
b_W_lep_pt_fitted   = array( 'f', [ -1.] )
b_W_lep_y_fitted    = array( 'f', [ -1.] )
b_W_lep_phi_fitted  = array( 'f', [ -1.] )
b_b_lep_px_fitted   = array( 'f', [ -1.] )
b_b_lep_py_fitted   = array( 'f', [ -1.] )
b_b_lep_pz_fitted   = array( 'f', [ -1.] )
b_b_lep_E_fitted    = array( 'f', [ -1.] )
b_b_lep_m_fitted    = array( 'f', [ -1.] )
b_b_lep_pt_fitted   = array( 'f', [ -1.] )
b_b_lep_y_fitted    = array( 'f', [ -1.] )
b_b_lep_phi_fitted  = array( 'f', [ -1.] )
b_t_lep_px_fitted   = array( 'f', [ -1.] )
b_t_lep_py_fitted   = array( 'f', [ -1.] )
b_t_lep_pz_fitted   = array( 'f', [ -1.] )
b_t_lep_E_fitted    = array( 'f', [ -1.] )
b_t_lep_m_fitted    = array( 'f', [ -1.] )
b_t_lep_pt_fitted   = array( 'f', [ -1.] )
b_t_lep_y_fitted    = array( 'f', [ -1.] )
b_t_lep_phi_fitted  = array( 'f', [ -1.] )

tree = TTree( "nominal", "nominal" )
tree.Branch( "eventNumber",     b_eventNumber,     'eventNumber/l' )
tree.Branch( 'runNumber',       b_runNumber,       'runNumber/i' )
tree.Branch( 'mcChannelNumber', b_mcChannelNumber, 'mcChannelNumber/i' )
tree.Branch( 'weight_mc',       b_weight_mc,       'weight_mc/F' )

tree.Branch( 'W_had_px_true',   b_W_had_px_true,   'W_had_px_true/F' )
tree.Branch( 'W_had_py_true',   b_W_had_py_true,   'W_had_py_true/F' )
tree.Branch( 'W_had_pz_true',   b_W_had_pz_true,   'W_had_pz_true/F' )
tree.Branch( 'W_had_E_true',    b_W_had_E_true,    'W_had_E_true/F' )
tree.Branch( 'W_had_m_true',    b_W_had_m_true,    'W_had_m_true/F' )
tree.Branch( 'W_had_pt_true',   b_W_had_pt_true,   'W_had_pt_true/F' )
tree.Branch( 'W_had_y_true',    b_W_had_y_true,    'W_had_y_true/F' )
tree.Branch( 'W_had_phi_true',  b_W_had_phi_true,  'W_had_phi_true/F' )
tree.Branch( 'b_had_px_true',   b_b_had_px_true,   'b_had_px_true/F' )
tree.Branch( 'b_had_py_true',   b_b_had_py_true,   'b_had_py_true/F' )
tree.Branch( 'b_had_pz_true',   b_b_had_pz_true,   'b_had_pz_true/F' )
tree.Branch( 'b_had_E_true',    b_b_had_E_true,    'b_had_E_true/F' )
tree.Branch( 'b_had_m_true',    b_b_had_m_true,    'b_had_m_true/F' )
tree.Branch( 'b_had_pt_true',   b_b_had_pt_true,   'b_had_pt_true/F' )
tree.Branch( 'b_had_y_true',    b_b_had_y_true,    'b_had_y_true/F' )
tree.Branch( 'b_had_phi_true',  b_b_had_phi_true,  'b_had_phi_true/F' )
tree.Branch( 't_had_px_true',   b_t_had_px_true,   't_had_px_true/F' )
tree.Branch( 't_had_py_true',   b_t_had_py_true,   't_had_py_true/F' )
tree.Branch( 't_had_pz_true',   b_t_had_pz_true,   't_had_pz_true/F' )
tree.Branch( 't_had_E_true',    b_t_had_E_true,    't_had_E_true/F' )
tree.Branch( 't_had_m_true',    b_t_had_m_true,    't_had_m_true/F' )
tree.Branch( 't_had_pt_true',   b_t_had_pt_true,   't_had_pt_true/F' )
tree.Branch( 't_had_y_true',    b_t_had_y_true,    't_had_y_true/F' )
tree.Branch( 't_had_phi_true',  b_t_had_phi_true,  't_had_phi_true/F' )
tree.Branch( 'W_lep_px_true',   b_W_lep_px_true,   'W_lep_px_true/F' )
tree.Branch( 'W_lep_py_true',   b_W_lep_py_true,   'W_lep_py_true/F' )
tree.Branch( 'W_lep_pz_true',   b_W_lep_pz_true,   'W_lep_pz_true/F' )
tree.Branch( 'W_lep_E_true',    b_W_lep_E_true,    'W_lep_E_true/F' )
tree.Branch( 'W_lep_m_true',    b_W_lep_m_true,    'W_lep_m_true/F' )
tree.Branch( 'W_lep_pt_true',   b_W_lep_pt_true,   'W_lep_pt_true/F' )
tree.Branch( 'W_lep_y_true',    b_W_lep_y_true,    'W_lep_y_true/F' )
tree.Branch( 'W_lep_phi_true',  b_W_lep_phi_true,  'W_lep_phi_true/F' )
tree.Branch( 'b_lep_px_true',   b_b_lep_px_true,   'b_lep_px_true/F' )
tree.Branch( 'b_lep_py_true',   b_b_lep_py_true,   'b_lep_py_true/F' )
tree.Branch( 'b_lep_pz_true',   b_b_lep_pz_true,   'b_lep_pz_true/F' )
tree.Branch( 'b_lep_E_true',    b_b_lep_E_true,    'b_lep_E_true/F' )
tree.Branch( 'b_lep_m_true',    b_b_lep_m_true,    'b_lep_m_true/F' )
tree.Branch( 'b_lep_pt_true',   b_b_lep_pt_true,   'b_lep_pt_true/F' )
tree.Branch( 'b_lep_y_true',    b_b_lep_y_true,    'b_lep_y_true/F' )
tree.Branch( 'b_lep_phi_true',  b_b_lep_phi_true,  'b_lep_phi_true/F' )
tree.Branch( 't_lep_px_true',   b_t_lep_px_true,   't_lep_px_true/F' )
tree.Branch( 't_lep_py_true',   b_t_lep_py_true,   't_lep_py_true/F' )
tree.Branch( 't_lep_pz_true',   b_t_lep_pz_true,   't_lep_pz_true/F' )
tree.Branch( 't_lep_E_true',    b_t_lep_E_true,    't_lep_E_true/F' )
tree.Branch( 't_lep_m_true',    b_t_lep_m_true,    't_lep_m_true/F' )
tree.Branch( 't_lep_pt_true',   b_t_lep_pt_true,   't_lep_pt_true/F' )
tree.Branch( 't_lep_y_true',    b_t_lep_y_true,    't_lep_y_true/F' )
tree.Branch( 't_lep_phi_true',  b_t_lep_phi_true,  't_lep_phi_true/F' )

tree.Branch( 'W_had_px_fitted',   b_W_had_px_fitted,   'W_had_px_fitted/F' )
tree.Branch( 'W_had_py_fitted',   b_W_had_py_fitted,   'W_had_py_fitted/F' )
tree.Branch( 'W_had_pz_fitted',   b_W_had_pz_fitted,   'W_had_pz_fitted/F' )
tree.Branch( 'W_had_E_fitted',    b_W_had_E_fitted,    'W_had_E_fitted/F' )
tree.Branch( 'W_had_m_fitted',    b_W_had_m_fitted,    'W_had_m_fitted/F' )
tree.Branch( 'W_had_pt_fitted',   b_W_had_pt_fitted,   'W_had_pt_fitted/F' )
tree.Branch( 'W_had_y_fitted',    b_W_had_y_fitted,    'W_had_y_fitted/F' )
tree.Branch( 'W_had_phi_fitted',  b_W_had_phi_fitted,  'W_had_phi_fitted/F' )
tree.Branch( 'b_had_px_fitted',   b_b_had_px_fitted,   'b_had_px_fitted/F' )
tree.Branch( 'b_had_py_fitted',   b_b_had_py_fitted,   'b_had_py_fitted/F' )
tree.Branch( 'b_had_pz_fitted',   b_b_had_pz_fitted,   'b_had_pz_fitted/F' )
tree.Branch( 'b_had_E_fitted',    b_b_had_E_fitted,    'b_had_E_fitted/F' )
tree.Branch( 'b_had_m_fitted',    b_b_had_m_fitted,    'b_had_m_fitted/F' )
tree.Branch( 'b_had_pt_fitted',   b_b_had_pt_fitted,   'b_had_pt_fitted/F' )
tree.Branch( 'b_had_y_fitted',    b_b_had_y_fitted,    'b_had_y_fitted/F' )
tree.Branch( 'b_had_phi_fitted',  b_b_had_phi_fitted,  'b_had_phi_fitted/F' )
tree.Branch( 't_had_px_fitted',   b_t_had_px_fitted,   't_had_px_fitted/F' )
tree.Branch( 't_had_py_fitted',   b_t_had_py_fitted,   't_had_py_fitted/F' )
tree.Branch( 't_had_pz_fitted',   b_t_had_pz_fitted,   't_had_pz_fitted/F' )
tree.Branch( 't_had_E_fitted',    b_t_had_E_fitted,    't_had_E_fitted/F' )
tree.Branch( 't_had_m_fitted',    b_t_had_m_fitted,    't_had_m_fitted/F' )
tree.Branch( 't_had_pt_fitted',   b_t_had_pt_fitted,   't_had_pt_fitted/F' )
tree.Branch( 't_had_y_fitted',    b_t_had_y_fitted,    't_had_y_fitted/F' )
tree.Branch( 't_had_phi_fitted',  b_t_had_phi_fitted,  't_had_phi_fitted/F' )
tree.Branch( 'W_lep_px_fitted',   b_W_lep_px_fitted,   'W_lep_px_fitted/F' )
tree.Branch( 'W_lep_py_fitted',   b_W_lep_py_fitted,   'W_lep_py_fitted/F' )
tree.Branch( 'W_lep_pz_fitted',   b_W_lep_pz_fitted,   'W_lep_pz_fitted/F' )
tree.Branch( 'W_lep_E_fitted',    b_W_lep_E_fitted,    'W_lep_E_fitted/F' )
tree.Branch( 'W_lep_m_fitted',    b_W_lep_m_fitted,    'W_lep_m_fitted/F' )
tree.Branch( 'W_lep_pt_fitted',   b_W_lep_pt_fitted,   'W_lep_pt_fitted/F' )
tree.Branch( 'W_lep_y_fitted',    b_W_lep_y_fitted,    'W_lep_y_fitted/F' )
tree.Branch( 'W_lep_phi_fitted',  b_W_lep_phi_fitted,  'W_lep_phi_fitted/F' )
tree.Branch( 'b_lep_px_fitted',   b_b_lep_px_fitted,   'b_lep_px_fitted/F' )
tree.Branch( 'b_lep_py_fitted',   b_b_lep_py_fitted,   'b_lep_py_fitted/F' )
tree.Branch( 'b_lep_pz_fitted',   b_b_lep_pz_fitted,   'b_lep_pz_fitted/F' )
tree.Branch( 'b_lep_E_fitted',    b_b_lep_E_fitted,    'b_lep_E_fitted/F' )
tree.Branch( 'b_lep_m_fitted',    b_b_lep_m_fitted,    'b_lep_m_fitted/F' )
tree.Branch( 'b_lep_pt_fitted',   b_b_lep_pt_fitted,   'b_lep_pt_fitted/F' )
tree.Branch( 'b_lep_y_fitted',    b_b_lep_y_fitted,    'b_lep_y_fitted/F' )
tree.Branch( 'b_lep_phi_fitted',  b_b_lep_phi_fitted,  'b_lep_phi_fitted/F' )
tree.Branch( 't_lep_px_fitted',   b_t_lep_px_fitted,   't_lep_px_fitted/F' )
tree.Branch( 't_lep_py_fitted',   b_t_lep_py_fitted,   't_lep_py_fitted/F' )
tree.Branch( 't_lep_pz_fitted',   b_t_lep_pz_fitted,   't_lep_pz_fitted/F' )
tree.Branch( 't_lep_E_fitted',    b_t_lep_E_fitted,    't_lep_E_fitted/F' )
tree.Branch( 't_lep_m_fitted',    b_t_lep_m_fitted,    't_lep_m_fitted/F' )
tree.Branch( 't_lep_pt_fitted',   b_t_lep_pt_fitted,   't_lep_pt_fitted/F' )
tree.Branch( 't_lep_y_fitted',    b_t_lep_y_fitted,    't_lep_y_fitted/F' )
tree.Branch( 't_lep_phi_fitted',  b_t_lep_phi_fitted,  't_lep_phi_fitted/F' )

print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    w = event_info[i][2]
    jets_n  = event_info[i][3]
    bjets_n = event_info[i][4]

    W_lep_true   = MakeP4( y_true_W_lep[i], m_W )
    W_lep_fitted = MakeP4( y_fitted[0][i],  m_W, max_momentum )

    W_had_true   = MakeP4( y_true_W_had[i], m_W )
    W_had_fitted = MakeP4( y_fitted[1][i],  m_W, max_momentum )

    b_lep_true   = MakeP4( y_true_b_lep[i], m_b )
    b_lep_fitted = MakeP4( y_fitted[2][i],  m_b, max_momentum )

    b_had_true   = MakeP4( y_true_b_had[i], m_b )
    b_had_fitted = MakeP4( y_fitted[3][i],  m_b, max_momentum )

    t_lep_true   = MakeP4( y_true_t_lep[i], m_t )
    #t_lep_fitted = MakeP4( y_fitted[4][i],  m_t, max_momentum )
    t_lep_fitted = W_lep_fitted + b_lep_fitted

    t_had_true   = MakeP4( y_true_t_had[i], m_t )
    #t_had_fitted = MakeP4( y_fitted[5][i],  m_t, max_momentum )
    t_had_fitted = W_had_fitted + b_had_fitted

    # fill branches
    b_eventNumber[0] = int(event_info[i][0])
    b_runNumber[0]   = int(event_info[i][1])
    #b_weight_mc[0]   = float(event_info[i][2])

    b_W_had_px_true[0]  = W_had_true.Px()
    b_W_had_py_true[0]  = W_had_true.Py()
    b_W_had_pz_true[0]  = W_had_true.Pz()
    b_W_had_E_true[0]   = W_had_true.E()
    b_W_had_m_true[0]   = W_had_true.M()
    b_W_had_pt_true[0]  = W_had_true.Pt()
    b_W_had_y_true[0]   = W_had_true.Rapidity()
    b_W_had_phi_true[0] = W_had_true.Phi()

    b_b_had_px_true[0]  = b_had_true.Px()
    b_b_had_py_true[0]  = b_had_true.Py()
    b_b_had_pz_true[0]  = b_had_true.Pz()
    b_b_had_E_true[0]   = b_had_true.E()
    b_b_had_m_true[0]   = b_had_true.M()
    b_b_had_pt_true[0]  = b_had_true.Pt()
    b_b_had_y_true[0]   = b_had_true.Rapidity()
    b_b_had_phi_true[0] = b_had_true.Phi()

    b_t_had_px_true[0]  = t_had_true.Px()
    b_t_had_py_true[0]  = t_had_true.Py()
    b_t_had_pz_true[0]  = t_had_true.Pz()
    b_t_had_E_true[0]   = t_had_true.E()
    b_t_had_m_true[0]   = t_had_true.M()
    b_t_had_pt_true[0]  = t_had_true.Pt()
    b_t_had_y_true[0]   = t_had_true.Rapidity()
    b_t_had_phi_true[0] = t_had_true.Phi()

    b_W_lep_px_true[0]  = W_lep_true.Px()
    b_W_lep_py_true[0]  = W_lep_true.Py()
    b_W_lep_pz_true[0]  = W_lep_true.Pz()
    b_W_lep_E_true[0]   = W_lep_true.E()
    b_W_lep_m_true[0]   = W_lep_true.M()
    b_W_lep_pt_true[0]  = W_lep_true.Pt()
    b_W_lep_y_true[0]   = W_lep_true.Rapidity()
    b_W_lep_phi_true[0] = W_lep_true.Phi()

    b_b_lep_px_true[0]  = b_lep_true.Px()
    b_b_lep_py_true[0]  = b_lep_true.Py()
    b_b_lep_pz_true[0]  = b_lep_true.Pz()
    b_b_lep_E_true[0]   = b_lep_true.E()
    b_b_lep_m_true[0]   = b_lep_true.M()
    b_b_lep_pt_true[0]  = b_lep_true.Pt()
    b_b_lep_y_true[0]   = b_lep_true.Rapidity()
    b_b_lep_phi_true[0] = b_lep_true.Phi()

    b_t_lep_px_true[0]  = t_lep_true.Px()
    b_t_lep_py_true[0]  = t_lep_true.Py()
    b_t_lep_pz_true[0]  = t_lep_true.Pz()
    b_t_lep_E_true[0]   = t_lep_true.E()
    b_t_lep_m_true[0]   = t_lep_true.M()
    b_t_lep_pt_true[0]  = t_lep_true.Pt()
    b_t_lep_y_true[0]   = t_lep_true.Rapidity()
    b_t_lep_phi_true[0] = t_lep_true.Phi()

    b_W_had_px_fitted[0]  = W_had_fitted.Px()
    b_W_had_py_fitted[0]  = W_had_fitted.Py()
    b_W_had_pz_fitted[0]  = W_had_fitted.Pz()
    b_W_had_E_fitted[0]   = W_had_fitted.E()
    b_W_had_m_fitted[0]   = W_had_fitted.M()
    b_W_had_pt_fitted[0]  = W_had_fitted.Pt()
    b_W_had_y_fitted[0]   = W_had_fitted.Rapidity()
    b_W_had_phi_fitted[0] = W_had_fitted.Phi()

    b_b_had_px_fitted[0]  = b_had_fitted.Px()
    b_b_had_py_fitted[0]  = b_had_fitted.Py()
    b_b_had_pz_fitted[0]  = b_had_fitted.Pz()
    b_b_had_E_fitted[0]   = b_had_fitted.E()
    b_b_had_m_fitted[0]   = b_had_fitted.M()
    b_b_had_pt_fitted[0]  = b_had_fitted.Pt()
    b_b_had_y_fitted[0]   = b_had_fitted.Rapidity()
    b_b_had_phi_fitted[0] = b_had_fitted.Phi()

    b_t_had_px_fitted[0]  = t_had_fitted.Px()
    b_t_had_py_fitted[0]  = t_had_fitted.Py()
    b_t_had_pz_fitted[0]  = t_had_fitted.Pz()
    b_t_had_E_fitted[0]   = t_had_fitted.E()
    b_t_had_m_fitted[0]   = t_had_fitted.M()
    b_t_had_pt_fitted[0]  = t_had_fitted.Pt()
    b_t_had_y_fitted[0]   = t_had_fitted.Rapidity()
    b_t_had_phi_fitted[0] = t_had_fitted.Phi()

    b_W_lep_px_fitted[0]  = W_lep_fitted.Px()
    b_W_lep_py_fitted[0]  = W_lep_fitted.Py()
    b_W_lep_pz_fitted[0]  = W_lep_fitted.Pz()
    b_W_lep_E_fitted[0]   = W_lep_fitted.E()
    b_W_lep_m_fitted[0]   = W_lep_fitted.M()
    b_W_lep_pt_fitted[0]  = W_lep_fitted.Pt()
    b_W_lep_y_fitted[0]   = W_lep_fitted.Rapidity()
    b_W_lep_phi_fitted[0] = W_lep_fitted.Phi()

    b_b_lep_px_fitted[0]  = b_lep_fitted.Px()
    b_b_lep_py_fitted[0]  = b_lep_fitted.Py()
    b_b_lep_pz_fitted[0]  = b_lep_fitted.Pz()
    b_b_lep_E_fitted[0]   = b_lep_fitted.E()
    b_b_lep_m_fitted[0]   = b_lep_fitted.M()
    b_b_lep_pt_fitted[0]  = b_lep_fitted.Pt()
    b_b_lep_y_fitted[0]   = b_lep_fitted.Rapidity()
    b_b_lep_phi_fitted[0] = b_lep_fitted.Phi()

    b_t_lep_px_fitted[0]  = t_lep_fitted.Px()
    b_t_lep_py_fitted[0]  = t_lep_fitted.Py()
    b_t_lep_pz_fitted[0]  = t_lep_fitted.Pz()
    b_t_lep_E_fitted[0]   = t_lep_fitted.E()
    b_t_lep_m_fitted[0]   = t_lep_fitted.M()
    b_t_lep_pt_fitted[0]  = t_lep_fitted.Pt()
    b_t_lep_y_fitted[0]   = t_lep_fitted.Rapidity()
    b_t_lep_phi_fitted[0] = t_lep_fitted.Phi()

    tree.Fill()

    n_good += 1

    if i < 10:
      PrintOut( t_had_true, t_had_fitted, event_info[i], "Hadronic top" )
      PrintOut( t_lep_true, t_lep_fitted, event_info[i], "Leptonic top" )

ofile.Write()
ofile.Close()

print("Finished. Saved output file:", ofilename)

f_good = 100. * float( n_good ) / float( n_events )
print("Good events: %.2f" % f_good)
