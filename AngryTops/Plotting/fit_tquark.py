#!/usr/bin/env python
import os, sys, time
import argparse
from ROOT import *
from array import array
import pickle
import numpy as np
import sklearn.preprocessing
from AngryTops.features import *
################################################################################
# CONSTANTS
training_dir = sys.argv[1]
representation = sys.argv[2]
scaling = True
if len(sys.argv) > 3: scaling = False
m_t = 172.5
np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(training_dir)

################################################################################
# HELPER FUNCTIONS

def PrintOut( p4_true, p4_fitted, event_info, label ):
    """
    Prints the true and predicted momentum vector
    """
    print("rn=%-10i en=%-10i ) %s :: true=(%4.1f, %3.2f, %3.2f, %4.1f ; %3.1f) \
           :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
          (event_info[0], event_info[1], label,
           p4_true.Px(),   p4_true.Py(),   p4_true.Pz(),   p4_true.Pt(),   p4_true.E(),\
           p4_fitted.Px(), p4_fitted.Py(), p4_fitted.Pz(), p4_fitted.Pt(), p4_fitted.E()))

def MakeP4(y, m):
    """
    Form the momentum vector.
    """
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[2]
    # Construction of momentum vector depends on the representation of the input
    if representation == "pxpypzE":
        E  = y[3]
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypzM":
        M  = y[3]
        E = np.sqrt(p0**2 + p1**2 + p2**2 + M**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypz" or representation == "pxpypzEM":
        E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "ptetaphiE":
        E  = y[3]
        p4.SetPtEtaPhiE(p0, p1, p2, E)
    elif representation == "ptetaphiM":
        M  = y[3]
        p4.SetPtEtaPhiM(p0, p1, p2, M)
    elif representation == "ptetaphi" or representation == "ptetaphiEM":
        p4.SetPtEtaPhiM(p0, p1, p2, m)
    else:
        raise Exception("Invalid Representation Given: {}".format(representation))
    return p4

################################################################################
# Load Predictions
print("INFO: fitting ttbar decay chain...")
predictions = np.load('{}/predictions.npz'.format(training_dir))
true = predictions['true']
y_fitted = predictions['pred']
event_info = predictions['events']
# Keep track of the old shape: (# of test particles, # of features per
# test particle, number of features for input)
old_shape = (true.shape[1], true.shape[2])

################################################################################
# UNDO NORMALIZATIONS
# Import scalars
scaler_filename = "{}/scalers.pkl".format(training_dir)
with open( scaler_filename, "rb" ) as file_scaler:
  jets_scalar = pickle.load(file_scaler)
  lep_scalar = pickle.load(file_scaler)
  output_scalar = pickle.load(file_scaler)

if scaling:
    # Rescale the truth array
    true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
    true = output_scalar.inverse_transform(true)
    true = true.reshape(true.shape[0], old_shape[0], old_shape[1])

    # Rescale the fitted array
    y_fitted = y_fitted.reshape(y_fitted.shape[0], y_fitted.shape[1]*y_fitted.shape[2])
    y_fitted = output_scalar.inverse_transform(y_fitted)
    y_fitted = y_fitted.reshape(y_fitted.shape[0], old_shape[0], old_shape[1])
################################################################################
# Truth
y_true_t_had = true[:,0,:]
y_true_t_lep = true[:,1,:]

# Fitted
y_fitted_t_had = y_fitted[:,0,:]
y_fitted_t_lep = y_fitted[:,1,:]

# Event Info
n_events = true.shape[0]
w = 1
print("Shape of tions: ", y_fitted.shape)
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

b_t_had_px_true   = array( 'f', [ -1.] )
b_t_had_py_true   = array( 'f', [ -1.] )
b_t_had_pz_true   = array( 'f', [ -1.] )
b_t_had_E_true    = array( 'f', [ -1.] )
b_t_had_m_true    = array( 'f', [ -1.] )
b_t_had_pt_true   = array( 'f', [ -1.] )
b_t_had_y_true    = array( 'f', [ -1.] )
b_t_had_phi_true  = array( 'f', [ -1.] )
b_t_lep_px_true   = array( 'f', [ -1.] )
b_t_lep_py_true   = array( 'f', [ -1.] )
b_t_lep_pz_true   = array( 'f', [ -1.] )
b_t_lep_E_true    = array( 'f', [ -1.] )
b_t_lep_m_true    = array( 'f', [ -1.] )
b_t_lep_pt_true   = array( 'f', [ -1.] )
b_t_lep_y_true    = array( 'f', [ -1.] )
b_t_lep_phi_true  = array( 'f', [ -1.] )

b_t_had_px_fitted   = array( 'f', [ -1.] )
b_t_had_py_fitted   = array( 'f', [ -1.] )
b_t_had_pz_fitted   = array( 'f', [ -1.] )
b_t_had_E_fitted    = array( 'f', [ -1.] )
b_t_had_m_fitted    = array( 'f', [ -1.] )
b_t_had_pt_fitted   = array( 'f', [ -1.] )
b_t_had_y_fitted    = array( 'f', [ -1.] )
b_t_had_phi_fitted  = array( 'f', [ -1.] )
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

tree.Branch( 't_had_px_true',   b_t_had_px_true,   't_had_px_true/F' )
tree.Branch( 't_had_py_true',   b_t_had_py_true,   't_had_py_true/F' )
tree.Branch( 't_had_pz_true',   b_t_had_pz_true,   't_had_pz_true/F' )
tree.Branch( 't_had_E_true',    b_t_had_E_true,    't_had_E_true/F' )
tree.Branch( 't_had_m_true',    b_t_had_m_true,    't_had_m_true/F' )
tree.Branch( 't_had_pt_true',   b_t_had_pt_true,   't_had_pt_true/F' )
tree.Branch( 't_had_y_true',    b_t_had_y_true,    't_had_y_true/F' )
tree.Branch( 't_had_phi_true',  b_t_had_phi_true,  't_had_phi_true/F' )

tree.Branch( 't_lep_px_true',   b_t_lep_px_true,   't_lep_px_true/F' )
tree.Branch( 't_lep_py_true',   b_t_lep_py_true,   't_lep_py_true/F' )
tree.Branch( 't_lep_pz_true',   b_t_lep_pz_true,   't_lep_pz_true/F' )
tree.Branch( 't_lep_E_true',    b_t_lep_E_true,    't_lep_E_true/F' )
tree.Branch( 't_lep_m_true',    b_t_lep_m_true,    't_lep_m_true/F' )
tree.Branch( 't_lep_pt_true',   b_t_lep_pt_true,   't_lep_pt_true/F' )
tree.Branch( 't_lep_y_true',    b_t_lep_y_true,    't_lep_y_true/F' )
tree.Branch( 't_lep_phi_true',  b_t_lep_phi_true,  't_lep_phi_true/F' )

tree.Branch( 't_had_px_fitted',   b_t_had_px_fitted,   't_had_px_fitted/F' )
tree.Branch( 't_had_py_fitted',   b_t_had_py_fitted,   't_had_py_fitted/F' )
tree.Branch( 't_had_pz_fitted',   b_t_had_pz_fitted,   't_had_pz_fitted/F' )
tree.Branch( 't_had_E_fitted',    b_t_had_E_fitted,    't_had_E_fitted/F' )
tree.Branch( 't_had_m_fitted',    b_t_had_m_fitted,    't_had_m_fitted/F' )
tree.Branch( 't_had_pt_fitted',   b_t_had_pt_fitted,   't_had_pt_fitted/F' )
tree.Branch( 't_had_y_fitted',    b_t_had_y_fitted,    't_had_y_fitted/F' )
tree.Branch( 't_had_phi_fitted',  b_t_had_phi_fitted,  't_had_phi_fitted/F' )

tree.Branch( 't_lep_px_fitted',   b_t_lep_px_fitted,   't_lep_px_fitted/F' )
tree.Branch( 't_lep_py_fitted',   b_t_lep_py_fitted,   't_lep_py_fitted/F' )
tree.Branch( 't_lep_pz_fitted',   b_t_lep_pz_fitted,   't_lep_pz_fitted/F' )
tree.Branch( 't_lep_E_fitted',    b_t_lep_E_fitted,    't_lep_E_fitted/F' )
tree.Branch( 't_lep_m_fitted',    b_t_lep_m_fitted,    't_lep_m_fitted/F' )
tree.Branch( 't_lep_pt_fitted',   b_t_lep_pt_fitted,   't_lep_pt_fitted/F' )
tree.Branch( 't_lep_y_fitted',    b_t_lep_y_fitted,    't_lep_y_fitted/F' )
tree.Branch( 't_lep_phi_fitted',  b_t_lep_phi_fitted,  't_lep_phi_fitted/F' )

################################################################################
# POPULATE TREE
print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0
w = 1  # Set all weights to 1 for now
# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    jets_n  = event_info[i][3]
    bjets_n = event_info[i][4]

    t_had_true   = MakeP4( y_true_t_had[i], m_t )
    t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t)

    t_lep_true   = MakeP4( y_true_t_lep[i], m_t )
    t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t)

    # Fill branches
    b_eventNumber[0] = int(event_info[i][0])
    b_runNumber[0]   = int(event_info[i][1])
    b_weight_mc[0]   = float(event_info[i][2])

    b_t_had_px_true[0]  = t_had_true.Px()
    b_t_had_py_true[0]  = t_had_true.Py()
    b_t_had_pz_true[0]  = t_had_true.Pz()
    b_t_had_E_true[0]   = t_had_true.E()
    b_t_had_m_true[0]   = t_had_true.M()
    b_t_had_pt_true[0]  = t_had_true.Pt()
    b_t_had_y_true[0]   = t_had_true.Rapidity()
    b_t_had_phi_true[0] = t_had_true.Phi()

    b_t_lep_px_true[0]  = t_lep_true.Px()
    b_t_lep_py_true[0]  = t_lep_true.Py()
    b_t_lep_pz_true[0]  = t_lep_true.Pz()
    b_t_lep_E_true[0]   = t_lep_true.E()
    b_t_lep_m_true[0]   = t_lep_true.M()
    b_t_lep_pt_true[0]  = t_lep_true.Pt()
    b_t_lep_y_true[0]   = t_lep_true.Rapidity()
    b_t_lep_phi_true[0] = t_lep_true.Phi()

    b_t_had_px_fitted[0]  = t_had_fitted.Px()
    b_t_had_py_fitted[0]  = t_had_fitted.Py()
    b_t_had_pz_fitted[0]  = t_had_fitted.Pz()
    b_t_had_E_fitted[0]   = t_had_fitted.E()
    b_t_had_m_fitted[0]   = t_had_fitted.M()
    b_t_had_pt_fitted[0]  = t_had_fitted.Pt()
    b_t_had_y_fitted[0]   = t_had_fitted.Rapidity()
    b_t_had_phi_fitted[0] = t_had_fitted.Phi()

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

################################################################################
# CLOSE PROGRAM
ofile.Write()
ofile.Close()

print("Finished. Saved output file:", ofilename)

f_good = 100. * float( n_good ) / float( n_events )
print("Good events: %.2f" % f_good)
