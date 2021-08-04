#!/usr/bin/env python
import os, sys, time
import argparse
from AngryTops.features import *
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
from AngryTops.Plotting.PlottingHelper import *

gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

################################################################################
# CONSTANTS
m_t = 172.5
m_W = 80.4
m_b = 4.95
if len(sys.argv) > 1: training_dir = sys.argv[1]
infilename = "{}/fitted.root".format(training_dir)
print(infilename)
caption = sys.argv[2]
if caption == "None": caption = None
outputdir = "img_chi_test"
if len(sys.argv) > 3:
    outputdir += sys.argv[3]

histsFilename = "{}/histograms.root".format(training_dir)
histsFile = TFile.Open(histsFilename)

################################################################################
# HELPER FUNCTIONS
def PrintOut( p4_true, p4_fitted, label ):
  print("%s :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) \
        :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
        (label, p4_true.Pt(), p4_true.Rapidity(), p4_true.Phi(), p4_true.E(), p4_true.M(),\
        p4_fitted.Pt(), p4_fitted.Rapidity(), p4_fitted.Phi(), p4_fitted.E(), p4_fitted.M()
        ))

################################################################################
# Read in input file
infile = TFile.Open( infilename )
tree   = infile.Get( "nominal")

################################################################################
# Draw Differences and resonances
sigma = {}
for obs in attributes:

    hist_name = "diff_{0}".format(obs)
    # True and fitted leaf
    hist = histsFile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)

    #Normalize(hist)
    if hist.Class() == TH2F.Class():
        hist = hist.ProfileX("hist_pfx")

    fwhm_single, sigma_single = getFwhm( hist )
    sigma[obs] = sigma_single

# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0.

W_had_phi = 0.
W_had_y = 0.
W_had_Pt = 0.

W_lep_phi = 0.
W_lep_y = 0.
W_lep_Pt = 0.

b_had_phi = 0.
b_had_y = 0.
b_had_Pt = 0.

b_lep_phi = 0.
b_lep_y = 0.
b_lep_Pt = 0.

# Print out example
for i in range(n_events):
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    tree.GetEntry(i)

    w = tree.weight_mc

    W_had_true = TLorentzVector(tree.W_had_px_true, tree.W_had_py_true, tree.W_had_pz_true, tree.W_had_E_true)
    b_had_true = TLorentzVector(tree.b_had_px_true, tree.b_had_py_true, tree.b_had_pz_true, tree.b_had_E_true)
    t_had_true = TLorentzVector(tree.t_had_px_true, tree.t_had_py_true, tree.t_had_pz_true, tree.t_had_E_true)
    W_lep_true = TLorentzVector(tree.W_lep_px_true, tree.W_lep_py_true, tree.W_lep_pz_true, tree.W_lep_E_true)
    b_lep_true = TLorentzVector(tree.b_lep_px_true, tree.b_lep_py_true, tree.b_lep_pz_true, tree.b_lep_E_true)
    t_lep_true = TLorentzVector(tree.t_lep_px_true, tree.t_lep_py_true, tree.t_lep_pz_true, tree.t_lep_E_true)

    W_had_fitted = TLorentzVector(tree.W_had_px_fitted, tree.W_had_py_fitted, tree.W_had_pz_fitted, tree.W_had_E_fitted)
    b_had_fitted = TLorentzVector(tree.b_had_px_fitted, tree.b_had_py_fitted, tree.b_had_pz_fitted, tree.b_had_E_fitted)
    t_had_fitted = TLorentzVector(tree.t_had_px_fitted, tree.t_had_py_fitted, tree.t_had_pz_fitted, tree.t_had_E_fitted)
    W_lep_fitted = TLorentzVector(tree.W_lep_px_fitted, tree.W_lep_py_fitted, tree.W_lep_pz_fitted, tree.W_lep_E_fitted)
    b_lep_fitted = TLorentzVector(tree.b_lep_px_fitted, tree.b_lep_py_fitted, tree.b_lep_pz_fitted, tree.b_lep_E_fitted)
    t_lep_fitted = TLorentzVector(tree.t_lep_px_fitted, tree.t_lep_py_fitted, tree.t_lep_pz_fitted, tree.t_lep_E_fitted)

    W_had_phi += (W_had_true.Phi() - W_had_fitted.Phi())**2
    W_had_y += (W_had_true.Rapidity() - W_had_fitted.Rapidity())**2
    W_had_Pt += (W_had_true.Pt() - W_had_fitted.Pt())**2

    W_lep_phi += (W_lep_true.Phi() - W_lep_fitted.Phi())**2
    W_lep_y += (W_lep_true.Rapidity() - W_lep_fitted.Rapidity())**2
    W_lep_Pt += (W_lep_true.Pt() - W_lep_fitted.Pt())**2

    b_had_phi += (b_had_true.Phi() - b_had_fitted.Phi())**2
    b_had_y += (b_had_true.Rapidity() - b_had_fitted.Rapidity())**2
    b_had_Pt += (b_had_true.Pt() - b_had_fitted.Pt())**2

    b_lep_phi += (b_lep_true.Phi() - b_lep_fitted.Phi())**2
    b_lep_y += (b_lep_true.Rapidity() - b_lep_fitted.Rapidity())**2
    b_lep_Pt += (b_lep_true.Pt() - b_lep_fitted.Pt())**2


W_had_phi = W_had_phi/( sigma['W_had_phi']**2 )
W_had_y = W_had_y/( sigma['W_had_y']**2 )
W_had_Pt = W_had_Pt/( sigma['W_had_pt']**2 )

W_lep_phi = W_lep_phi/( sigma['W_lep_phi']**2 )
W_lep_y = W_lep_y/( sigma['W_lep_y']**2 )
W_lep_Pt = W_lep_Pt/( sigma['W_lep_pt']**2 )

b_had_phi = b_had_phi/( sigma['b_had_phi']**2 )
b_had_y = b_had_y/( sigma['b_had_y']**2 )
b_had_Pt = b_had_Pt/( sigma['b_had_pt']**2 )

b_lep_phi = b_lep_phi/( sigma['b_lep_phi']**2 )
b_lep_y = b_lep_y/( sigma['b_lep_y']**2 )
b_lep_Pt = b_lep_Pt/( sigma['b_lep_pt']**2 )

print("W had phi: {}".format(W_had_phi/n_events))
print("W had y: {}".format(W_had_y/n_events))
print("W had pt: {}".format(W_had_Pt/n_events))

print("W lep phi: {}".format(W_lep_phi/n_events))
print("W lep y: {}".format(W_lep_y/n_events))
print("W lep pt: {}".format(W_lep_Pt/n_events))

print("b had phi: {}".format(b_had_phi/n_events))
print("b had y: {}".format(b_had_y/n_events))
print("b had pt: {}".format(b_had_Pt/n_events))

print("b lep phi: {}".format(b_lep_phi/n_events))
print("b lep y: {}".format(b_lep_y/n_events))
print("b lep pt: {}".format(b_lep_Pt/n_events))