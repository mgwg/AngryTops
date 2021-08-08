#!/usr/bin/env python
import os, sys, time
import argparse
from AngryTops.features import *
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
from AngryTops.Plotting.identification_helper import * 

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
outputdir = "/img_chi_fwhm/"
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
fwhm = {}
for obs in attributes:

    hist_name = "diff_{0}".format(obs)
    # True and fitted leaf
    hist = histsFile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)

    #Normalize(hist)
    if hist.Class() == TH2F.Class():
        hist = hist.ProfileX("hist_pfx")

    # Extract FWHM from helper function.
    fwhm_single = getFwhm( hist )
    fwhm[obs] = fwhm_single

################################################################################
histograms = {}

# Distribution of Chi-Squareds
histograms['chi_squared_all'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_all'].SetTitle("#chi^{2} of all events;Unitless;A.U.")
histograms['chi_squared_all_NDF'] = TH1F("#chi^{2}/NDF",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_all_NDF'].SetTitle("#chi^{2}/NDF of all events;Unitless;A.U.")

################################################################################

# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)

# Define sums of squares to be used for calculating sample chi-squared/NDF
# One sum for each variable to be augmented in each event
W_had_phi_sum = 0.
W_had_rapidity_sum = 0.
W_had_pt_sum = 0.

W_lep_phi_sum = 0.
W_lep_rapidity_sum = 0.
W_lep_pt_sum = 0.

b_had_phi_sum = 0.
b_had_rapidity_sum = 0.
b_had_pt_sum = 0.

b_lep_phi_sum = 0.
b_lep_rapidity_sum = 0.
b_lep_pt_sum = 0.


# Iterate through all events
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

    # Calculate the squared differences between truth and predicted variables
    W_had_phi_diff = ( W_had_true.Phi() - W_had_fitted.Phi() )**2
    W_had_rapidity_diff = ( W_had_true.Rapidity() - W_had_fitted.Rapidity() )**2
    W_had_pt_diff = ( W_had_true.Pt() - W_had_fitted.Pt() )**2

    W_lep_phi_diff = ( W_lep_true.Phi() - W_lep_fitted.Phi() )**2
    W_lep_rapidity_diff = ( W_lep_true.Rapidity() - W_lep_fitted.Rapidity() )**2
    W_lep_pt_diff = ( W_lep_true.Pt() - W_lep_fitted.Pt() )**2

    b_had_phi_diff = ( b_had_true.Phi() - b_had_fitted.Phi() )**2
    b_had_rapidity_diff = ( b_had_true.Rapidity() - b_had_fitted.Rapidity() )**2
    b_had_pt_diff = ( b_had_true.Pt() - b_had_fitted.Pt() )**2

    b_lep_phi_diff = ( b_lep_true.Phi() - b_lep_fitted.Phi() )**2
    b_lep_rapidity_diff = ( b_lep_true.Rapidity() - b_lep_fitted.Rapidity() )**2
    b_lep_pt_diff = ( b_lep_true.Pt() - b_lep_fitted.Pt() )**2

    # Chi-squared for each variable over all events
    W_had_phi_sum += W_had_phi_diff
    W_had_rapidity_sum += W_had_rapidity_diff
    W_had_pt_sum += W_had_pt_diff

    W_lep_phi_sum += W_lep_phi_diff
    W_lep_rapidity_sum += W_lep_rapidity_diff
    W_lep_pt_sum += W_lep_pt_diff

    b_had_phi_sum += b_had_phi_diff
    b_had_rapidity_sum += b_had_rapidity_diff
    b_had_pt_sum += b_had_pt_diff

    b_lep_phi_sum += b_lep_phi_diff
    b_lep_rapidity_sum += b_lep_rapidity_diff
    b_lep_pt_sum += b_lep_pt_diff

    # chi squared for each event using all variables... 
    chi22 = 0.

    chi22 += W_had_phi_diff / ( fwhm['W_had_phi']**2 )
    chi22 += W_had_rapidity_diff / ( fwhm['W_had_y']**2 )
    chi22 += W_had_pt_diff / ( fwhm['W_had_pt']**2 )

    chi22 += W_lep_phi_diff / ( fwhm['W_lep_phi']**2 )
    chi22 += W_lep_rapidity_diff / ( fwhm['W_lep_y']**2 )
    chi22 += W_lep_pt_diff / ( fwhm['W_lep_pt']**2 )

    chi22 += b_had_phi_diff / ( fwhm['b_had_phi']**2 )
    chi22 += b_had_rapidity_diff / ( fwhm['b_had_y']**2 )
    chi22 += b_had_pt_diff / ( fwhm['b_had_pt']**2 )

    chi22 += b_lep_phi_diff / ( fwhm['b_lep_phi']**2 )
    chi22 += b_lep_rapidity_diff / ( fwhm['b_lep_y']**2 )
    chi22 += b_lep_pt_diff / ( fwhm['b_lep_pt']**2 )

    # Calculate chi-squared/NDF
    chi22NDF = chi22 / 12.0

    # Populate the histograms:
    histograms['chi_squared_all'].Fill(chi22)
    histograms['chi_squared_all_NDF'].Fill(chi22NDF)


# Normalize sums of squares by standard deviations and number of events
W_had_phi_chi2NDF = W_had_phi_sum / n_events / ( fwhm['W_had_phi']**2 )
W_had_rapidity_chi2NDF = W_had_rapidity_sum / n_events / ( fwhm['W_had_y']**2 )
W_had_pt_chi2NDF = W_had_pt_sum / n_events / ( fwhm['W_had_pt']**2 )

W_lep_phi_chi2NDF = W_lep_phi_sum / n_events / ( fwhm['W_lep_phi']**2 )
W_lep_rapidity_chi2NDF = W_lep_rapidity_sum / n_events / ( fwhm['W_lep_y']**2 )
W_lep_pt_chi2NDF = W_lep_pt_sum / n_events / ( fwhm['W_lep_pt']**2 )

b_had_phi_chi2NDF = b_had_phi_sum / n_events / ( fwhm['b_had_phi']**2 )
b_had_rapidity_chi2NDF = b_had_rapidity_sum / n_events / ( fwhm['b_had_y']**2 )
b_had_pt_chi2NDF = b_had_pt_sum / n_events / ( fwhm['b_had_pt']**2 )

b_lep_phi_chi2NDF = b_lep_phi_sum / n_events / ( fwhm['b_lep_phi']**2 )
b_lep_rapidity_chi2NDF = b_lep_rapidity_sum / n_events / ( fwhm['b_lep_y']**2 )
b_lep_pt_chi2NDF = b_lep_pt_sum / n_events / ( fwhm['b_lep_pt']**2 )

# Print Chi-Squareds/NDF.
print("\nW_had_phi_chi2NDF: {0}".format(W_had_phi_chi2NDF))
print("W_had_rapidity_chi2NDF: {0}".format(W_had_rapidity_chi2NDF))
print("W_had_pt_chi2NDF: {0}\n".format(W_had_pt_chi2NDF))

print("W_lep_phi_chi2NDF: {0}".format(W_lep_phi_chi2NDF))
print("W_lep_rapidity_chi2NDF: {0}".format(W_lep_rapidity_chi2NDF))
print("W_lep_pt_chi2NDF: {0}\n".format(W_lep_pt_chi2NDF))

print("b_had_phi_chi2NDF: {0}".format(b_had_phi_chi2NDF))
print("b_had_rapidity_chi2NDF: {0}".format(b_had_rapidity_chi2NDF))
print("b_had_pt_chi2NDF: {0}\n".format(b_had_pt_chi2NDF))

print("b_lep_phi_chi2NDF: {0}".format(b_lep_phi_chi2NDF))
print("b_lep_rapidity_chi2NDF: {0}".format(b_lep_rapidity_chi2NDF))
print("b_lep_pt_chi2NDF: {0}".format(b_lep_pt_chi2NDF))


try:
    os.mkdir('{}/{}'.format(training_dir, outputdir))
except Exception as e:
    print("Overwriting existing files")

# Plot histograms inside outputdir, a subdir of training_dir
for key in histograms:
    plot_hists(key, histograms[key], training_dir+outputdir)
