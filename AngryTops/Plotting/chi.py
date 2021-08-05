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
outputdir = "img_chi_fwhm"
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

    # Extract FWHM and std. deviation from helper function.
    fwhm_single, sigma_single = getFwhm( hist )
    fwhm[obs] = fwhm_single
    sigma[obs] = sigma_single

################################################################################
histograms = {}

# True
histograms['W_had_px_true']       = TH1F( "W_had_px_true",  ";Hadronic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_had_py_true']       = TH1F( "W_had_py_true",  ";Hadronic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_had_pz_true']       = TH1F( "W_had_pz_true",  ";Hadronic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_had_pt_true']       = TH1F( "W_had_pt_true",  ";Hadronic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_had_y_true']        = TH1F( "W_had_y_true",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_true']      = TH1F( "W_had_phi_true", ";Hadronic W #phi", 16, -3.2, 3.2 )
histograms['W_had_E_true']        = TH1F( "W_had_E_true",   ";Hadronic W E [GeV]", 50, 0., 500. )
histograms['W_had_m_true']        = TH1F( "W_had_m_true",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_px_true']       = TH1F( "b_had_px_true",  ";Hadronic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_had_py_true']       = TH1F( "b_had_py_true",  ";Hadronic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_had_pz_true']       = TH1F( "b_had_pz_true",  ";Hadronic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_had_pt_true']       = TH1F( "b_had_pt_true",  ";Hadronic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_had_y_true']        = TH1F( "b_had_y_true",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_true']      = TH1F( "b_had_phi_true", ";Hadronic b #phi", 16, -3.2, 3.2 )
histograms['b_had_E_true']        = TH1F( "b_had_E_true",   ";Hadronic b E [GeV]", 50, 0., 500. )
histograms['b_had_m_true']        = TH1F( "b_had_m_true",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_px_true']       = TH1F( "t_had_px_true",  ";Hadronic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_had_py_true']       = TH1F( "t_had_py_true",  ";Hadronic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_had_pz_true']       = TH1F( "t_had_pz_true",  ";Hadronic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_had_pt_true']       = TH1F( "t_had_pt_true",  ";Hadronic t p_{T} [GeV]", 50, 0., 500. )
histograms['t_had_y_true']        = TH1F( "t_had_y_true",   ";Hadronic t #eta", 25, -5., 5. )
histograms['t_had_phi_true']      = TH1F( "t_had_phi_true", ";Hadronic t #phi", 16, -3.2, 3.2 )
histograms['t_had_E_true']        = TH1F( "t_had_E_true",   ";Hadronic t E [GeV]", 50, 0., 500. )
histograms['t_had_m_true']        = TH1F( "t_had_m_true",   ";Hadronic t m [GeV]", 30, 0., 300.  )

histograms['W_lep_px_true']       = TH1F( "W_lep_px_true",   ";Leptonic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_lep_py_true']       = TH1F( "W_lep_py_true",   ";Leptonic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pz_true']       = TH1F( "W_lep_pz_true",   ";Leptonic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pt_true']       = TH1F( "W_lep_pt_true",   ";Leptonic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_lep_y_true']        = TH1F( "W_lep_y_true",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_true']      = TH1F( "W_lep_phi_true",  ";Leptonic W #phi", 16, -3.2, 3.2 )
histograms['W_lep_E_true']        = TH1F( "W_lep_E_true",    ";Leptonic W E [GeV]", 50, 0., 500. )
histograms['W_lep_m_true']        = TH1F( "W_lep_m_true",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_px_true']       = TH1F( "b_lep_px_true",   ";Leptonic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_lep_py_true']       = TH1F( "b_lep_py_true",   ";Leptonic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pz_true']       = TH1F( "b_lep_pz_true",   ";Leptonic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pt_true']       = TH1F( "b_lep_pt_true",   ";Leptonic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_lep_y_true']        = TH1F( "b_lep_y_true",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_true']      = TH1F( "b_lep_phi_true",  ";Leptonic b #phi", 16, -3.2, 3.2 )
histograms['b_lep_E_true']        = TH1F( "b_lep_E_true",    ";Leptonic b E [GeV]", 50, 0., 500. )
histograms['b_lep_m_true']        = TH1F( "b_lep_m_true",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_px_true']       = TH1F( "t_lep_px_true",   ";Leptonic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_lep_py_true']       = TH1F( "t_lep_py_true",   ";Leptonic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pz_true']       = TH1F( "t_lep_pz_true",   ";Leptonic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pt_true']       = TH1F( "t_lep_pt_true",   ";Leptonic t p_{T} [GeV]", 50, 0., 500. )
histograms['t_lep_y_true']        = TH1F( "t_lep_y_true",    ";Leptonic t #eta", 25, -5., 5. )
histograms['t_lep_phi_true']      = TH1F( "t_lep_phi_true",  ";Leptonic t #phi", 16, -3.2, 3.2 )
histograms['t_lep_E_true']        = TH1F( "t_lep_E_true",    ";Leptonic t E [GeV]", 50, 0., 500. )
histograms['t_lep_m_true']        = TH1F( "t_lep_m_true",    ";Leptonic t m [GeV]", 30, 0., 300. )

# Fitted
histograms['W_had_px_fitted']       = TH1F( "W_had_px_predicted",  ";Hadronic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_had_py_fitted']       = TH1F( "W_had_py_predicted",  ";Hadronic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_had_pz_fitted']       = TH1F( "W_had_pz_predicted",  ";Hadronic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_had_pt_fitted']       = TH1F( "W_had_pt_predicted",  ";Hadronic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_had_y_fitted']        = TH1F( "W_had_y_predicted",   ";Hadronic W #eta", 25, -5., 5. )
histograms['W_had_phi_fitted']      = TH1F( "W_had_phi_predicted", ";Hadronic W #phi", 16, -3.2, 3.2 )
histograms['W_had_E_fitted']        = TH1F( "W_had_E_predicted",   ";Hadronic W E [GeV]", 50, 0., 500. )
histograms['W_had_m_fitted']        = TH1F( "W_had_m_predicted",   ";Hadronic W m [GeV]", 30, 0., 300.  )

histograms['b_had_px_fitted']       = TH1F( "b_had_px_predicted",  ";Hadronic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_had_py_fitted']       = TH1F( "b_had_py_predicted",  ";Hadronic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_had_pz_fitted']       = TH1F( "b_had_pz_predicted",  ";Hadronic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_had_pt_fitted']       = TH1F( "b_had_pt_predicted",  ";Hadronic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_had_y_fitted']        = TH1F( "b_had_y_predicted",   ";Hadronic b #eta", 25, -5., 5. )
histograms['b_had_phi_fitted']      = TH1F( "b_had_phi_predicted", ";Hadronic b #phi", 16, -3.2, 3.2 )
histograms['b_had_E_fitted']        = TH1F( "b_had_E_predicted",   ";Hadronic b E [GeV]", 50, 0., 500. )
histograms['b_had_m_fitted']        = TH1F( "b_had_m_predicted",   ";Hadronic b m [GeV]", 30, 0., 300.  )

histograms['t_had_px_fitted']       = TH1F( "t_had_px_predicted",  ";Hadronic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_had_py_fitted']       = TH1F( "t_had_py_predicted",  ";Hadronic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_had_pz_fitted']       = TH1F( "t_had_pz_predicted",  ";Hadronic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_had_pt_fitted']       = TH1F( "t_had_pt_predicted",  ";Hadronic top p_{T} [GeV]", 50, 0., 500. )
histograms['t_had_y_fitted']        = TH1F( "t_had_y_predicted",   ";Hadronic top #eta", 25, -5., 5. )
histograms['t_had_phi_fitted']      = TH1F( "t_had_phi_predicted", ";Hadronic top #phi", 16, -3.2, 3.2 )
histograms['t_had_E_fitted']        = TH1F( "t_had_E_predicted",   ";Hadronic top E [GeV]", 50, 0., 500. )
histograms['t_had_m_fitted']        = TH1F( "t_had_m_predicted",   ";Hadronic top m [GeV]", 30, 0., 300.  )

histograms['W_lep_px_fitted']       = TH1F( "W_lep_px_predicted",   ";Leptonic W p_{x} [GeV]", 50, -1000., 1000. )
histograms['W_lep_py_fitted']       = TH1F( "W_lep_py_predicted",   ";Leptonic W p_{y} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pz_fitted']       = TH1F( "W_lep_pz_predicted",   ";Leptonic W p_{z} [GeV]", 50, -1000., 1000. )
histograms['W_lep_pt_fitted']       = TH1F( "W_lep_pt_predicted",   ";Leptonic W p_{T} [GeV]", 50, 0., 500. )
histograms['W_lep_y_fitted']        = TH1F( "W_lep_y_predicted",    ";Leptonic W #eta", 25, -5., 5. )
histograms['W_lep_phi_fitted']      = TH1F( "W_lep_phi_predicted",  ";Leptonic W #phi", 16, -3.2, 3.2 )
histograms['W_lep_E_fitted']        = TH1F( "W_lep_E_predicted",    ";Leptonic W E [GeV]", 50, 0., 500. )
histograms['W_lep_m_fitted']        = TH1F( "W_lep_m_predicted",    ";Leptonic W m [GeV]", 30, 0., 300. )

histograms['b_lep_px_fitted']       = TH1F( "b_lep_px_predicted",   ";Leptonic b p_{x} [GeV]", 50, -1000., 1000. )
histograms['b_lep_py_fitted']       = TH1F( "b_lep_py_predicted",   ";Leptonic b p_{y} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pz_fitted']       = TH1F( "b_lep_pz_predicted",   ";Leptonic b p_{z} [GeV]", 50, -1000., 1000. )
histograms['b_lep_pt_fitted']       = TH1F( "b_lep_pt_predicted",   ";Leptonic b p_{T} [GeV]", 50, 0., 500. )
histograms['b_lep_y_fitted']        = TH1F( "b_lep_y_predicted",    ";Leptonic b #eta", 25, -5., 5. )
histograms['b_lep_phi_fitted']      = TH1F( "b_lep_phi_predicted",  ";Leptonic b #phi", 16, -3.2, 3.2 )
histograms['b_lep_E_fitted']        = TH1F( "b_lep_E_predicted",    ";Leptonic b E [GeV]", 50, 0., 500. )
histograms['b_lep_m_fitted']        = TH1F( "b_lep_m_predicted",    ";Leptonic b m [GeV]", 30, 0., 300. )

histograms['t_lep_px_fitted']       = TH1F( "t_lep_px_predicted",   ";Leptonic t p_{x} [GeV]", 50, -1000., 1000. )
histograms['t_lep_py_fitted']       = TH1F( "t_lep_py_predicted",   ";Leptonic t p_{y} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pz_fitted']       = TH1F( "t_lep_pz_predicted",   ";Leptonic t p_{z} [GeV]", 50, -1000., 1000. )
histograms['t_lep_pt_fitted']       = TH1F( "t_lep_pt_predicted",   ";Leptonic top p_{T} [GeV]", 50, 0., 500. )
histograms['t_lep_y_fitted']        = TH1F( "t_lep_y_predicted",    ";Leptonic top #eta", 25, -5., 5. )
histograms['t_lep_phi_fitted']      = TH1F( "t_lep_phi_predicted",  ";Leptonic top #phi", 16, -3.2, 3.2 )
histograms['t_lep_E_fitted']        = TH1F( "t_lep_E_predicted",    ";Leptonic top E [GeV]", 50, 0., 500. )
histograms['t_lep_m_fitted']        = TH1F( "t_lep_m_predicted",    ";Leptonic top m [GeV]", 30, 0., 300. )

# 2D correlations
histograms['corr_t_had_pt']    = TH2F( "corr_t_had_pt",      ";True Hadronic top p_{T} [GeV];Predicted Hadronic top p_{T} [GeV]", 50, 0., 400., 50, 0., 400. )
histograms['corr_t_had_px']    = TH2F( "corr_t_had_px",      ";True Hadronic top p_{x} [GeV];Predicted Hadronic top p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_t_had_py']    = TH2F( "corr_t_had_py",      ";True Hadronic top p_{y} [GeV];Predicted Hadronic top p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_t_had_pz']    = TH2F( "corr_t_had_pz",      ";True Hadronic top p_{z} [GeV];Predicted Hadronic top p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_t_had_y']     = TH2F( "corr_t_had_y",       ";True Hadronic top y;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_had_phi']   = TH2F( "corr_t_had_phi",     ";True Hadronic top #phi;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_t_had_E']     = TH2F( "corr_t_had_E",       ";True Hadronic top E [GeV];Predicted Hadronic top E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_t_had_m']     = TH2F( "corr_t_had_m",       ";True Hadronic top m [GeV];Predicted Hadronic top m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_t_lep_pt']    = TH2F( "corr_t_lep_pt",     ";True Leptonic top p_{T} [GeV];Predicted Leptonic top p_{T} [GeV]", 50, 0., 400., 50, 0., 400. )
histograms['corr_t_lep_px']    = TH2F( "corr_t_lep_px",      ";True Leptonic top p_{x} [GeV];Predicted Leptonic top p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_t_lep_py']    = TH2F( "corr_t_lep_py",      ";True Leptonic top p_{y} [GeV];Predicted Leptonic top p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_t_lep_pz']    = TH2F( "corr_t_lep_pz",      ";True Leptonic top p_{z} [GeV];Predicted Leptonic top p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_t_lep_y']     = TH2F( "corr_t_lep_y",      ";True Leptonic top y;Predicted Leptonic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_t_lep_phi']   = TH2F( "corr_t_lep_phi",    ";True Leptonic top #phi;Predicted Leptonic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_t_lep_E']     = TH2F( "corr_t_lep_E",      ";True Leptonic top E [GeV];Predicted Leptonic top E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_t_lep_m']     = TH2F( "corr_t_lep_m",      ";True Leptonic top m [GeV];Predicted Leptonic top m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_b_lep_pt']    = TH2F( "corr_b_lep_pt",     ";True Leptonic bot p_{T} [GeV];Predicted Leptonic bot p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_b_lep_px']    = TH2F( "corr_b_lep_px",      ";True Leptonic bot p_{x} [GeV];Predicted Leptonic bot p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_b_lep_py']    = TH2F( "corr_b_lep_py",      ";True Leptonic bot p_{y} [GeV];Predicted Leptonic bot p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_b_lep_pz']    = TH2F( "corr_b_lep_pz",      ";True Leptonic bot p_{z} [GeV];Predicted Leptonic bot p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_b_lep_y']     = TH2F( "corr_b_lep_y",      ";True Leptonic bot y;Predicted Leptonic bot y", 25, -5., 5., 25, -5., 5. )
histograms['corr_b_lep_phi']   = TH2F( "corr_b_lep_phi",    ";True Leptonic bot #phi;Predicted Leptonic bot #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_b_lep_E']     = TH2F( "corr_b_lep_E",      ";True Leptonic bot E [GeV];Predicted Leptonic bot E [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_b_lep_m']     = TH2F( "corr_b_lep_m",      ";True Leptonic bot m [GeV];Predicted Leptonic bot m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_b_had_pt']    = TH2F( "corr_b_had_pt",      ";True Hadronic bot p_{T} [GeV];Predicted Hadronic bot p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_b_had_px']    = TH2F( "corr_b_had_px",      ";True Hadronic bot p_{x} [GeV];Predicted Hadronic bot p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_b_had_py']    = TH2F( "corr_b_had_py",      ";True Hadronic bot p_{y} [GeV];Predicted Hadronic bot p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_b_had_pz']    = TH2F( "corr_b_had_pz",      ";True Hadronic bot p_{z} [GeV];Predicted Hadronic bot p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_b_had_y']     = TH2F( "corr_b_had_y",       ";True Hadronic bot y;Predicted Hadronic bot y", 25, -5., 5., 25, -5., 5. )
histograms['corr_b_had_phi']   = TH2F( "corr_b_had_phi",     ";True Hadronic bot #phi;Predicted Hadronic bot #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_b_had_E']     = TH2F( "corr_b_had_E",       ";True Hadronic bot E [GeV];Predicted Hadronic bot E [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_b_had_m']     = TH2F( "corr_b_had_m",       ";True Hadronic bot m [GeV];Predicted Hadronic bot m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_W_had_pt']    = TH2F( "corr_W_had_pt",      ";True Hadronic W p_{T} [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_W_had_px']    = TH2F( "corr_W_had_px",      ";True Hadronic W p_{x} [GeV];Predicted Hadronic W p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_W_had_py']    = TH2F( "corr_W_had_py",      ";True Hadronic W p_{y} [GeV];Predicted Hadronic W p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_W_had_pz']    = TH2F( "corr_W_had_pz",      ";True Hadronic W p_{z} [GeV];Predicted Hadronic W p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_W_had_y']     = TH2F( "corr_W_had_y",       ";True Hadronic W y;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
histograms['corr_W_had_phi']   = TH2F( "corr_W_had_phi",     ";True Hadronic W #phi;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_W_had_E']     = TH2F( "corr_W_had_E",       ";True Hadronic W E [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )
histograms['corr_W_had_m']     = TH2F( "corr_W_had_m",       ";True Hadronic W m [GeV];Predicted Hadronic top m [GeV]", 25, 170., 175., 20, 150., 250. )

histograms['corr_W_lep_pt']    = TH2F( "corr_W_lep_pt",      ";True Leptonic W p_{T} [GeV];Predicted Leptonic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
histograms['corr_W_lep_px']    = TH2F( "corr_W_lep_px",      ";True Leptonic W p_{x} [GeV];Predicted Leptonic W p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_W_lep_py']    = TH2F( "corr_W_lep_py",      ";True Leptonic W p_{y} [GeV];Predicted Leptonic W p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
histograms['corr_W_lep_pz']    = TH2F( "corr_W_lep_pz",      ";True Leptonic W p_{z} [GeV];Predicted Leptonic W p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
histograms['corr_W_lep_y']     = TH2F( "corr_W_lep_y",       ";True Leptonic W y;Predicted Leptonic W y", 25, -5., 5., 25, -5., 5. )
histograms['corr_W_lep_phi']   = TH2F( "corr_W_lep_phi",     ";True Leptonic W #phi;Predicted Leptonic W #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
histograms['corr_W_lep_E']     = TH2F( "corr_W_lep_E",       ";True Leptonic W E [GeV];Predicted Leptonic W E [GeV]", 50, 70., 400., 50, 70., 400. )
histograms['corr_W_lep_m']     = TH2F( "corr_W_lep_m",       ";True Leptonic W m [GeV];Predicted Leptonic W m [GeV]", 25, 170., 175., 20, 150., 250. )
################################################################################

# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0.

# Define sums of squares to be used for calculating sample chi-squared/NDF
# One sum for each variable to be augmented in each event
W_had_phi_sum = 0
W_had_rapidity_sum = 0
W_had_pt_sum = 0

W_lep_phi_sum = 0
W_lep_rapidity_sum = 0
W_lep_pt_sum = 0

b_had_phi_sum = 0
b_had_rapidity_sum = 0
b_had_pt_sum = 0

b_lep_phi_sum = 0
b_lep_rapidity_sum = 0
b_lep_pt_sum = 0


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

    # chi22 += ( W_had_true.E() - W_had_fitted.E() )**2 / ( fwhm['W_had_E']**2 )
    # chi22 += ( W_lep_true.E() - W_lep_fitted.E() )**2 / ( fwhm['W_lep_E']**2 )
    # chi22 += ( b_had_true.E() - b_had_fitted.E() )**2 / ( fwhm['b_had_E']**2 )
    # chi22 += ( b_lep_true.E() - b_lep_fitted.E() )**2 / ( fwhm['b_lep_E']**2 )

    # print("chi^2: {}, reduced chi^2: {}".format(chi22, chi22/12.0))
    
    if chi22 < 1.5 and chi22 > 0.5:
      n_good += 1.
      histograms['W_had_px_true'].Fill(  W_had_true.Px(),  w)
      histograms['W_had_py_true'].Fill(  W_had_true.Py(),  w )
      histograms['W_had_pz_true'].Fill(  W_had_true.Pz(),  w )
      histograms['W_had_pt_true'].Fill(  W_had_true.Pt(),  w )
      histograms['W_had_y_true'].Fill(   W_had_true.Rapidity(), w )
      histograms['W_had_phi_true'].Fill( W_had_true.Phi(), w )
      histograms['W_had_E_true'].Fill(   W_had_true.E(),   w )
      histograms['W_had_m_true'].Fill(   W_had_true.M(),   w )

      histograms['b_had_px_true'].Fill(  b_had_true.Px(),  w )
      histograms['b_had_py_true'].Fill(  b_had_true.Py(),  w )
      histograms['b_had_pz_true'].Fill(  b_had_true.Pz(),  w )
      histograms['b_had_pt_true'].Fill(  b_had_true.Pt(),  w )
      histograms['b_had_y_true'].Fill(   b_had_true.Rapidity(), w )
      histograms['b_had_phi_true'].Fill( b_had_true.Phi(), w )
      histograms['b_had_E_true'].Fill(   b_had_true.E(),   w )
      histograms['b_had_m_true'].Fill(   b_had_true.M(),   w )

      histograms['t_had_px_true'].Fill(  t_had_true.Px(),  w )
      histograms['t_had_py_true'].Fill(  t_had_true.Py(),  w )
      histograms['t_had_pz_true'].Fill(  t_had_true.Pz(),  w )
      histograms['t_had_pt_true'].Fill(  t_had_true.Pt(),  w )
      histograms['t_had_y_true'].Fill(   t_had_true.Rapidity(), w )
      histograms['t_had_phi_true'].Fill( t_had_true.Phi(), w )
      histograms['t_had_E_true'].Fill(   t_had_true.E(),   w )
      histograms['t_had_m_true'].Fill(   t_had_true.M(),   w )

      histograms['W_lep_px_true'].Fill(  W_lep_true.Px(),  w )
      histograms['W_lep_py_true'].Fill(  W_lep_true.Py(),  w )
      histograms['W_lep_pz_true'].Fill(  W_lep_true.Pz(),  w )
      histograms['W_lep_pt_true'].Fill(  W_lep_true.Pt(),  w )
      histograms['W_lep_y_true'].Fill(   W_lep_true.Rapidity(), w )
      histograms['W_lep_phi_true'].Fill( W_lep_true.Phi(), w )
      histograms['W_lep_E_true'].Fill(   W_lep_true.E(),   w )
      histograms['W_lep_m_true'].Fill(   W_lep_true.M(),   w )

      histograms['b_lep_px_true'].Fill(  b_lep_true.Px(),  w )
      histograms['b_lep_py_true'].Fill(  b_lep_true.Py(),  w )
      histograms['b_lep_pz_true'].Fill(  b_lep_true.Pz(),  w )
      histograms['b_lep_pt_true'].Fill(  b_lep_true.Pt(),  w )
      histograms['b_lep_y_true'].Fill(   b_lep_true.Rapidity(), w )
      histograms['b_lep_phi_true'].Fill( b_lep_true.Phi(), w )
      histograms['b_lep_E_true'].Fill(   b_lep_true.E(),   w )
      histograms['b_lep_m_true'].Fill(   b_lep_true.M(),   w )

      histograms['t_lep_px_true'].Fill(  t_lep_true.Px(),  w )
      histograms['t_lep_py_true'].Fill(  t_lep_true.Py(),  w )
      histograms['t_lep_pz_true'].Fill(  t_lep_true.Pz(),  w )
      histograms['t_lep_pt_true'].Fill(  t_lep_true.Pt(),  w )
      histograms['t_lep_y_true'].Fill(   t_lep_true.Rapidity(), w )
      histograms['t_lep_phi_true'].Fill( t_lep_true.Phi(), w )
      histograms['t_lep_E_true'].Fill(   t_lep_true.E(),   w )
      histograms['t_lep_m_true'].Fill(   t_lep_true.M(),   w )

      # Fitted
      histograms['W_had_px_fitted'].Fill(  W_had_fitted.Px(),  w )
      histograms['W_had_py_fitted'].Fill(  W_had_fitted.Py(),  w )
      histograms['W_had_pz_fitted'].Fill(  W_had_fitted.Pz(),  w )
      histograms['W_had_pt_fitted'].Fill(  W_had_fitted.Pt(),  w )
      histograms['W_had_y_fitted'].Fill(   W_had_fitted.Rapidity(), w )
      histograms['W_had_phi_fitted'].Fill( W_had_fitted.Phi(), w )
      histograms['W_had_E_fitted'].Fill(   W_had_fitted.E(),   w )
      histograms['W_had_m_fitted'].Fill(   W_had_fitted.M(),   w )

      histograms['b_had_px_fitted'].Fill(  b_had_fitted.Px(),  w )
      histograms['b_had_py_fitted'].Fill(  b_had_fitted.Py(),  w )
      histograms['b_had_pz_fitted'].Fill(  b_had_fitted.Pz(),  w )
      histograms['b_had_pt_fitted'].Fill(  b_had_fitted.Pt(),  w )
      histograms['b_had_y_fitted'].Fill(   b_had_fitted.Rapidity(), w )
      histograms['b_had_phi_fitted'].Fill( b_had_fitted.Phi(), w )
      histograms['b_had_E_fitted'].Fill(   b_had_fitted.E(),   w )
      histograms['b_had_m_fitted'].Fill(   b_had_fitted.M(),   w )

      histograms['t_had_px_fitted'].Fill(  t_had_fitted.Px(),  w )
      histograms['t_had_py_fitted'].Fill(  t_had_fitted.Py(),  w )
      histograms['t_had_pz_fitted'].Fill(  t_had_fitted.Pz(),  w )
      histograms['t_had_pt_fitted'].Fill(  t_had_fitted.Pt(),  w )
      histograms['t_had_y_fitted'].Fill(   t_had_fitted.Rapidity(), w )
      histograms['t_had_phi_fitted'].Fill( t_had_fitted.Phi(), w )
      histograms['t_had_E_fitted'].Fill(   t_had_fitted.E(),   w )
      histograms['t_had_m_fitted'].Fill(   t_had_fitted.M(),   w )

      histograms['W_lep_px_fitted'].Fill(  W_lep_fitted.Px(),  w )
      histograms['W_lep_py_fitted'].Fill(  W_lep_fitted.Py(),  w )
      histograms['W_lep_pz_fitted'].Fill(  W_lep_fitted.Pz(),  w )
      histograms['W_lep_pt_fitted'].Fill(  W_lep_fitted.Pt(),  w )
      histograms['W_lep_y_fitted'].Fill(   W_lep_fitted.Rapidity(), w )
      histograms['W_lep_phi_fitted'].Fill( W_lep_fitted.Phi(), w )
      histograms['W_lep_E_fitted'].Fill(   W_lep_fitted.E(),   w )
      histograms['W_lep_m_fitted'].Fill(   W_lep_fitted.M(),   w )

      histograms['b_lep_px_fitted'].Fill(  b_lep_fitted.Px(),  w )
      histograms['b_lep_py_fitted'].Fill(  b_lep_fitted.Py(),  w )
      histograms['b_lep_pz_fitted'].Fill(  b_lep_fitted.Pz(),  w )
      histograms['b_lep_pt_fitted'].Fill(  b_lep_fitted.Pt(),  w )
      histograms['b_lep_y_fitted'].Fill(   b_lep_fitted.Rapidity(), w )
      histograms['b_lep_phi_fitted'].Fill( b_lep_fitted.Phi(), w )
      histograms['b_lep_E_fitted'].Fill(   b_lep_fitted.E(),   w )
      histograms['b_lep_m_fitted'].Fill(   b_lep_fitted.M(),   w )

      histograms['t_lep_px_fitted'].Fill(  t_lep_fitted.Px(),  w )
      histograms['t_lep_py_fitted'].Fill(  t_lep_fitted.Py(),  w )
      histograms['t_lep_pz_fitted'].Fill(  t_lep_fitted.Pz(),  w )
      histograms['t_lep_pt_fitted'].Fill(  t_lep_fitted.Pt(),  w )
      histograms['t_lep_y_fitted'].Fill(   t_lep_fitted.Rapidity(), w )
      histograms['t_lep_phi_fitted'].Fill( t_lep_fitted.Phi(), w )
      histograms['t_lep_E_fitted'].Fill(   t_lep_fitted.E(),   w )
      histograms['t_lep_m_fitted'].Fill(   t_lep_fitted.M(),   w )

      # 2D correlations
      histograms['corr_t_had_pt'].Fill(   t_had_true.Pt(),       t_had_fitted.Pt(),  w )
      histograms['corr_t_had_px'].Fill(   t_had_true.Px(),       t_had_fitted.Px(),  w )
      histograms['corr_t_had_py'].Fill(   t_had_true.Py(),       t_had_fitted.Py(),  w )
      histograms['corr_t_had_pz'].Fill(   t_had_true.Pz(),       t_had_fitted.Pz(),  w )
      histograms['corr_t_had_y'].Fill(    t_had_true.Rapidity(), t_had_fitted.Rapidity(), w )
      histograms['corr_t_had_phi'].Fill(  t_had_true.Phi(),      t_had_fitted.Phi(), w )
      histograms['corr_t_had_E'].Fill(    t_had_true.E(),        t_had_fitted.E(),   w )
      histograms['corr_t_had_m'].Fill(    t_had_true.M(),        t_had_fitted.M(),   w )

      histograms['corr_t_lep_pt'].Fill(  t_lep_true.Pt(),       t_lep_fitted.Pt(),  w )
      histograms['corr_t_lep_px'].Fill(  t_lep_true.Px(),       t_lep_fitted.Px(),  w )
      histograms['corr_t_lep_py'].Fill(  t_lep_true.Py(),       t_lep_fitted.Py(),  w )
      histograms['corr_t_lep_pz'].Fill(  t_lep_true.Pz(),       t_lep_fitted.Pz(),  w )
      histograms['corr_t_lep_y'].Fill(   t_lep_true.Rapidity(), t_lep_fitted.Rapidity(), w )
      histograms['corr_t_lep_phi'].Fill( t_lep_true.Phi(),      t_lep_fitted.Phi(), w )
      histograms['corr_t_lep_E'].Fill(   t_lep_true.E(),        t_lep_fitted.E(),   w )
      histograms['corr_t_lep_m'].Fill(   t_lep_true.M(),        t_lep_fitted.M(),   w )

      histograms['corr_W_lep_pt'].Fill(  W_lep_true.Pt(),       W_lep_fitted.Pt(),  w )
      histograms['corr_W_lep_px'].Fill(  W_lep_true.Px(),       W_lep_fitted.Px(),  w )
      histograms['corr_W_lep_py'].Fill(  W_lep_true.Py(),       W_lep_fitted.Py(),  w )
      histograms['corr_W_lep_pz'].Fill(  W_lep_true.Pz(),       W_lep_fitted.Pz(),  w )
      histograms['corr_W_lep_y'].Fill(   W_lep_true.Rapidity(), W_lep_fitted.Rapidity(), w )
      histograms['corr_W_lep_phi'].Fill( W_lep_true.Phi(),      W_lep_fitted.Phi(), w )
      histograms['corr_W_lep_E'].Fill(   W_lep_true.E(),        W_lep_fitted.E(),   w )
      histograms['corr_W_lep_m'].Fill(   W_lep_true.M(),        W_lep_fitted.M(),   w )

      histograms['corr_W_had_pt'].Fill(  W_had_true.Pt(),       W_had_fitted.Pt(),  w )
      histograms['corr_W_had_px'].Fill(  W_had_true.Px(),       W_had_fitted.Px(),  w )
      histograms['corr_W_had_py'].Fill(  W_had_true.Py(),       W_had_fitted.Py(),  w )
      histograms['corr_W_had_pz'].Fill(  W_had_true.Pz(),       W_had_fitted.Pz(),  w )
      histograms['corr_W_had_y'].Fill(   W_had_true.Rapidity(), W_had_fitted.Rapidity(), w )
      histograms['corr_W_had_phi'].Fill( W_had_true.Phi(),      W_had_fitted.Phi(), w )
      histograms['corr_W_had_E'].Fill(   W_had_true.E(),        W_had_fitted.E(),   w )
      histograms['corr_W_had_m'].Fill(   W_had_true.M(),        W_had_fitted.M(),   w )

      histograms['corr_b_lep_pt'].Fill(  b_lep_true.Pt(),       b_lep_fitted.Pt(),  w )
      histograms['corr_b_lep_px'].Fill(  b_lep_true.Px(),       b_lep_fitted.Px(),  w )
      histograms['corr_b_lep_py'].Fill(  b_lep_true.Py(),       b_lep_fitted.Py(),  w )
      histograms['corr_b_lep_pz'].Fill(  b_lep_true.Pz(),       b_lep_fitted.Pz(),  w )
      histograms['corr_b_lep_y'].Fill(   b_lep_true.Rapidity(), b_lep_fitted.Rapidity(), w )
      histograms['corr_b_lep_phi'].Fill( b_lep_true.Phi(),      b_lep_fitted.Phi(), w )
      histograms['corr_b_lep_E'].Fill(   b_lep_true.E(),        b_lep_fitted.E(),   w )
      histograms['corr_b_lep_m'].Fill(   b_lep_true.M(),        b_lep_fitted.M(),   w )

      histograms['corr_b_had_pt'].Fill(  b_had_true.Pt(),       b_had_fitted.Pt(),  w )
      histograms['corr_b_had_px'].Fill(  b_had_true.Px(),       b_had_fitted.Px(),  w )
      histograms['corr_b_had_py'].Fill(  b_had_true.Py(),       b_had_fitted.Py(),  w )
      histograms['corr_b_had_pz'].Fill(  b_had_true.Pz(),       b_had_fitted.Pz(),  w )
      histograms['corr_b_had_y'].Fill(   b_had_true.Rapidity(), b_had_fitted.Rapidity(), w )
      histograms['corr_b_had_phi'].Fill( b_had_true.Phi(),      b_had_fitted.Phi(), w )
      histograms['corr_b_had_E'].Fill(   b_had_true.E(),        b_had_fitted.E(),   w )
      histograms['corr_b_had_m'].Fill(   b_had_true.M(),        b_had_fitted.M(),   w )

# Normalize sums of squares by standard deviations and number of events
W_had_phi_chi2NDF = W_had_phi_sum / n_events / ( sigma['W_had_phi']**2 )
W_had_rapidity_chi2NDF = W_had_rapidity_sum / n_events / ( sigma['W_had_y']**2 )
W_had_pt_chi2NDF = W_had_pt_sum / n_events / ( sigma['W_had_pt']**2 )

W_lep_phi_chi2NDF = W_lep_phi_sum / n_events / ( sigma['W_lep_phi']**2 )
W_lep_rapidity_chi2NDF = W_lep_rapidity_sum / n_events / ( sigma['W_lep_y']**2 )
W_lep_pt_chi2NDF = W_lep_pt_sum / n_events / ( sigma['W_lep_pt']**2 )

b_had_phi_chi2NDF = b_had_phi_sum / n_events / ( sigma['b_had_phi']**2 )
b_had_rapidity_chi2NDF = b_had_rapidity_sum / n_events / ( sigma['b_had_y']**2 )
b_had_pt_chi2NDF = b_had_pt_sum / n_events / ( sigma['b_had_pt']**2 )

b_lep_phi_chi2NDF = b_lep_phi_sum / n_events / ( sigma['b_lep_phi']**2 )
b_lep_rapidity_chi2NDF = b_lep_rapidity_sum / n_events / ( sigma['b_lep_y']**2 )
b_lep_pt_chi2NDF = b_lep_pt_sum / n_events / ( sigma['b_lep_pt']**2 )

# Print Chi-Squareds/NDF.
print('')
print("W_had_phi_chi2NDF: {0}".format(W_had_phi_chi2NDF))
print("W_had_rapidity_chi2NDF: {0}".format(W_had_rapidity_chi2NDF))
print("W_had_pt_chi2NDF: {0}".format(W_had_pt_chi2NDF))
print('')
print("W_lep_phi_chi2NDF: {0}".format(W_lep_phi_chi2NDF))
print("W_lep_rapidity_chi2NDF: {0}".format(W_lep_rapidity_chi2NDF))
print("W_lep_pt_chi2NDF: {0}".format(W_lep_pt_chi2NDF))
print('')
print("b_had_phi_chi2NDF: {0}".format(b_had_phi_chi2NDF))
print("b_had_rapidity_chi2NDF: {0}".format(b_had_rapidity_chi2NDF))
print("b_had_pt_chi2NDF: {0}".format(b_had_pt_chi2NDF))
print('')
print("b_lep_phi_chi2NDF: {0}".format(b_lep_phi_chi2NDF))
print("b_lep_rapidity_chi2NDF: {0}".format(b_lep_rapidity_chi2NDF))
print("b_lep_pt_chi2NDF: {0}".format(b_lep_pt_chi2NDF))
print('')

try:
    os.mkdir('{}/{}'.format(training_dir, outputdir))
except Exception as e:
    print("Overwriting existing files")

print("good events: {}, {}%".format(n_good, n_good/n_events*100))

from AngryTops.Plotting.PlottingHelper import MakeCanvas

for obs in attributes:
    # Load the histograms
    hname_true = "%s_true" % (obs)
    hname_fitted = "%s_fitted" % (obs)

    # True and fitted leaf
    h_true = histograms[hname_true]
    h_fitted = histograms[hname_fitted]
    if h_true == None:
        print ("ERROR: invalid histogram for", obs)

    # Axis titles
    xtitle = h_true.GetXaxis().GetTitle()
    ytitle = h_true.GetYaxis().SetTitle("A.U.")
    if h_true.Class() == TH2F.Class():
        h_true = h_true.ProfileX("pfx")
        h_true.GetYaxis().SetTitle( ytitle )
    else:
        Normalize(h_true)
        Normalize(h_fitted)

    # Set Style
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_true.Draw("h")
    h_fitted.Draw("h same")
    hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum() ] )
    h_fitted.SetMaximum( hmax )
    h_true.SetMaximum( hmax )
    h_fitted.SetMinimum( 0. )
    h_true.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, "MG5+Py8", "f" )
    leg.AddEntry( h_fitted, "Predicted", "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    gPad.RedrawAxis()
    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

        gPad.RedrawAxis()

    pad1.cd()

    yrange = [0.4, 1.6]
    frame, tot_unc, ratio = DrawRatio(h_true, h_fitted, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/{1}/{2}.png".format(training_dir, outputdir, obs))
    pad0.Close()
    pad1.Close()
    c.Close()

for hist_name in corr_2d:

    # True and fitted leaf
    hist = histograms[hist_name]
    if hist == None:
        print ("ERROR: invalid histogram for", hist_name)

    #Normalize(hist)

    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    #pad0.SetTopMargin( 0.14 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()
    pad0.cd()

    hist.Draw("colz")

    corr = hist.GetCorrelationFactor()
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

    gPad.RedrawAxis()

    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

    c.cd()

    c.SaveAs("{0}/{1}/{2}.png".format(training_dir, outputdir, hist_name))
    pad0.Close()
    c.Close()


