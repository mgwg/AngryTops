# Calculates chi-squared for key variables in truth vs. fitted comparison. 
#  Plots distribution of chi-squareds where one chi-squared is calculated for each event.
#  Plots distribution of p-values using the chi-squared variables, 
#   assuming that they follow a chi-squared distribution. 
#!/usr/bin/env python
import os, sys, time
import argparse
from AngryTops.features import *
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
from AngryTops.Plotting.identification_helper import * 
from scipy.stats import chi2

################################################################################
# CONSTANTS
m_t = 172.5
m_W = 80.4
m_b = 4.95

# First directory is the training directory where the truth and fitted data for the sample of interest is saved.
if len(sys.argv) > 1: training_dir = sys.argv[1]
infilename = "{}/fitted.root".format(training_dir)
print("Training directory filename: {0}".format(infilename))
# Second directory is the directory containing the difference plots whose FWHM 
#  will be used to calculate the Chi-Squareds.
FWHM_dir = sys.argv[2]
# Output directory
outputdir = "/img_chi/"
if len(sys.argv) > 3:
    outputdir = "/img_chi{}/".format(sys.argv[3])

# Read in histograms from the FWHM directory.
histsFilename = "{}/histograms.root".format(FWHM_dir)
histsFile = TFile.Open(histsFilename)
print("FWHM directory filename: {0}\n".format(histsFilename))
print(histsFilename)

################################################################################
# HELPER FUNCTIONS
def PrintOut( p4_true, p4_fitted, label ):
  print("%s :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) \
        :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
        (label, p4_true.Pt(), p4_true.Rapidity(), p4_true.Phi(), p4_true.E(), p4_true.M(),\
        p4_fitted.Pt(), p4_fitted.Rapidity(), p4_fitted.Phi(), p4_fitted.E(), p4_fitted.M()
        ))

################################################################################
# Read in input file from training directory, contains truth and fitted data
infile = TFile.Open( infilename )
tree   = infile.Get( "nominal")

################################################################################
# Draw Differences and resonances
fwhm = {}
sigma = {}
for obs in attributes:
    # Use only the difference plots from the FWHM folder.
    hist_name = "diff_{0}".format(obs)
    # True and fitted leaf
    hist = histsFile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)

    #Normalize(hist)
    if hist.Class() == TH2F.Class():
        hist = hist.ProfileX("hist_pfx")

    # Extract FWHM using helper function.
    fwhm_single, sigma_single = getFwhm( hist )
    fwhm[obs] = fwhm_single
    sigma[obs] = sigma_single

################################################################################
histograms = {}

# Distribution of Chi-Squareds
histograms['chi_squared_all'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 50.)
histograms['chi_squared_all'].SetTitle("#chi^{2} of all events; #chi^{2}, Unitless; A.U.")
histograms['chi_squared_all_NDF'] = TH1F("#chi^{2}/NDF",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_all_NDF'].SetTitle("#chi^{2}/NDF of all events; #chi^{2}, Unitless; A.U.")

# Distribution of p-values
histograms['p-values'] = TH1F("p-values",  ";Unitless", 100, 0., 1.)
histograms['p-values'].SetTitle("p-value distribution of #chi^{2} statistics; p-values, Unitless; A.U.")
histograms['p-values_log'] = TH1F("p-values",  ";Unitless", 100, 0., 1.)
histograms['p-values_log'].SetTitle("p-value distribution of #chi^{2} statistics; p-values, Unitless; A.U.")

histograms['chi_squared_had_W_phi'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_W_phi'].SetTitle("#chi^{2} of had W #phi; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_had_W_eta'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_W_eta'].SetTitle("#chi^{2} of had W #eta; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_had_W_pT'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_W_pT'].SetTitle("#chi^{2} of had W p_{T}; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_W_phi'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_W_phi'].SetTitle("#chi^{2} of lep W #phi; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_W_eta'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_W_eta'].SetTitle("#chi^{2} of lep W #eta; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_W_pT'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_W_pT'].SetTitle("#chi^{2} of lep W p_{T}; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_had_b_phi'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_b_phi'].SetTitle("#chi^{2} of had b #phi; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_had_b_eta'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_b_eta'].SetTitle("#chi^{2} of had b #eta; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_had_b_pT'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_had_b_pT'].SetTitle("#chi^{2} of had b p_{T}; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_b_phi'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_b_phi'].SetTitle("#chi^{2} of of lep b #phi; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_b_eta'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_b_eta'].SetTitle("#chi^{2} of lep b #eta; #chi^{2}, Unitless; A.U.")

histograms['chi_squared_lep_b_pT'] = TH1F("#chi^{2}",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_lep_b_pT'].SetTitle("#chi^{2} of lep b p_{T}; #chi^{2}, Unitless; A.U.")

################################################################################

# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)

# Extract and print relevant FWHMs
W_had_phi_FWHM = fwhm['W_had_phi']
W_had_rapidity_FWHM = fwhm['W_had_y']
W_had_pt_FWHM = fwhm['W_had_pt']

W_lep_phi_FWHM = fwhm['W_lep_phi']
W_lep_rapidity_FWHM = fwhm['W_lep_y']
W_lep_pt_FWHM = fwhm['W_lep_pt']

b_had_phi_FWHM = fwhm['b_had_phi']
b_had_rapidity_FWHM = fwhm['b_had_y']
b_had_pt_FWHM = fwhm['b_had_pt']

b_lep_phi_FWHM = fwhm['b_lep_phi']
b_lep_rapidity_FWHM = fwhm['b_lep_y']
b_lep_pt_FWHM = fwhm['b_lep_pt']

print("\nFWHMs:")

print("\nW_had_phi_FWHM: {0}".format(W_had_phi_FWHM))
print("W_had_rapidity_FWHM: {0}".format(W_had_rapidity_FWHM))
print("W_had_pt_FWHM: {0}\n".format(W_had_pt_FWHM))

print("W_lep_phi_FWHM: {0}".format(W_lep_phi_FWHM))
print("W_lep_rapidity_FWHM: {0}".format(W_lep_rapidity_FWHM))
print("W_lep_pt_FWHM: {0}\n".format(W_lep_pt_FWHM))

print("b_had_phi_FWHM: {0}".format(b_had_phi_FWHM))
print("b_had_rapidity_FWHM: {0}".format(b_had_rapidity_FWHM))
print("b_had_pt_FWHM: {0}\n".format(b_had_pt_FWHM))

print("b_lep_phi_FWHM: {0}".format(b_lep_phi_FWHM))
print("b_lep_rapidity_FWHM: {0}".format(b_lep_rapidity_FWHM))
print("b_lep_pt_FWHM: {0}\n".format(b_lep_pt_FWHM))

# find sigmas
W_had_phi_sigma = sigma['W_had_phi']
W_had_rapidity_sigma = sigma['W_had_y']
W_had_pt_sigma = sigma['W_had_pt']

W_lep_phi_sigma = sigma['W_lep_phi']
W_lep_rapidity_sigma = sigma['W_lep_y']
W_lep_pt_sigma = sigma['W_lep_pt']

b_had_phi_sigma = sigma['b_had_phi']
b_had_rapidity_sigma = sigma['b_had_y']
b_had_pt_sigma = sigma['b_had_pt']

b_lep_phi_sigma = sigma['b_lep_phi']
b_lep_rapidity_sigma = sigma['b_lep_y']
b_lep_pt_sigma = sigma['b_lep_pt']

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

    chi22 += W_had_phi_diff / ( W_had_phi_sigma**2 )
    chi22 += W_had_rapidity_diff / ( W_had_rapidity_sigma**2 )
    chi22 += W_had_pt_diff / ( W_had_pt_sigma**2 )

    chi22 += W_lep_phi_diff / ( W_lep_phi_sigma**2 )
    chi22 += W_lep_rapidity_diff / ( W_lep_rapidity_sigma**2 )
    chi22 += W_lep_pt_diff / ( W_lep_pt_sigma**2 )

    chi22 += b_had_phi_diff / ( b_had_phi_sigma**2 )
    chi22 += b_had_rapidity_diff / ( b_had_rapidity_sigma**2 )
    chi22 += b_had_pt_diff / ( b_had_pt_sigma**2 )

    chi22 += b_lep_phi_diff / ( b_lep_phi_sigma**2 )
    chi22 += b_lep_rapidity_diff / ( b_lep_rapidity_sigma**2 )
    chi22 += b_lep_pt_diff / ( b_lep_pt_sigma**2 )

    # Calculate chi-squared/NDF
    chi22NDF = chi22 / 12.0

    # Calculate a p-value assuming a chi-squared distribution with 12 degrees of freedom.
    #  Use the survival function defined as 1-CDF.
    p_value = chi2.sf(chi22, 12)

    # Populate the histograms:
    histograms['chi_squared_all'].Fill(chi22)
    histograms['chi_squared_all_NDF'].Fill(chi22NDF)
    histograms['p-values'].Fill(p_value)
    histograms['p-values_log'].Fill(p_value)

    histograms['chi_squared_had_W_phi'].Fill(W_had_phi_diff / ( W_had_phi_sigma**2 ))
    histograms['chi_squared_had_W_eta'].Fill(W_had_rapidity_diff / ( W_had_rapidity_sigma**2 ))
    histograms['chi_squared_had_W_pT'].Fill(W_had_pt_diff / ( W_had_pt_sigma**2 ))
    histograms['chi_squared_lep_W_phi'].Fill(W_lep_phi_diff / ( W_lep_phi_sigma**2 ))
    histograms['chi_squared_lep_W_eta'].Fill( W_lep_rapidity_diff / ( W_lep_rapidity_sigma**2 ))
    histograms['chi_squared_lep_W_pT'].Fill(W_lep_pt_diff / ( W_lep_pt_sigma**2 )) 
    histograms['chi_squared_had_b_phi'].Fill(b_had_phi_diff / ( b_had_phi_sigma**2 )) 
    histograms['chi_squared_had_b_eta'].Fill(b_had_rapidity_diff / ( b_had_rapidity_sigma**2 )) 
    histograms['chi_squared_had_b_pT'].Fill(b_had_pt_diff / ( b_had_pt_sigma**2 )) 
    histograms['chi_squared_lep_b_phi'].Fill(b_lep_phi_diff / ( b_lep_phi_sigma**2 )) 
    histograms['chi_squared_lep_b_eta'].Fill(b_lep_rapidity_diff / ( b_lep_rapidity_sigma**2 ))
    histograms['chi_squared_lep_b_pT'].Fill(b_lep_pt_diff / ( b_lep_pt_sigma**2 ))

# Normalize sums of squares by standard deviations and number of events
W_had_phi_chi2NDF = W_had_phi_sum / n_events / ( W_had_phi_sigma**2 )
W_had_rapidity_chi2NDF = W_had_rapidity_sum / n_events / ( W_had_rapidity_sigma**2 )
W_had_pt_chi2NDF = W_had_pt_sum / n_events / ( W_had_pt_sigma**2 )

W_lep_phi_chi2NDF = W_lep_phi_sum / n_events / ( W_lep_phi_sigma**2 )
W_lep_rapidity_chi2NDF = W_lep_rapidity_sum / n_events / ( W_lep_rapidity_sigma**2 )
W_lep_pt_chi2NDF = W_lep_pt_sum / n_events / ( W_lep_pt_sigma**2 )

b_had_phi_chi2NDF = b_had_phi_sum / n_events / ( b_had_phi_sigma**2 )
b_had_rapidity_chi2NDF = b_had_rapidity_sum / n_events / ( b_had_rapidity_sigma**2 )
b_had_pt_chi2NDF = b_had_pt_sum / n_events / ( b_had_pt_sigma**2 )

b_lep_phi_chi2NDF = b_lep_phi_sum / n_events / ( b_lep_phi_sigma**2 )
b_lep_rapidity_chi2NDF = b_lep_rapidity_sum / n_events / ( b_lep_rapidity_sigma**2 )
b_lep_pt_chi2NDF = b_lep_pt_sum / n_events / ( b_lep_pt_sigma**2 )

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
print("b_lep_pt_chi2NDF: {0}\n".format(b_lep_pt_chi2NDF))

try:
    os.mkdir('{}/{}'.format(training_dir, outputdir))
except Exception as e:
    print("Overwriting existing files")

# Plot histograms inside outputdir, a subdir of training_dir
for key in histograms:
    plot_hists(key, histograms[key], training_dir+outputdir)
