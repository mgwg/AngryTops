#!/usr/bin/env python
import os, sys, time
import argparse
from AngryTops.features import *
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np
from AngryTops.Plotting.PlottingHelper import *
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

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

# Distribution of Chi-Squareds
histograms['chi_squared_all_events'] = TH1F("Distribution of Chi Squared values of all events in the sample",  "Unitless", 500, 0., 20.)
histograms['chi_squared_all_events_NDF'] = TH1F("Distribution of Chi Squared values / NDF of all events in the sample",  "Unitless", 500, 0., 20.)

################################################################################

# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)

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

    # Calculate chi-squared/NDF
    chi22NDF = chi22 / 12

    # Populate the histograms:
    histograms['chi_squared_all_events'].Fill(chi22, w)
    histograms['chi_squared_all_events_NDF'].Fill(chi22NDF, w)


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


from AngryTops.Plotting.PlottingHelper import MakeCanvas

# Plot histograms inside outputdir, a subdir of training_dir
for key in histograms:
    plot_hists(key, histograms[key], training_dir+outputdir)





# for obs in attributes:
#     # Load the histograms
#     hname_true = "%s_true" % (obs)
#     hname_fitted = "%s_fitted" % (obs)

#     # True and fitted leaf
#     h_true = histograms[hname_true]
#     h_fitted = histograms[hname_fitted]
#     if h_true == None:
#         print ("ERROR: invalid histogram for", obs)

#     # Axis titles
#     xtitle = h_true.GetXaxis().GetTitle()
#     ytitle = h_true.GetYaxis().SetTitle("A.U.")
#     if h_true.Class() == TH2F.Class():
#         h_true = h_true.ProfileX("pfx")
#         h_true.GetYaxis().SetTitle( ytitle )
#     else:
#         Normalize(h_true)
#         Normalize(h_fitted)

#     # Set Style
#     SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
#     SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

#     c, pad0, pad1 = MakeCanvas()
#     pad0.cd()
#     gStyle.SetOptTitle(0)

#     h_true.Draw("h")
#     h_fitted.Draw("h same")
#     hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum() ] )
#     h_fitted.SetMaximum( hmax )
#     h_true.SetMaximum( hmax )
#     h_fitted.SetMinimum( 0. )
#     h_true.SetMinimum( 0. )

#     leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
#     leg.SetFillColor(0)
#     leg.SetFillStyle(0)
#     leg.SetBorderSize(0)
#     leg.SetTextFont(42)
#     leg.SetTextSize(0.05)
#     leg.AddEntry( h_true, "MG5+Py8", "f" )
#     leg.AddEntry( h_fitted, "Predicted", "f" )
#     leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
#     leg.Draw()

#     gPad.RedrawAxis()
#     if caption is not None:
#         newpad = TPad("newpad","a caption",0.1,0,1,1)
#         newpad.SetFillStyle(4000)
#         newpad.Draw()
#         newpad.cd()
#         title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
#         title.SetFillColor(16)
#         title.SetTextFont(52)
#         title.Draw()

#         gPad.RedrawAxis()

#     pad1.cd()

#     yrange = [0.4, 1.6]
#     frame, tot_unc, ratio = DrawRatio(h_true, h_fitted, xtitle, yrange)

#     gPad.RedrawAxis()

#     c.cd()

#     c.SaveAs("{0}/{1}/{2}.png".format(training_dir, outputdir, obs))
#     pad0.Close()
#     pad1.Close()
#     c.Close()

# for hist_name in corr_2d:

#     # True and fitted leaf
#     hist = histograms[hist_name]
#     if hist == None:
#         print ("ERROR: invalid histogram for", hist_name)

#     #Normalize(hist)

#     SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

#     c = TCanvas()
#     c.cd()

#     pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
#     pad0.SetLeftMargin( 0.18 ) #0.16
#     pad0.SetRightMargin( 0.05 )
#     pad0.SetBottomMargin( 0.18 )
#     #pad0.SetTopMargin( 0.14 )
#     pad0.SetTopMargin( 0.07 ) #0.05
#     pad0.SetFillColor(0)
#     pad0.SetFillStyle(4000)
#     pad0.Draw()
#     pad0.cd()

#     hist.Draw("colz")

#     corr = hist.GetCorrelationFactor()
#     l = TLatex()
#     l.SetNDC()
#     l.SetTextFont(42)
#     l.SetTextColor(kBlack)
#     l.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

#     gPad.RedrawAxis()

#     if caption is not None:
#         newpad = TPad("newpad","a caption",0.1,0,1,1)
#         newpad.SetFillStyle(4000)
#         newpad.Draw()
#         newpad.cd()
#         title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
#         title.SetFillColor(16)
#         title.SetTextFont(52)
#         title.Draw()

#     c.cd()

#     c.SaveAs("{0}/{1}/{2}.png".format(training_dir, outputdir, hist_name))
#     pad0.Close()
#     c.Close()


