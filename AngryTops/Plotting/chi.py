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

pval_cut = [0.0, 0.01, 0.05, 0.1]
# Number of variables to add to chi-squared that is calculated for each event:
ndf = 12
root_dir = '../CheckPoints/'

# first directory is the directory containing the difference plots whose sigma 
# will be used to calculate the Chi-Squareds. Plots are also outputted to this directory.
sigma_dir = sys.argv[1]
# Read in histograms from the sigma directory.
histsFilename = root_dir + sigma_dir + "/histograms.root"
histsFile = TFile.Open(histsFilename)
print("sigma directory filename: {0}\n".format(histsFilename))
print(histsFilename)

# subsequent arguments are the subdirectory names of the training directories where the truth and fitted data are saved.
# data from filtered sample only
good_dir = sys.argv[2]
infilename_good = root_dir + good_dir + "/fitted.root"
print("Training directory filename: {0}".format(infilename_good))

# data of bad events from unfiltered sample
bad_dir = sys.argv[3]
infilename_bad = root_dir + bad_dir + "/fitted.root"
print("Bad events training directory filename: {0}".format(infilename_bad))

# specifies the events in the second hist as 'all events' or 'unreconstructable'
legend = sys.argv[4]
print("type of events: {}".format(legend))

# output directory
outputdir = root_dir + sigma_dir + "/img_chi_pval2/"
infiles = {good_dir: infilename_good, bad_dir: infilename_bad}

################################################################################
# Get sigmas
sigma = {}
for obs in attributes:
    # Use only the difference plots from the sigma folder.
    hist_name = "diff_{0}".format(obs)
    # True and fitted leaf
    hist = histsFile.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", obs)

    #Normalize(hist)
    if hist.Class() == TH2F.Class():
        hist = hist.ProfileX("hist_pfx")

    # Extract FWHM and sigma using helper function.
    __, sigma_single = getFwhm( hist )
    sigma[obs] = sigma_single

# Extract and print relevant standard deviations
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

print("\nsigmas:")

print("\nW_had_phi_sigma: {0}".format(W_had_phi_sigma))
print("W_had_rapidity_sigma: {0}".format(W_had_rapidity_sigma))
print("W_had_pt_sigma: {0}\n".format(W_had_pt_sigma))

print("W_lep_phi_sigma: {0}".format(W_lep_phi_sigma))
print("W_lep_rapidity_sigma: {0}".format(W_lep_rapidity_sigma))
print("W_lep_pt_sigma: {0}\n".format(W_lep_pt_sigma))

print("b_had_phi_sigma: {0}".format(b_had_phi_sigma))
print("b_had_rapidity_sigma: {0}".format(b_had_rapidity_sigma))
print("b_had_pt_sigma: {0}\n".format(b_had_pt_sigma))

print("b_lep_phi_sigma: {0}".format(b_lep_phi_sigma))
print("b_lep_rapidity_sigma: {0}".format(b_lep_rapidity_sigma))
print("b_lep_pt_sigma: {0}\n".format(b_lep_pt_sigma))

################################################################################

histograms = {}
# good events
histograms['chi_squared_all_' + good_dir] = TH1F("#chi^{2} reconstructable",  ";Unitless", 100, 0., 50.)
histograms['chi_squared_all_' + good_dir].SetTitle("#chi^{2} of all reconstructable events; #chi^{2}, Unitless; A.U.")
histograms['chi_squared_all_NDF_' + good_dir] = TH1F("#chi^{2}/NDF reconstructable",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_all_NDF_' + good_dir].SetTitle("#chi^{2}/NDF of all reconstructable events; #chi^{2}, Unitless; A.U.")

histograms['p-values_' + good_dir] = TH1F("p-values reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_' + good_dir].SetTitle("p-value distribution of #chi^{2} statistics for reconstructable events; p-values, Unitless; A.U.")
histograms['p-values_semilog_' + good_dir] = TH1F("p-values reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_semilog_' + good_dir].SetTitle("p-value distribution of #chi^{2} statistics for reconstructable events; p-values, Unitless; A.U.")
histograms['p-values_loglog_' + good_dir] = TH1F("p-values reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_loglog_' + good_dir].SetTitle("p-value distribution of #chi^{2} statistics for reconstructable events; p-values, Unitless; A.U.")

# bad events
histograms['chi_squared_all_' + bad_dir] = TH1F("#chi^{2} un-reconstructable",  ";Unitless", 100, 0., 50.)
histograms['chi_squared_all_' + bad_dir].SetTitle("#chi^{2} of all un-reconstructable events; #chi^{2}, Unitless; A.U.")
histograms['chi_squared_all_NDF_' + bad_dir] = TH1F("#chi^{2}/NDF un-reconstructable",  ";Unitless", 100, 0., 20.)
histograms['chi_squared_all_NDF_' + bad_dir].SetTitle("#chi^{2}/NDF of all un-reconstructable events; #chi^{2}, Unitless; A.U.")

histograms['p-values_' + bad_dir] = TH1F("p-values un-reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_' + bad_dir].SetTitle("p-value distribution of #chi^{2} statistics for un-reconstructable events; p-values, Unitless; A.U.")
histograms['p-values_semilog_' + bad_dir] = TH1F("p-values un-reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_semilog_' + bad_dir].SetTitle("p-value distribution of #chi^{2} statistics for un-reconstructable events; p-values, Unitless; A.U.")
histograms['p-values_loglog_' + bad_dir] = TH1F("p-values un-reconstructable",  ";Unitless", 100, 0., 1.)
histograms['p-values_loglog_' + bad_dir].SetTitle("p-value distribution of #chi^{2} statistics for un-reconstructable events; p-values, Unitless; A.U.")

recon_count = [0., 0., 0., 0.]
all_count = [0., 0., 0., 0.]
efficiency = [0., 0., 0., 0.] # efficiency = percent reconstructable events passing cut
rejection_fact = [0., 0., 0., 0.] # rejection factor = 1/ percent all events passing cut

################################################################################
for subdir in infiles:
    print("PRINTING VALUES FOR: {}".format(subdir))

    infilename = infiles[subdir]
    # Read in input file from training directory, contains truth and fitted data
    infile = TFile.Open( infilename )
    tree   = infile.Get( "nominal")

    ################################################################################
    # Number of events
    n_events = tree.GetEntries()

    print("INFO: starting event loop. Found %i events" % n_events)

    # Define sums of squares to be used for calculating sample chi-squared/NDF
    # One sum for each variable to be augmented in each event
    W_had_phi_sum, W_had_rapidity_sum, W_had_pt_sum = 0., 0., 0.
    W_lep_phi_sum, W_lep_rapidity_sum, W_lep_pt_sum = 0., 0., 0.
    b_had_phi_sum, b_had_rapidity_sum, b_had_pt_sum = 0., 0., 0.
    b_lep_phi_sum, b_lep_rapidity_sum, b_lep_pt_sum = 0., 0., 0.

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
        chi22NDF = chi22 / ndf

        # Calculate a p-value assuming the distribution of sample chi-squared statistics follows
        #  a chi-squared distribution with number of variables degrees of freedom.
        #  Use the survival function defined as 1-CDF.
        p_value = chi2.sf(chi22, ndf)

        # Populate the histograms:
        histograms['chi_squared_all_' + subdir].Fill(chi22)
        histograms['chi_squared_all_NDF_' + subdir].Fill(chi22NDF)
        histograms['p-values_' + subdir].Fill(p_value)
        histograms['p-values_semilog_' + subdir].Fill(p_value)
        histograms['p-values_loglog_' + subdir].Fill(p_value)

        # Augment counters for number of events that pass p-value cuts.
        if subdir == good_dir:
            recon_count[0] += (p_value >= pval_cut[0])
            recon_count[1] += (p_value >= pval_cut[1])
            recon_count[2] += (p_value >= pval_cut[2])
            recon_count[3] += (p_value >= pval_cut[3])
        elif subdir == bad_dir:
            all_count[0] += (p_value >= pval_cut[0])
            all_count[1] += (p_value >= pval_cut[1])
            all_count[2] += (p_value >= pval_cut[2])
            all_count[3] += (p_value >= pval_cut[3])

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

    if subdir == good_dir:
        for i in range(len(efficiency)):
            efficiency[i] = recon_count[i]/n_events
            print("events with p-value greater than {}: {}, {}%".format(pval_cut[i], recon_count[i], recon_count[i]/n_events*100))

    if subdir == bad_dir:
        for i in range(len(efficiency)):
            rejection_fact[i] = all_count[i]/n_events
            print("events with p-value greater than {}: {}, {}%".format(pval_cut[i], all_count[i], all_count[i]/n_events*100))

print("Reconstructable events that pass p-value cuts as a fraction of all events that pass p-value cuts" )
for i in range(len(pval_cut)):
    print("p-val {} : {}%".format(pval_cut[i], recon_count[i]/all_count[i]*100.0))

print("Efficiency x rejection factor (percent reconstructable events / percent all events that pass p-value cuts" )
for i in range(len(pval_cut)):
    print("p-val {} : {}%".format(pval_cut[i], efficiency[i]/rejection_fact[i]))

try:
    os.mkdir(outputdir)
except Exception as e:
    print("Overwriting existing files")


# Plot each histogram in histograms
for key in histograms:
    Normalize(histograms[key])
    plot_hists(key, histograms[key], outputdir)

# these should go into another file, along with all files that require the style formatting
from AngryTops.Plotting.PlottingHelper import *
gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

def plot_observables(h_good, h_bad, caption):

    # Axis titles
    xtitle = h_good.GetXaxis().GetTitle()
    ytitle = h_good.GetYaxis().SetTitle("A.U.")
    h_good.SetStats(0)
    h_bad.SetStats(0)

    # Set Style
    SetTH1FStyle( h_good,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( h_bad, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0 = MakeCanvas3()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_good.Draw("h")
    h_bad.Draw("h same")
    hmax = 1.5 * max( [ h_good.GetMaximum(), h_bad.GetMaximum() ] )
    h_bad.SetMaximum( hmax )
    h_good.SetMaximum( hmax )

    if "semilog" in caption:
        pad0.SetLogy()

    if "loglog" in caption:
        pad0.SetLogx()
        pad0.SetLogy()

    leg = TLegend( 0.15, 0.85, 0.40, 0.90 )
    # leg = TLegend( 0.55, 0.85, 0.80, 0.90 ) # legend on left
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.04)
    leg.AddEntry( h_good, "reconstructable", "f" )
    leg.AddEntry( h_bad, legend, "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    binWidth = h_good.GetBinWidth(0)
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.SetTextSize(0.04)
    l.DrawLatex( 0.65, 0.80, "Bin Width: %.2f" % binWidth )
    # l.DrawLatex( 0.56, 0.70, "Bin Width: %.2f" % binWidth ) # legend on left

    gPad.RedrawAxis()
    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        if 'chi' in caption:
            title = TPaveLabel(0.1,0.94,0.9,0.99, "#chi^{2}/NDF")
        else:
            title = TPaveLabel(0.1,0.94,0.9,0.99, "p-values")
        title.SetFillColor(16)
        title.SetTextFont(52)
        title.Draw()

        gPad.RedrawAxis()

    c.SaveAs("{0}/{1}.png".format(outputdir, caption))
    pad0.Close()
    c.Close()


plot_observables(histograms['chi_squared_all_NDF_' + good_dir], histograms['chi_squared_all_NDF_' + bad_dir], 'chi-squared')
plot_observables(histograms['p-values_semilog_' + good_dir], histograms['p-values_semilog_' + bad_dir], 'pvalues_semilog')
