#!/usr/bin/env python
import os, sys, time
import argparse
from ROOT import *
from array import array
import pickle
import numpy as np
import sklearn.preprocessing
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

# To create histograms of the sum of W and b energies for comparison with the t energy histograms 
# fitted.root must already be created using fit.py, which is usually done with make_plots.sh 

################################################################################
# CONSTANTS

training_dir = sys.argv[1]
representation = sys.argv[2]
caption = sys.argv[3]

m_t = 172.5
m_W = 80.4
m_b = 4.95
if len(sys.argv) > 1: training_dir = sys.argv[1]
infilename = "{}/fitted.root".format(training_dir)
ofilename = "{}/histograms_W_plus_b.root".format(training_dir)
print(infilename)

if caption == "None": caption = None

np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(training_dir)

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

# Open output file
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

################################################################################
# MAKE EMPTY DICTIONARY OF DIFFERENT HISTOGRAMS
histograms = {}

# True
histograms['Wb_had_E_true']        = TH1F( "Wb_had_E_true",   ";Hadronic W+b E [GeV]", 50, 0., 500. )
histograms['Wb_lep_E_true']        = TH1F( "Wb_lep_E_true",   ";Leptonic W+b E [GeV]", 50, 0., 500. )
histograms['t_had_E_true']        = TH1F( "t_had_E_true",   ";Hadronic t E [GeV]", 50, 0., 500. )
histograms['t_lep_E_true']        = TH1F( "t_lep_E_true",    ";Leptonic t E [GeV]", 50, 0., 500. )
# # Fitted
histograms['Wb_had_E_fitted']        = TH1F( "Wb_had_E_fitted",   ";Hadronic W+b E [GeV]", 50, 0., 500. )
histograms['Wb_lep_E_fitted']        = TH1F( "Wb_lep_E_fitted",   ";Leptonic W+b E [GeV]", 50, 0., 500. )
histograms['t_had_E_fitted']        = TH1F( "t_had_E_fitted",   ";Hadronic t E [GeV]", 50, 0., 500. )
histograms['t_lep_E_fitted']        = TH1F( "t_lep_E_fitted",    ";Leptonic t E [GeV]", 50, 0., 500. )
# Correlations; t for true, p for predicted
histograms['corr_tp_had_E']        = TH2F( "corr_tp_had_E",   ";True Hadronic W+b E [GeV];Predicted Hadronic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_tp_lep_E']        = TH2F( "corr_tp_lep_E",   ";True Leptonic W+b E [GeV];Predicted Leptonic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_pp_had_E']        = TH2F( "corr_pp_had_E",   ";Predicted Hadronic t E [GeV];Predicted Hadronic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_pp_lep_E']        = TH2F( "corr_pp_lep_E",    ";Predicted Leptonic t E [GeV];Predicted Leptonic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_tt_had_E']        = TH2F( "corr_tt_had_E",   ";True Hadronic t E [GeV];True Hadronic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )
histograms['corr_tt_lep_E']        = TH2F( "corr_tt_lep_E",    ";True Leptonic t E [GeV];True Leptonic W+b E [GeV]", 50, 150., 500., 50, 150., 500. )

################################################################################
# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)
n_good = 0
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

    try:
        Wb_had_E_true = W_had_true.E() + b_had_true.E()
        Wb_lep_E_true = W_lep_true.E() + b_lep_true.E()
        Wb_had_E_fitted = W_had_fitted.E() + b_had_fitted.E()
        Wb_lep_E_fitted = W_lep_fitted.E() + b_lep_fitted.E()

    except Exception as e:
        print("WARNING: invalid, skipping event ( rn=%-10i en=%-10i )" % ( tree.runNumber, tree.eventNumber ))
        PrintOut( t_lep_true, t_lep_fitted, "Leptonic top" )
        print(e)
        continue

################################################################################
# FILL TREES

    histograms['Wb_had_E_true'].Fill(  Wb_had_E_true,  w )
    histograms['Wb_lep_E_true'].Fill(  Wb_lep_E_true,  w )
    histograms['Wb_had_E_fitted'].Fill(  Wb_had_E_fitted,  w )
    histograms['Wb_lep_E_fitted'].Fill(  Wb_lep_E_fitted,  w )
    histograms['t_had_E_true'].Fill( t_had_true.E(), w)
    histograms['t_lep_E_true'].Fill( t_lep_true.E(), w)
    histograms['t_had_E_fitted'].Fill( t_had_fitted.E(), w)
    histograms['t_lep_E_fitted'].Fill( t_lep_fitted.E(), w)
  
    histograms['corr_tp_had_E'].Fill( Wb_had_E_true, Wb_had_E_fitted, w )  
    histograms['corr_tp_lep_E'].Fill( Wb_lep_E_true, Wb_lep_E_fitted, w )
    histograms['corr_pp_had_E'].Fill( t_had_fitted.E(), Wb_had_E_fitted, w )  
    histograms['corr_pp_lep_E'].Fill( t_lep_fitted.E(), Wb_lep_E_fitted, w )  
    histograms['corr_tt_had_E'].Fill( t_had_true.E(), Wb_had_E_true, w )
    histograms['corr_tt_lep_E'].Fill( t_lep_true.E(), Wb_lep_E_true, w )

    n_good += 1

for histname in histograms:
    histograms[histname].Write(histname)

ofile.Write()
ofile.Close()

print("Finished. Saved output file:", ofilename)
f_good = 100. * float( n_good ) / float( n_events )
print("Good events: %.2f" % f_good)

# plotting
gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

def plot_observables(fitted, true):
    # Load the histograms
    hname_true = true
    hame_fitted = fitted

    # True and fitted leaf
    h_true = infile_plot.Get(hname_true)
    h_fitted = infile_plot.Get(hame_fitted)
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

    # set legend labels
    if "_fitted" in fitted and "_true" in true:
        leg_true = "MG5+Py8 W+b"
        leg_fitted = "Predicted W+b"
    elif "_fitted" in fitted and "_fitted" in true: 
        leg_true = "Predicted t"
        leg_fitted = "Predicted W+b"
    else: # plot truth W+b against truth t
        leg_true = "MG5+Py8 t"
        leg_fitted = "MG5+Py8 W+b"

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, leg_true, "f" )
    leg.AddEntry( h_fitted, leg_fitted, "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    KS = h_true.KolmogorovTest( h_fitted )
    X2 = ChiSquared(h_true, h_fitted) # UU NORM
    
    # get mean and standard deviation
    h_true.GetMean() #axis=1 by default for x-axis
    h_true.GetStdDev()

    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.7, 0.80, "KS test: %.2f" % KS )
    l.DrawLatex( 0.7, 0.75, "#chi^{2}/NDF = %.2f" % X2 )

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
    # h_true must appear before h_fitted as a parameter in DrawRatio to get
    #  the correct ratio of predicted/Monte Carlo
    frame, tot_unc, ratio = DrawRatio(h_true, h_fitted, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/E_fit/{1}_{2}.png".format(training_dir, fitted, true))
    pad0.Close()
    pad1.Close()
    c.Close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_correlations(hist_name):

    # True and fitted leaf
    hist = infile_plot.Get(hist_name)
    if hist == None:
        print ("ERROR: invalid histogram for", hist_name)

    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 )
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    pad0.SetTopMargin( 0.07 )
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

    c.SaveAs("{0}/E_fit/{1}.png".format(training_dir, hist_name))
    pad0.Close()
    c.Close()

################################################################################
if __name__==   "__main__":
    try:
        os.mkdir('{}/E_fit'.format(training_dir))
    except Exception as e:
        print("Overwriting existing files")
    infilename_plot = "{}/histograms_W_plus_b.root".format(training_dir)
    infile_plot = TFile.Open(infilename_plot)

    plot_observables('Wb_had_E_fitted', 'Wb_had_E_true')
    plot_observables('Wb_lep_E_fitted', 'Wb_lep_E_true')
    plot_observables('Wb_had_E_fitted', 't_had_E_fitted')
    plot_observables('Wb_lep_E_fitted', 't_lep_E_fitted')
    plot_observables('Wb_had_E_true', 't_had_E_true')
    plot_observables('Wb_lep_E_true', 't_lep_E_true')    

    # Draw 2D Correlations
    corr_2d = ["corr_tp_had_E", "corr_tp_lep_E", "corr_pp_had_E", "corr_pp_lep_E", "corr_tt_had_E", "corr_tt_lep_E"]
    for corr in corr_2d:
        plot_correlations(corr)
