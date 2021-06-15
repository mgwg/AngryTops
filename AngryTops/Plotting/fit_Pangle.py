#!/usr/bin/env python
import os, sys, time
import argparse
from ROOT import *
from array import array
import pickle
import numpy as np
from numpy.linalg import norm
import sklearn.preprocessing
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

# To create histograms of the angle between the t momentum and the vector sum of the W and b momenta. 
#  Also, to create histograms of the "projection" of the vector sum onto the t momentum. 
# fitted.root must already be created using fit.py, which is usually done with make_plots.sh 

################################################################################
# CONSTANTS

training_dir = sys.argv[1]
representation = sys.argv[2]
caption = sys.argv[3]
logyaxis = sys.argv[4] #Whether or not to use logarithmic y-axis

m_t = 172.5
m_W = 80.4
m_b = 4.95
if len(sys.argv) > 1: training_dir = sys.argv[1]
infilename = "{}/fitted.root".format(training_dir)
ofilename = "{}/histograms_P_angle".format(training_dir)
print(infilename)

if caption == "None": caption = None

np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(training_dir)

# ################################################################################
# # HELPER FUNCTIONS
# def PrintOut( p4_true, p4_fitted, label ):
#   print("%s :: true=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f ) \
#         :: fitted=( %4.1f, %3.2f, %3.2f, %4.1f ; %3.1f )" % \
#         (label, p4_true.Pt(), p4_true.Rapidity(), p4_true.Phi(), p4_true.E(), p4_true.M(),\
#         p4_fitted.Pt(), p4_fitted.Rapidity(), p4_fitted.Phi(), p4_fitted.E(), p4_fitted.M()
#         ))

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
histograms['Wbt_had_angle_true']        = TH1F( "Wbt_had_angle_true",   ";Hadronic Angle [Rad]", 50, 0.0, 3.2)
histograms['Wbt_lep_angle_true']        = TH1F( "Wbt_lep_angle_true",   ";Leptonic Angle [Rad]", 50, 0.0, 3.2)

histograms['Wbt_had_proj_true'] = TH1F( "Wbt_had_proj_true", ";Hadronic \
    #frac{(#vec{p}_{W}+#vec{p}_{b}) #upoint #vec{p}_{top}}{#vec{p}_{top} #upoint #vec{p}_{top}}", 80, -3, 5)
histograms['Wbt_lep_proj_true'] = TH1F( "Wbt_lep_proj_true",\
       ";Leptonic #frac{(#vec{p}_{W}+#vec{p}_{b}) #upoint #vec{p}_{top}}{#vec{p}_{top} #upoint #vec{p}_{top}}", 80, -3, 5)

# Fitted
histograms['Wbt_had_angle_fitted']        = TH1F( "Wbt_had_angle_fitted",   ";Hadronic Angle [Rad]", 50, 0.0, 3.2)
histograms['Wbt_lep_angle_fitted']        = TH1F( "Wbt_lep_angle_fitted",   ";Leptonic Angle [Rad]", 50, 0.0, 3.2)

histograms['Wbt_had_proj_fitted'] = TH1F( "Wbt_had_proj_fitted",\
       ";Hadronic #frac{(#vec{p}_{W}+#vec{p}_{b}) #upoint #vec{p}_{top}}{#vec{p}_{top} #upoint #vec{p}_{top}}", 80, -3, 5)
histograms['Wbt_lep_proj_fitted'] = TH1F( "Wbt_lep_proj_fitted",\
       ";Leptonic #frac{(#vec{p}_{W}+#vec{p}_{b}) #upoint #vec{p}_{top}}{#vec{p}_{top} #upoint #vec{p}_{top}}", 80, -3, 5)

################################################################################
# FORMAT HISTOGRAMS
for hname, h in histograms.iteritems():
  h.Sumw2()
  if hname.endswith("true")>-1:
    h.SetMarkerColor(kRed)
    h.SetLineColor(kRed)
    h.SetMarkerStyle(24)

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

    # Create lists for the momentum vectors, no energy. 
    #  This allows numpy methods to be used on whole vectors at once instead of element-wise
    W_had_true = [tree.W_had_px_true, tree.W_had_py_true, tree.W_had_pz_true]
    b_had_true = [tree.b_had_px_true, tree.b_had_py_true, tree.b_had_pz_true]
    Wb_had_true = np.add(W_had_true, b_had_true)
    t_had_true = [tree.t_had_px_true, tree.t_had_py_true, tree.t_had_pz_true]

    W_lep_true = [tree.W_lep_px_true, tree.W_lep_py_true, tree.W_lep_pz_true]
    b_lep_true = [tree.b_lep_px_true, tree.b_lep_py_true, tree.b_lep_pz_true]
    Wb_lep_true = np.add(W_lep_true, b_lep_true)
    t_lep_true = [tree.t_lep_px_true, tree.t_lep_py_true, tree.t_lep_pz_true]
    
    W_had_fitted = [tree.W_had_px_fitted, tree.W_had_py_fitted, tree.W_had_pz_fitted]
    b_had_fitted = [tree.b_had_px_fitted, tree.b_had_py_fitted, tree.b_had_pz_fitted]
    Wb_had_fitted = np.add(W_had_fitted, b_had_fitted)
    t_had_fitted = [tree.t_had_px_fitted, tree.t_had_py_fitted, tree.t_had_pz_fitted]
    
    W_lep_fitted = [tree.W_lep_px_fitted, tree.W_lep_py_fitted, tree.W_lep_pz_fitted]
    b_lep_fitted = [tree.b_lep_px_fitted, tree.b_lep_py_fitted, tree.b_lep_pz_fitted]
    Wb_lep_fitted = np.add(W_lep_fitted, b_lep_fitted)
    t_lep_fitted = [tree.t_lep_px_fitted, tree.t_lep_py_fitted, tree.t_lep_pz_fitted]

    # Calculate angles
    try:
        # angle between vectors = a dot b / norm of a / norm of b
        Wbt_had_angle_true = np.arccos(np.dot(Wb_had_true, t_had_true) / norm(Wb_had_true) / norm(t_had_true))
        Wbt_lep_angle_true = np.arccos(np.dot(Wb_lep_true, t_lep_true) / norm(Wb_lep_true) / norm(t_lep_true))
        Wbt_had_angle_fitted = np.arccos(np.dot(Wb_had_fitted, t_had_fitted) / norm(Wb_had_fitted) / norm(t_had_fitted))
        Wbt_lep_angle_fitted = np.arccos(np.dot(Wb_lep_fitted, t_lep_fitted) / norm(Wb_lep_fitted) / norm(t_lep_fitted))
        
        # projection is dot product between p_W + p_b and p_t divided by norm squared of p_t
        Wbt_had_proj_true = np.dot(Wb_had_true, t_had_true) / np.dot(t_had_true, t_had_true)
        Wbt_lep_proj_true = np.dot(Wb_lep_true, t_lep_true) / np.dot(t_lep_true, t_lep_true)
        Wbt_had_proj_fitted = np.dot(Wb_had_fitted, t_had_fitted) / np.dot(t_had_fitted, t_had_fitted)
        Wbt_lep_proj_fitted = np.dot(Wb_lep_fitted, t_lep_fitted) / np.dot(t_lep_fitted, t_lep_fitted)

    except Exception as e:
        print("WARNING: invalid, skipping event ( rn=%-10i en=%-10i )" % ( tree.runNumber, tree.eventNumber ))
        # PrintOut( t_lep_true, t_lep_fitted, "Leptonic top" )
        print(e)
        continue

################################################################################
# FILL TREES

    histograms['Wbt_had_angle_true'].Fill(  Wbt_had_angle_true,  w )
    histograms['Wbt_lep_angle_true'].Fill(  Wbt_lep_angle_true,  w )
    histograms['Wbt_had_angle_fitted'].Fill(  Wbt_had_angle_fitted,  w )
    histograms['Wbt_lep_angle_fitted'].Fill(  Wbt_lep_angle_fitted,  w )

    histograms['Wbt_had_proj_true'].Fill(  Wbt_had_proj_true,  w )
    histograms['Wbt_lep_proj_true'].Fill(  Wbt_lep_proj_true,  w )
    histograms['Wbt_had_proj_fitted'].Fill(  Wbt_had_proj_fitted,  w )
    histograms['Wbt_lep_proj_fitted'].Fill(  Wbt_lep_proj_fitted,  w )

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

def plot_observables(obs):

    # Load only the appropriate histograms for each observable 
    hname_true = "%s_true" % (obs)
    hame_fitted = "%s_fitted" % (obs)

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
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 ) # color = kMagenta-8, fillcolor = kMagenta-10
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 ) # color = kMagenta-1

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_true.Draw("hist") # "hist" to remove error bars
    h_fitted.Draw("hist same")
    hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum() ] )
    h_fitted.SetMaximum( hmax )
    h_true.SetMaximum( hmax )
    # h_fitted.SetMinimum( 0. ) # Don't set minimum value to 0 for a log scale to work
    # h_true.SetMinimum( 0. ) # Don't set minimum value to 0 for a log scale to work

    # If logyaxis is true, then make y-axis have a log scale
    if logyaxis == 'logyaxis=True':
        pad0.SetLogy()
        pad1.SetLogy()

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

    # Display bin width
    binWidth = h_true.GetBinWidth(0)
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.65, 0.80, "Bin Width: %.2f" % binWidth )

    gPad.RedrawAxis()
    if caption is not None:
        newpad = TPad("newpad","a caption",0.1,0,1,1)
        newpad.SetFillStyle(4000)
        newpad.Draw()
        newpad.cd()
        title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
        title.SetFillColor(16) #kMagenta-10
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

    # Save histograms as png files with titles depending on whether or not y-axis is logarithmic.
    if logyaxis == 'logyaxis=True':
        c.SaveAs("{0}/P_angle/{1}_log.png".format(training_dir, obs))
    else:
        c.SaveAs("{0}/P_angle/{1}.png".format(training_dir, obs))

    pad0.Close()
    pad1.Close()
    c.Close()

################################################################################
if __name__==   "__main__":
    try:
        os.mkdir('{}/P_angle'.format(training_dir))
    except Exception as e:
        print("Overwriting existing files")
    infilename_plot = "{}/histograms_P_angle".format(training_dir)
    infile_plot = TFile.Open(infilename_plot)

    # attributes = list of histograms for which to compare true and fitted
    attributes = ['Wbt_had_angle', 'Wbt_lep_angle', 'Wbt_had_proj', 'Wbt_lep_proj']

    # Make a plot for each observable
    for obs in attributes:
        plot_observables(obs)