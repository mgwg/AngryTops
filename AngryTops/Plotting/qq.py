#!/usr/bin/env python
import os, sys, time
import argparse
from AngryTops.features import *
from ROOT import *
from array import array
import cPickle as pickle
import numpy as np

################################################################################
# CONSTANTS
m_t = 172.5
m_W = 80.4
m_b = 4.95
if len(sys.argv) > 1: training_dir = sys.argv[1]
infilename = "{}/fitted.root".format(training_dir)
print(infilename)
logfilename = "{}/log.txt".format(training_dir)

crop = None
if len(sys.argv) > 4: 
    crop = float(sys.argv[4])

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

# Read in log file and extract values for the legend on the QQ-plots.
with open(logfilename, "r") as logfile:
    counter = 0
    while counter < 4:
        line = logfile.readline()
        if "Architecture" in line:
        	architecture = line
        	print(architecture)
        	counter += 1
    
        elif "Total Number of Epochs" in line:
        	total_epochs = line
        	print(total_epochs)
        	counter +=1

        elif "Representation" in line:
        	representation = line
        	print(representation)
        	counter +=1

        elif "Scaling" in line:
        	scaling = line
        	print(scaling)
        	counter +=1

################################################################################
# MAKE EMPTY DICTIONARY OF DIFFERENT LISTS

trueLists = {}
fittedLists = {}

# True
trueLists['W_had_px']       = []
trueLists['W_had_py']       = []
trueLists['W_had_pz']       = []
trueLists['W_had_pt']       = []
trueLists['W_had_y']        = []
trueLists['W_had_phi']      = []
trueLists['W_had_E']        = []
trueLists['W_had_m']        = []

trueLists['b_had_px']       = []
trueLists['b_had_py']       = []
trueLists['b_had_pz']       = []
trueLists['b_had_pt']       = []
trueLists['b_had_y']        = []
trueLists['b_had_phi']      = []
trueLists['b_had_E']        = []
trueLists['b_had_m']        = []

trueLists['t_had_px']       = []
trueLists['t_had_py']       = []
trueLists['t_had_pz']       = []
trueLists['t_had_pt']       = []
trueLists['t_had_y']        = []
trueLists['t_had_phi']      = []
trueLists['t_had_E']        = []
trueLists['t_had_m']        = []

trueLists['W_lep_px']       = []
trueLists['W_lep_py']       = []
trueLists['W_lep_pz']       = []
trueLists['W_lep_pt']       = []
trueLists['W_lep_y']        = []
trueLists['W_lep_phi']      = []
trueLists['W_lep_E']        = []
trueLists['W_lep_m']        = []

trueLists['b_lep_px']       = []
trueLists['b_lep_py']       = []
trueLists['b_lep_pz']       = []
trueLists['b_lep_pt']       = []
trueLists['b_lep_y']        = []
trueLists['b_lep_phi']      = []
trueLists['b_lep_E']        = []
trueLists['b_lep_m']        = []

trueLists['t_lep_px']       = []
trueLists['t_lep_py']       = []
trueLists['t_lep_pz']       = []
trueLists['t_lep_pt']       = []
trueLists['t_lep_y']        = []
trueLists['t_lep_phi']      = []
trueLists['t_lep_E']        = []
trueLists['t_lep_m']        = []

# Fitted
fittedLists['W_had_px']       = []
fittedLists['W_had_py']       = []
fittedLists['W_had_pz']       = []
fittedLists['W_had_pt']       = []
fittedLists['W_had_y']        = []
fittedLists['W_had_phi']      = []
fittedLists['W_had_E']        = []
fittedLists['W_had_m']        = []

fittedLists['b_had_px']       = []
fittedLists['b_had_py']       = []
fittedLists['b_had_pz']       = []
fittedLists['b_had_pt']       = []
fittedLists['b_had_y']        = []
fittedLists['b_had_phi']      = []
fittedLists['b_had_E']        = []
fittedLists['b_had_m']        = []

fittedLists['t_had_px']       = []
fittedLists['t_had_py']       = []
fittedLists['t_had_pz']       = []
fittedLists['t_had_pt']       = []
fittedLists['t_had_y']        = []
fittedLists['t_had_phi']      = []
fittedLists['t_had_E']        = []
fittedLists['t_had_m']        = []

fittedLists['W_lep_px']       = []
fittedLists['W_lep_py']       = []
fittedLists['W_lep_pz']       = []
fittedLists['W_lep_pt']       = []
fittedLists['W_lep_y']        = []
fittedLists['W_lep_phi']      = []
fittedLists['W_lep_E']        = []
fittedLists['W_lep_m']        = []

fittedLists['b_lep_px']       = []
fittedLists['b_lep_py']       = []
fittedLists['b_lep_pz']       = []
fittedLists['b_lep_pt']       = []
fittedLists['b_lep_y']        = []
fittedLists['b_lep_phi']      = []
fittedLists['b_lep_E']        = []
fittedLists['b_lep_m']        = []

fittedLists['t_lep_px']       = []
fittedLists['t_lep_py']       = []
fittedLists['t_lep_pz']       = []
fittedLists['t_lep_pt']       = []
fittedLists['t_lep_y']        = []
fittedLists['t_lep_phi']      = []
fittedLists['t_lep_E']        = []
fittedLists['t_lep_m']        = []

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

    # True
    trueLists['W_had_px'].append(  W_had_true.Px())
    trueLists['W_had_py'].append(  W_had_true.Py())
    trueLists['W_had_pz'].append(  W_had_true.Pz())
    trueLists['W_had_pt'].append(  W_had_true.Pt())
    trueLists['W_had_y'].append(   W_had_true.Rapidity())
    trueLists['W_had_phi'].append( W_had_true.Phi())
    trueLists['W_had_E'].append(   W_had_true.E())
    trueLists['W_had_m'].append(   W_had_true.M())

    trueLists['b_had_px'].append(  b_had_true.Px())
    trueLists['b_had_py'].append(  b_had_true.Py())
    trueLists['b_had_pz'].append(  b_had_true.Pz())
    trueLists['b_had_pt'].append(  b_had_true.Pt())
    trueLists['b_had_y'].append(   b_had_true.Rapidity())
    trueLists['b_had_phi'].append( b_had_true.Phi())
    trueLists['b_had_E'].append(   b_had_true.E())
    trueLists['b_had_m'].append(   b_had_true.M())

    trueLists['t_had_px'].append(  t_had_true.Px())
    trueLists['t_had_py'].append(  t_had_true.Py())
    trueLists['t_had_pz'].append(  t_had_true.Pz())
    trueLists['t_had_pt'].append(  t_had_true.Pt())
    trueLists['t_had_y'].append(   t_had_true.Rapidity())
    trueLists['t_had_phi'].append( t_had_true.Phi())
    trueLists['t_had_E'].append(   t_had_true.E())
    trueLists['t_had_m'].append(   t_had_true.M())

    trueLists['W_lep_px'].append(  W_lep_true.Px())
    trueLists['W_lep_py'].append(  W_lep_true.Py())
    trueLists['W_lep_pz'].append(  W_lep_true.Pz())
    trueLists['W_lep_pt'].append(  W_lep_true.Pt())
    trueLists['W_lep_y'].append(  W_lep_true.Rapidity())
    trueLists['W_lep_phi'].append(  W_lep_true.Phi())
    trueLists['W_lep_E'].append(  W_lep_true.E())
    trueLists['W_lep_m'].append(  W_lep_true.M())

    trueLists['b_lep_px'].append(  b_lep_true.Px())
    trueLists['b_lep_py'].append(  b_lep_true.Py())
    trueLists['b_lep_pz'].append(  b_lep_true.Pz())
    trueLists['b_lep_pt'].append(  b_lep_true.Pt())
    trueLists['b_lep_y'].append(   b_lep_true.Rapidity())
    trueLists['b_lep_phi'].append( b_lep_true.Phi())
    trueLists['b_lep_E'].append(   b_lep_true.E())
    trueLists['b_lep_m'].append(   b_lep_true.M())

    trueLists['t_lep_px'].append(  t_lep_true.Px())
    trueLists['t_lep_py'].append(  t_lep_true.Py())
    trueLists['t_lep_pz'].append(  t_lep_true.Pz())
    trueLists['t_lep_pt'].append(  t_lep_true.Pt())
    trueLists['t_lep_y'].append(   t_lep_true.Rapidity())
    trueLists['t_lep_phi'].append( t_lep_true.Phi())
    trueLists['t_lep_E'].append(   t_lep_true.E())
    trueLists['t_lep_m'].append(   t_lep_true.M())

    # Fitted
    fittedLists['W_had_px'].append(  W_had_fitted.Px())
    fittedLists['W_had_py'].append(  W_had_fitted.Py())
    fittedLists['W_had_pz'].append(  W_had_fitted.Pz())
    fittedLists['W_had_pt'].append(  W_had_fitted.Pt())
    fittedLists['W_had_y'].append(   W_had_fitted.Rapidity())
    fittedLists['W_had_phi'].append( W_had_fitted.Phi())
    fittedLists['W_had_E'].append(   W_had_fitted.E())
    fittedLists['W_had_m'].append(   W_had_fitted.M())

    fittedLists['b_had_px'].append(  b_had_fitted.Px())
    fittedLists['b_had_py'].append(  b_had_fitted.Py())
    fittedLists['b_had_pz'].append(  b_had_fitted.Pz())
    fittedLists['b_had_pt'].append(  b_had_fitted.Pt())
    fittedLists['b_had_y'].append(   b_had_fitted.Rapidity())
    fittedLists['b_had_phi'].append( b_had_fitted.Phi())
    fittedLists['b_had_E'].append(   b_had_fitted.E())
    fittedLists['b_had_m'].append(   b_had_fitted.M())

    fittedLists['t_had_px'].append(  t_had_fitted.Px())
    fittedLists['t_had_py'].append(  t_had_fitted.Py())
    fittedLists['t_had_pz'].append(  t_had_fitted.Pz())
    fittedLists['t_had_pt'].append(  t_had_fitted.Pt())
    fittedLists['t_had_y'].append(   t_had_fitted.Rapidity())
    fittedLists['t_had_phi'].append( t_had_fitted.Phi())
    fittedLists['t_had_E'].append(   t_had_fitted.E())
    fittedLists['t_had_m'].append(   t_had_fitted.M())

    fittedLists['W_lep_px'].append(  W_lep_fitted.Px())
    fittedLists['W_lep_py'].append(  W_lep_fitted.Py())
    fittedLists['W_lep_pz'].append(  W_lep_fitted.Pz())
    fittedLists['W_lep_pt'].append(  W_lep_fitted.Pt())
    fittedLists['W_lep_y'].append(  W_lep_fitted.Rapidity())
    fittedLists['W_lep_phi'].append(  W_lep_fitted.Phi())
    fittedLists['W_lep_E'].append(  W_lep_fitted.E())
    fittedLists['W_lep_m'].append(  W_lep_fitted.M())

    fittedLists['b_lep_px'].append(  b_lep_fitted.Px())
    fittedLists['b_lep_py'].append(  b_lep_fitted.Py())
    fittedLists['b_lep_pz'].append(  b_lep_fitted.Pz())
    fittedLists['b_lep_pt'].append(  b_lep_fitted.Pt())
    fittedLists['b_lep_y'].append(   b_lep_fitted.Rapidity())
    fittedLists['b_lep_phi'].append( b_lep_fitted.Phi())
    fittedLists['b_lep_E'].append(   b_lep_fitted.E())
    fittedLists['b_lep_m'].append(   b_lep_fitted.M())

    fittedLists['t_lep_px'].append(  t_lep_fitted.Px())
    fittedLists['t_lep_py'].append(  t_lep_fitted.Py())
    fittedLists['t_lep_pz'].append(  t_lep_fitted.Pz())
    fittedLists['t_lep_pt'].append(  t_lep_fitted.Pt())
    fittedLists['t_lep_y'].append(   t_lep_fitted.Rapidity())
    fittedLists['t_lep_phi'].append( t_lep_fitted.Phi())
    fittedLists['t_lep_E'].append(   t_lep_fitted.E())
    fittedLists['t_lep_m'].append(   t_lep_fitted.M())

print("Finished making lists")

################################################################################
# DEFINE AXIS TITLES

labels = {'W_had_px': 'Hadronic W p_{x}',
          'W_had_py': 'Hadronic W p_{y}',
          'W_had_pz': 'Hadronic W p_{z}',
          'W_had_pt': 'Hadronic W p_{t}',
          'W_had_y': 'Hadronic W #eta',
          'W_had_phi': 'Hadronic W #phi',
          'W_had_E': 'Hadronic W E',
          'W_had_m': 'Hadronic W m',

          'b_had_px': 'Hadronic b p_{x}',
          'b_had_py': 'Hadronic b p_{y}',
          'b_had_pz': 'Hadronic b p_{z}',
          'b_had_pt': 'Hadronic b p_{t}',
          'b_had_y': 'Hadronic b #eta',
          'b_had_phi': 'Hadronic b #phi',
          'b_had_E': 'Hadronic b E',
          'b_had_m': 'Hadronic b m',

          't_had_px': 'Hadronic t p_{x}',
          't_had_py': 'Hadronic t p_{y}',
          't_had_pz': 'Hadronic t p_{z}',
          't_had_pt': 'Hadronic t p_{t}',
          't_had_y': 'Hadronic t #eta',
          't_had_phi': 'Hadronic t #phi',
          't_had_E': 'Hadronic t E',
          't_had_m': 'Hadronic t m',

          'W_lep_px': 'Leptonic W p_{x}',
          'W_lep_py': 'Leptonic W p_{y}',
          'W_lep_pz': 'Leptonic W p_{z}',
          'W_lep_pt': 'Leptonic W p_{t}',
          'W_lep_y': 'Leptonic W #eta',
          'W_lep_phi': 'Leptonic W #phi',
          'W_lep_E': 'Leptonic W E',
          'W_lep_m': 'Leptonic W m',

          'b_lep_px': 'Leptonic b p_{x}',
          'b_lep_py': 'Leptonic b p_{y}',
          'b_lep_pz': 'Leptonic b p_{z}',
          'b_lep_pt': 'Leptonic b p_{t}',
          'b_lep_y': 'Leptonic b #eta',
          'b_lep_phi': 'Leptonic b #phi',
          'b_lep_E': 'Leptonic b E',
          'b_lep_m': 'Leptonic b m',

          't_lep_px': 'Leptonic t p_{x}',
          't_lep_py': 'Leptonic t p_{y}',
          't_lep_pz': 'Leptonic t p_{z}',
          't_lep_pt': 'Leptonic t p_{t}',
          't_lep_y': 'Leptonic t #eta',
          't_lep_phi': 'Leptonic t #phi',
          't_lep_E': 'Leptonic t E',
          't_lep_m': 'Leptonic t m'
}

################################################################################
# MAKE PLOTS

try:
    os.mkdir('{}/qq_plots'.format(training_dir))
except Exception as e:
    print("Overwriting existing files")

try:
    os.mkdir('{}/qq_plots/cropped'.format(training_dir))
except Exception as e:
    print("Overwriting existing files")


# make reference line and legend
line = TF1("line", "x", -4000, 4000)
line.SetLineColor(kGray)
line.SetLineWidth(1)

# make legend
l = TLatex()
l.SetNDC()
l.SetTextFont(42)
l.SetTextSize(0.03)
l.SetTextColor(kBlack)

for key in trueLists:
  # sort because points need to be in order to get correct indices
  if crop:
    trueLists[key].sort()
    fittedLists[key].sort()
    n = len(trueLists[key])

  if '_y' in key:
      unit = '#eta'
  elif '_phi' in key:
      unit = '#phi'
  else:
      unit = '[GeV]'

  qq = TGraphQQ(len(fittedLists[key]), np.array(fittedLists[key]), len(trueLists[key]), np.array(trueLists[key]))

  c1 = TCanvas()

  qq.SetMarkerStyle(20)
  qq.SetMarkerSize(1)
  qq.SetMarkerColor(kGray+2)
  qq.SetLineWidth(0)

  qq.SetTitle("QQ Plot of True vs Predicted {}".format(labels[key]))
  qq.GetXaxis().SetTitle("True {}".format(unit)) 
  qq.GetYaxis().SetTitle("Predicted {}".format(unit))

  qq.Draw()

  # set y=x line to same range as variable
  line.SetRange(qq.GetXaxis().GetXmin(), qq.GetXaxis().GetXmax())
  line.Draw("same")

  # draw legend
  l.DrawLatex( 0.15, 0.65, scaling)
  l.DrawLatex( 0.15, 0.7, representation)
  l.DrawLatex( 0.15, 0.75, total_epochs)
  l.DrawLatex( 0.15, 0.8, architecture)
  # Take the date of the run from the third input to make_plots.sh 
  l.DrawLatex( 0.15, 0.85, sys.argv[3]) 
  gPad.RedrawAxis()

  c1.SaveAs("{}/qq_plots/qq_{}.png".format(training_dir, key))
  c1.Close()

  if crop:
    upper = (Double(0), Double(0))
    lower = (Double(0), Double(0))

    if '_m' in key:
          continue     

    elif '_E' in key or '_pt' in key:
        # variables starting at 0
        lower = (0, 0)
        qq.GetPoint(n-1, upper[0], upper[1])
    else:
        # px, py, pz, y, m
        # crop to [-i, i] because of symmetry
        qq.GetPoint(1, lower[0], lower[1])
        qq.GetPoint(n-1, upper[0], upper[1])
        print(lower[0], upper[0])
        print(lower[1], upper[1])

    c2 = TCanvas()
    qq.GetXaxis().SetRangeUser(lower[0]*crop, upper[0]*crop)
    qq.GetYaxis().SetRangeUser(lower[1]*crop, upper[1]*crop)
    qq.Draw()

    # set y=x line to same range as variable
    line.SetRange(qq.GetXaxis().GetXmin(), qq.GetXaxis().GetXmax())
    line.Draw("same")

    # draw legend
    l.DrawLatex( 0.15, 0.6, "Zoom: {}".format(crop))
    l.DrawLatex( 0.15, 0.65, scaling)
    l.DrawLatex( 0.15, 0.7, representation)
    l.DrawLatex( 0.15, 0.75, total_epochs)
    l.DrawLatex( 0.15, 0.8, architecture)
    # Take the date of the run from the third input to make_plots.sh 
    l.DrawLatex( 0.15, 0.85, sys.argv[3]) 
    gPad.RedrawAxis()

    c2.SaveAs("{}/qq_plots/cropped/qq_crop_{}.png".format(training_dir, key))
    c2.Close()   

print("Finished making plots")