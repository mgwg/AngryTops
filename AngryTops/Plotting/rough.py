#!/usr/bin/env python
import os, sys
from ROOT import *
import numpy as np
from AngryTops.features import *
# from AngryTops.Plotting.PlottingHelper import *

def Normalize( h, sf=1.0 ):
  if h == None: return
  A = h.Integral()
  if A == 0.: return
  h.Scale( sf / A )

def plot_hist(obs):
    h = infile.Get("%s_true" % (obs))
    h.SetTitle(obs+" true")
    c = TCanvas()
    Normalize(h)
    h.SetMarkerStyle(0)
    h.Draw("H")

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    
    c.SaveAs("{0}/img/{1}_true.png".format("../CheckPoints/Oct16", obs))
    # pad0.Close()
    c.Close()

infilename = "{}/histograms.root".format("../CheckPoints/May27")
infile = TFile.Open(infilename)

# Make a plot for each observable
for obs in attributes:
    if "_pt" in obs:
        plot_hist(obs)