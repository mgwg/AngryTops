import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists, plot_corr

training_dir = sys.argv[1]
representation = sys.argv[2]
dir_name = ''

if len(dir_name) > 3:
    date = dir_name[3]

subdir = '/bW_obs_img{}/'.format(dir_name)
scaling = True # whether the dataset has been passed through a scaling function or not
m_t = 172.5
m_W = 80.4
m_b = 4.95

###############################################################################
# Read in input file
infilename_good = "{}/tree_fitted_good.root".format(training_dir)
infile_good = TFile.Open( infilename_good )
print(infilename_good)
tree_good   = infile_good.Get( "nominal")

infilename_bad = "{}/tree_fitted_bad.root".format(training_dir)
infile_bad = TFile.Open( infilename_bad )
print(infilename_bad)
tree_bad   = infile_bad.Get( "nominal")


good_events = tree_good.GetEntries()
bad_events = tree_bad.GetEntries()
print("num good events: {} \n num bad events: {}".format(good_events, bad_events))

################################################################################
# MAKE HISTOGRAMS

hists = {}

for i in range(5):
    hists['max_jet_{}_Pt_good'.format(i)] = TH1F("max_jet_{}_Pt_good".format(i),"pT (GeV)", 50, 0, 500)
    hists['max_jet_{}_Pt_good'.format(i)].SetTitle("{} Leading Jet pT Good Events; pT (GeV);A.U.".format(i+1))
    hists['max_jet_{}_Pt_bad'.format(i)] = TH1F("max_jet_{}_Pt_bad".format(i),"pT (GeV)", 50, 0, 500)
    hists['max_jet_{}_Pt_bad'.format(i)].SetTitle("{} Leading Jet pT Bad Events; pT (GeV);A.U.".format(i+1))

################################################################################
# GET VALUES FROM TREE
jet1pt_good = tree_good.AsMatrix(["jet1_pt_obs"]).flatten()
jet2pt_good = tree_good.AsMatrix(["jet2_pt_obs"]).flatten()
jet3pt_good = tree_good.AsMatrix(["jet3_pt_obs"]).flatten()
jet4pt_good = tree_good.AsMatrix(["jet4_pt_obs"]).flatten()
jet5pt_good = tree_good.AsMatrix(["jet5_pt_obs"]).flatten()

jet1pt_bad = tree_bad.AsMatrix(["jet1_pt_obs"]).flatten()
jet2pt_bad = tree_bad.AsMatrix(["jet2_pt_obs"]).flatten()
jet3pt_bad = tree_bad.AsMatrix(["jet3_pt_obs"]).flatten()
jet4pt_bad = tree_bad.AsMatrix(["jet4_pt_obs"]).flatten()
jet5pt_bad = tree_bad.AsMatrix(["jet5_pt_obs"]).flatten()

################################################################################
# POPULATE HISTOGRAMS

jets_Pt_good = np.stack([jet1pt_good, jet2pt_good, jet3pt_good, jet4pt_good, jet5pt_good], axis = 1)
jets_Pt_bad = np.stack([jet1pt_bad, jet2pt_bad, jet3pt_bad, jet4pt_bad, jet5pt_bad], axis = 1)

# sort jets from smallest to largest
jets_Pt_good = np.sort(jets_Pt_good)
jets_Pt_bad = np.sort(jets_Pt_bad)
# reverse array order to list jets from largest to smallest to match order of leading jet (i.e. 0th is the largest, etc.)
jets_Pt_good = np.flip(jets_Pt_good, axis = 1)
jets_Pt_bad = np.flip(jets_Pt_bad, axis = 1)

for i in range(5): 
    jets_Pt_max_good = jets_Pt_good[:,i].flatten()
    jets_Pt_max_bad = jets_Pt_bad[:,i].flatten()

    for j in jets_Pt_max_good.nonzero()[0]: # skip events where the leading jet has 0 pT
        hists['max_jet_{}_Pt_good'.format(i)].Fill( jets_Pt_max_good[j] )

    for j in jets_Pt_max_bad.nonzero()[0]:
        hists['max_jet_{}_Pt_bad'.format(i)].Fill( jets_Pt_max_bad[j] )

try:
    os.mkdir('{}/{}'.format(training_dir, subdir))
except Exception as e:
    print("Overwriting existing files")

for key in hists:
    hist = hists[key]
    plot_hists(key, hist, training_dir+subdir)


def plot_observables(good, bad, i):
    # i is the ith order leading jet

    caption = str(i+1)+' Leading jet p_{T} of Good and Bad Events'
    obs = 'jet_{}_Pt'.format(i)

    # Axis titles
    xtitle = good.GetXaxis().GetTitle()
    ytitle = good.GetYaxis().SetTitle("A.U.")

    Normalize(good)
    Normalize(bad)

    # Set Style
    SetTH1FStyle( good,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( bad, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    good.Draw("h")
    bad.Draw("h same")
    hmax = 1.5 * max( [ good.GetMaximum(), bad.GetMaximum() ] )
    good.SetMaximum( hmax )
    bad.SetMaximum( hmax )
    good.SetMinimum( 0. )
    bad.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( good, "Good Events", "f" )
    leg.AddEntry( bad, "Bad Events", "f" )
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
    frame, tot_unc, ratio = DrawRatio(good, bad, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/{1}.png".format(training_dir+subdir, obs))
    pad0.Close()
    pad1.Close()
    c.Close()

def Normalize( h, sf=1.0 ):
  if h == None: return
  A = h.Integral()
  if A == 0.: return
  h.Scale( sf / A )

from AngryTops.Plotting.PlottingHelper import *

for i in range(5):

    good = hists['max_jet_{}_Pt_good'.format(i)]
    bad = hists['max_jet_{}_Pt_bad'.format(i)]
    good.SetStats(0)
    bad.SetStats(0)

    plot_observables(good, bad, i)
