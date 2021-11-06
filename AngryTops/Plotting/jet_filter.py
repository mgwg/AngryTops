import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4
from AngryTops.features import *

################################################################################
# CONSTANTS
infilename = "../May21/predictions_May21.root"
output_dir = sys.argv[2]
representation = sys.argv[3]
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

b_tagging = NONE   

# t = TFile.Open(infilename)
infile = Tfile(infilename, "READ")
t = infile.Get("nominal")

################################################################################
# MAKE ROOT FILE FOR HISTS

ofilename = "{}/jet_filter_hists".format(output_dir)
# Open output file
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

################################################################################
# POPULATE TREE

print("INFO: starting event loop. Found %i events" % n_events)

jet1pt = t.AsMatrix(["b_jet1_pt_obs"])
jet2pt = t.AsMatrix(["b_jet2_pt_obs"])
jet3pt = t.AsMatrix(["b_jet3_pt_obs"])
jet4pt = t.AsMatrix(["b_jet4_pt_obs"])
jet5pt = t.AsMatrix(["b_jet5_pt_obs"])
jet1btag = t.AsMatrix(["b_jet1_btag_obs"])
jet2btag = t.AsMatrix(["b_jet2_btag_obs"])
jet3btag = t.AsMatrix(["b_jet3_btag_obs"])
jet4btag = t.AsMatrix(["b_jet4_btag_obs"])
jet5btag = t.AsMatrix(["b_jet5_btag_obs"])
hadWpt = t.AsMatrix(["b_W_had_pt_true"])

n_events = t.GetEntries()

################################################################################
# HISTOGRAMS
hists = {}
t.Draw("b_W_had_pt_true >> had_W_Pt(50,0,500)")
t.Draw("b_b_had_pt_true >> had_b_Pt(50,0,500)")
t.Draw("b_b_lep_pt_true >> lep_W_Pt(50,0,500)")
had_W_Pt = gROOT.FindObject ("had_W_Pt")
had_b_Pt = gROOT.FindObject ("had_b_Pt")
lep_b_Pt = gROOT.FindObject ("lep_b_Pt")
hists['had_W_Pt'] = htemp.Clone("had_W_Pt")
hists['had_b_Pt'] = htemp.Clone("had_b_Pt")
hists['lep_b_Pt'] = htemp.Clone("lep_b_Pt")

for i in range(5):
    hists['max_jet_{}_Pt'.format(i)] = TH1F("max_jet_{}_Pt'.format(i)","p_{T} (GeV)", 50, 0, 500)
    hists['max_jet_{}_Pt'.format(i)].SetTitle("{} leading jet pT".format(i+1))
    hists['jet_{}_had_W_Pt_diff'.format(i)] = TH1F("jet_{}_had_W_Pt_diff".format(i),"p_{T} (GeV)", 50, -300, 300)
    hists['jet_{}_had_W_Pt_diff'.format(i)].SetTitle("Leading Jet p_{T} - Had W p_{T}; p_{T} (GeV);A.U.")
    
    # b tagged jets
    hists['b_max_jet_{}_Pt'.format(i)] = TH1F("max_jet_{}_Pt'.format(i)","p_{T} (GeV)", 50, 0, 500)
    hists['b_max_jet_{}_Pt'.format(i)].SetTitle("{} leading jet pT".format(i+1))

# each row corresponds to an event
jets_Pt = np.stack([jet1pt, jet2pt, jet3pt, jet4pt, jet5pt], axis = 1)
# btagging mask
jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)

# only keep jets that are NOT btagged
jets_notb_Pt[jets_btag == 0] = 0
# remove events where all jets are btagged
jets_notb_Pt[~np.all(jets_notb_Pt==0, axis =1)]

# find btagged jets
jets_b_Pt[jets_btag != 0] = 0
jets_b_Pt[~np.all(jets_b_Pt==0, axis =1)]

# sort jets from smallest to largest
jets_notb_Pt = np.sort(jets_notb_Pt)
jets_b_Pt = np.sort(jets_b_Pt)
# reverse array order to list jets from largest to smallest to match order of leading jet (i.e. 0th is the largest, etc.)
jets_b_Pt = np.flip(jets_b_Pt, axis = 1)

for i in range(5): 
    jets_Pt_max = jets_Pt[:,i] 
    hists['max_jet_{}_Pt'.format(i)].FillN(len(jets_Pt_max), jets_Pt_max)
    hists['jet_{}_had_W_Pt_diff'.format(i)].FillN(len(jets_Pt_max), jets_Pt_max - hadWpt)

    hists['b_max_jet_{}_Pt'.format(i)].FillN(len(jets_Pt_max), jets_Pt_max)

for histname in hists:
    hists[histname].Write(histname)

ofile.Write()
ofile.Close()

infile = TFile.Open(ofilename)

def plot_hist(h, obs, caption):

    c = TCanvas()
    h.Draw("H")

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    
    c.SaveAs("{0}/img/{1}.png".format(output_dir, obs))
    # pad0.Close()
    c.Close()

def plot_observables(h_jet, h_quark, obs, caption):

    # Axis titles
    xtitle = h_jet.GetXaxis().GetTitle()
    ytitle = h_jet.GetYaxis().SetTitle("A.U.")
    if h_jet.Class() == TH2F.Class():
        h_jet = h_jet.ProfileX("pfx")
        h_jet.GetYaxis().SetTitle( ytitle )
    else:
        Normalize(h_jet)
        Normalize(h_quark)

    # Set Style
    SetTH1FStyle( h_jet,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( h_quark, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_jet.Draw("h")
    h_quark.Draw("h same")
    hmax = 1.5 * max( [ h_jet.GetMaximum(), h_quark.GetMaximum() ] )
    h_quark.SetMaximum( hmax )
    h_jet.SetMaximum( hmax )
    h_quark.SetMinimum( 0. )
    h_jet.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_jet, "Leading Jet", "f" )
    leg.AddEntry( h_quark, "Hadronic W", "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    KS = h_jet.KolmogorovTest( h_quark )
    X2 = ChiSquared(h_jet, h_quark) # UU NORM
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
    frame, tot_unc, ratio = DrawRatio(h_jet, h_quark, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/img/{1}.png".format(output_dir, obs))
    pad0.Close()
    pad1.Close()
    c.Close()

def plot_correlations(hist_name, caption):

    hist = hists['corr_max_jet_v_had_W_Pt']
    if hist == None:
        print ("ERROR: invalid histogram for", hist_name)

    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
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

    c.SaveAs("{0}/img/{1}.png".format(output_dir, hist_name))
    pad0.Close()
    c.Close()

try:
    os.mkdir('{}/img'.format(output_dir))
except Exception as e:
    print("Overwriting existing files")

def Normalize( h, sf=1.0 ):
  if h == None: return
  A = h.Integral()
  if A == 0.: return
  h.Scale( sf / A )

hist_W = infile.Get('had_W_Pt')
hist_b_had = infile.Get('had_b_Pt')
hist_b_lep = infile.Get('lep_b_Pt')

gStyle.SetOptStat("emr")#;
for i in range(5):
    hist_pt_diff = infile.Get('jet_{}_had_W_Pt_diff'.format(i))
    plot_hist(hist_pt_diff, 'jet_{}_W_Pt_diff'.format(i), str(i+1)+' Leading Jet p_{T} - Hadronic W p_{T}')

# plot in a separate loop because the style changes and I haven't figured out how to reverse it yet
from AngryTops.Plotting.PlottingHelper import *
Normalize(hist_W)
Normalize(hist_b_had)
Normalize(hist_b_lep)
for i in range(5):
    hist_jet = infile.Get('max_jet_{}_Pt'.format(i))
    hist_jet_b = infile.Get('b_max_jet_{}_Pt'.format(i))
    Normalize(hist_jet)
    Normalize(hist_jet_b)
    plot_observables(hist_jet, hist_W, 'jet_{}_v_W_Pt'.format(i), str(i+1)+' Leading Jet p_{T} vs Hadronic W p_{T}')
    plot_observables(hist_jet, hist_b_had, 'jet_{}_v_had_b_Pt'.format(i), str(i+1)+' Leading Jet p_{T} vs Hadronic b p_{T}')
    plot_observables(hist_jet, hist_b_lep, 'jet_{}_v_lep_b_Pt'.format(i), str(i+1)+' Leading Jet p_{T} vs Leptonic b p_{T}')
