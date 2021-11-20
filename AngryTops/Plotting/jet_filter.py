import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4
from AngryTops.features import *
from array import array

################################################################################
# CONSTANTS
output_dir = sys.argv[2]
infilename = "{}/predictions_{}.root".format(output_dir, sys.argv[1])
representation = sys.argv[3]
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

# t = TFile.Open(infilename)
infile = TFile(infilename, "READ")
t = infile.Get("nominal")

################################################################################
# MAKE ROOT FILE FOR HISTS

ofilename = "{}/jet_filter_hists.root".format(output_dir)
# Open output file
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

################################################################################
# GET VALUES FROM TREE
jet1pt = t.AsMatrix(["jet1_pt_obs"]).flatten()
jet2pt = t.AsMatrix(["jet2_pt_obs"]).flatten()
jet3pt = t.AsMatrix(["jet3_pt_obs"]).flatten()
jet4pt = t.AsMatrix(["jet4_pt_obs"]).flatten()
jet5pt = t.AsMatrix(["jet5_pt_obs"]).flatten()
jet1btag = t.AsMatrix(["jet1_btag_obs"]).flatten()
jet2btag = t.AsMatrix(["jet2_btag_obs"]).flatten()
jet3btag = t.AsMatrix(["jet3_btag_obs"]).flatten()
jet4btag = t.AsMatrix(["jet4_btag_obs"]).flatten()
jet5btag = t.AsMatrix(["jet5_btag_obs"]).flatten()
hadWpt = t.AsMatrix(["W_had_pt_true"]).flatten()
hadbpt = t.AsMatrix(["b_had_pt_true"]).flatten()
lepbpt = t.AsMatrix(["b_lep_pt_true"]).flatten()

n_events = t.GetEntries()

################################################################################
# HISTOGRAMS
hists = {}
hists['had_W_Pt'] = TH1F("had_W_Pt","p_{T} (GeV)", 50, 0, 500)
hists['had_b_Pt'] = TH1F("had_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['lep_b_Pt'] = TH1F("lep_b_Pt","p_{T} (GeV)", 50, 0, 500)

hists['jet12_had_W_Pt_diff'] = TH1F("jet12_had_W_Pt_diff","p_{T} (GeV)", 50, -300, 300)
hists['jet12_had_W_Pt_diff'].SetTitle("1+2 Leading Jet p_{T} - Had W p_{T}; p_{T} (GeV);A.U.")

hists['jet12_Pt'] = TH1F("jet12_Pt","p_{T} (GeV)", 50, 0, 500)
hists['jet12_Pt'].SetTitle("1+2 Leading Jet p_{T}; p_{T} (GeV);A.U.")

# leading b tagged jet
hists['jet1_b_Pt'] = TH1F("jet1_b_Pt","p_{T} (GeV)", 50, 0, 500)

# b-tagging categories
# had_W_Pt_3 includes events with 3 or more b-tagged jets
for j in range(4):
    hists['had_W_Pt_{}'.format(j)] = TH1F("had_W_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    
    hists['jet12_Pt_{}'.format(j)] = TH1F("jet12_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    hists['jet12_Pt_{}'.format(j)].SetTitle("1+2 Leading Jet p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")

    hists['jet12_had_W_Pt_{}_diff'.format(j)] = TH1F("jet12_had_W_Pt_{}_diff".format(j),"p_{T} (GeV)", 50, -300, 300)
    hists['jet12_had_W_Pt_{}_diff'.format(j)].SetTitle("1+2 Leading Jet p_{T} - Had W p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")

# each row corresponds to an event, one array for non-btagged jets and btagged jets
jets_Pt = np.stack([jet1pt, jet2pt, jet3pt, jet4pt, jet5pt], axis = 1)
jets_b_Pt = np.stack([jet1pt, jet2pt, jet3pt, jet4pt, jet5pt], axis = 1)
# btagging mask
jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)

# only keep jets that are NOT btagged; set btagged jets to 0
jets_Pt[jets_btag != 0] = 0
# find btagged jets
jets_b_Pt[jets_btag == 0] = 0

# sort jets from smallest to largest
jets_Pt = np.sort(jets_Pt)
jets_b_Pt = np.sort(jets_b_Pt)
# reverse array order to list jets from largest to smallest to match order of leading jet (i.e. 0th is the largest, etc.)
jets_Pt = np.flip(jets_Pt, axis = 1)
jets_b_Pt = np.flip(jets_b_Pt, axis = 1)

################################################################################
# FILL HISTOGRAMS
jet12_Pt = jets_Pt[:,0].flatten() + jets_Pt[:,1].flatten()
jet1_b_Pt = jets_b_Pt[:,0].flatten()

# Fill b quark histograms
for i in jet1_b_Pt.nonzero()[0]:
    hists['had_b_Pt'].Fill( hadbpt[i] )
    hists['lep_b_Pt'].Fill( lepbpt[i] )
    hists['jet1_b_Pt'].Fill(jet1_b_Pt[i])

for i in jet12_Pt.nonzero()[0]: # skip events where the leading jet has 0 pT
    b_tag_type = int(sum(jets_btag[i]))
    if b_tag_type >= 3: # set to 3 if greater or equal to 3 for easier indexing 
        b_tag_type = 3

    hists['jet12_had_W_Pt_diff'].Fill( jet12_Pt[i] - hadWpt[i] )
    hists['jet12_had_W_Pt_{}_diff'.format(b_tag_type)].Fill( jet12_Pt[i] - hadWpt[i] )
    
    hists['jet12_Pt'].Fill( jet12_Pt[i] )
    hists['jet12_Pt_{}'.format(b_tag_type)].Fill( jet12_Pt[i] )

    hists['had_W_Pt'].Fill( hadWpt[i] )
    hists['had_W_Pt_{}'.format(b_tag_type)].Fill( hadWpt[i] )

# get number of events in each b-tagging category
num_btag = np.sum(jets_btag, axis = 1)
btag0 = len(np.where(num_btag==0)[0])
btag1 = len(np.where(num_btag==1)[0])
btag2 = len(np.where(num_btag==2)[0])
btag3 = len(np.where(num_btag==3)[0])

print("number of events: {} \n 2 b-tagged jets: {}, {}% \n 1 b-tagged jets: {}, {}% \n 0 b-tagged jets: {}, {}% \n 3 or more b-tagged jets: {}, {}%".format(
        n_events, btag2, (float(btag2)/n_events)*100.0, btag1, (float(btag1)/n_events)*100.0, btag0, (float(btag0)/n_events)*100.0, btag3, (float(btag3)/n_events)*100.0
))

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

def plot_observables(h_jet, h_quark, wb, j=0, i=""):
    # i is the ith order leading jet, j is the number of b-tagged jets

    if wb == "W":
        label = "Hadronic W"
        caption = '1+ 2 Leading jet p_{T} vs Hadronic W p_{T}'
        obs = 'jet_v_{}_Pt'.format(wb)
    elif wb == "had_b":
        label = "Hadronic b"
        caption = 'Leading jet p_{T} vs Hadronic b p_{T}'
        obs = 'jet_v_{}_Pt'.format(wb)
    elif wb == "lep_b":
        label = "Leptonic b"
        caption = 'Leading jet p_{T} vs Leptonic b p_{T}'
        obs = 'jet_v_{}_Pt'.format(wb)

    if j:
        caption += " for {} b-tagged jets".format(j)
        obs += "_{}_btag".format(j)

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
    leg.AddEntry( h_jet, "Jet", "f" )
    leg.AddEntry( h_quark, label, "f" )
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

hist_pt_diff = infile.Get('jet12_had_W_Pt_diff')
plot_hist(hist_pt_diff, 'jet12_had_W_Pt_diff', '1+2 Leading Jet p_{T} - Hadronic W p_{T}')
for j in range(4):
    hist_pt_diff = infile.Get('jet12_had_W_Pt_{}_diff'.format(j))
    plot_hist(hist_pt_diff, 'jet12_had_W_Pt_{}_diff'.format(j), '1+2 Leading Jet p_{T} - Hadronic W p_{T}')

# plot in a separate loop because the style changes and I haven't figured out how to reverse it yet
from AngryTops.Plotting.PlottingHelper import *

hist_jet12 = infile.Get('jet12_Pt')
hist_jet_b = infile.Get('jet1_b_Pt')
Normalize(hist_jet12)
Normalize(hist_jet_b)
Normalize(hist_W)
Normalize(hist_b_had)
Normalize(hist_b_lep)

plot_observables(hist_jet12, hist_W, 'W')
plot_observables(hist_jet_b, hist_b_had, 'had_b')
plot_observables(hist_jet_b, hist_b_lep, 'lep_b')

for j in range(4):
    hist_jet12_btag = infile.Get('jet12_Pt_{}'.format(j))
    hist_W_btag = infile.Get('had_W_Pt_{}'.format(j))
    Normalize(hist_jet12_btag)
    Normalize(hist_W_btag)
    plot_observables(hist_jet12_btag, hist_W_btag, 'W', j)


