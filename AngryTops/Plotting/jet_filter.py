import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4
from AngryTops.features import *
from array import array

################################################################################
# CONSTANTS
infilename = sys.argv[1]
output_dir = sys.argv[2]
representation = sys.argv[3]
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

W_had_m_cutoff = (30, 130)

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
# HISTOGRAMS
hists = {}
hists['had_W_Pt'] = TH1F("had_W_Pt","p_{T} (GeV)", 50, 0, 500)
hists['had_b_Pt'] = TH1F("had_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['lep_b_Pt'] = TH1F("lep_b_Pt","p_{T} (GeV)", 50, 0, 500)

hists['obs_jet_had_W_Pt_diff'] = TH1F("obs_jet_had_W_Pt_diff","p_{T} (GeV)", 50, -300, 300)
hists['obs_jet_had_W_Pt_diff'].SetTitle("Leading Jets p_{T} - Had W p_{T}; p_{T} (GeV);A.U.")

hists['obs_jet_Pt'] = TH1F("obs_jet_Pt","p_{T} (GeV)", 50, 0, 500)
hists['obs_jet_Pt'].SetTitle("Leading Jets p_{T}; p_{T} (GeV);A.U.")

hists['obs_jet_m'] = TH1F("obs_jet_m","mass (GeV)", 50, 0, 250)
hists['obs_jet_m'].SetTitle("Leading Jets mass; mass (GeV);A.U.")

# leading b tagged jet
hists['jet1_b_Pt'] = TH1F("jet1_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['jet1_b_m'] = TH1F("jet1_b_m","mass (GeV)", 50, 0, 250)
hists['jet1_b_m'].SetTitle("1 b-tagged Leading Jet mass;mass (GeV);A.U.")

# b-tagging categories
# had_W_Pt_3 includes events with 3 or more b-tagged jets
for j in range(4):
    hists['had_W_Pt_{}'.format(j)] = TH1F("had_W_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    
    hists['obs_jet_Pt_{}'.format(j)] = TH1F("obs_jet_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    hists['obs_jet_Pt_{}'.format(j)].SetTitle("Leading Jets p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")
    
    hists['obs_jet_m_{}'.format(j)] = TH1F("obs_jet_m_{}".format(j),"m (GeV)", 50, 0, 250)
    hists['obs_jet_m_{}'.format(j)].SetTitle("Leading Jets mass, " + "{} b-tagged jets".format(j) + "; mass (GeV);A.U.")

    hists['obs_jet_had_W_Pt_{}_diff'.format(j)] = TH1F("obs_jet_had_W_Pt_{}_diff".format(j),"p_{T} (GeV)", 50, -300, 300)
    hists['obs_jet_had_W_Pt_{}_diff'.format(j)].SetTitle("Leading Jets p_{T} - Had W p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")

################################################################################
# GET VALUES FROM TREE
jet1px = t.AsMatrix(["jet1_px_obs"]).flatten()
jet2px = t.AsMatrix(["jet2_px_obs"]).flatten()
jet3px = t.AsMatrix(["jet3_px_obs"]).flatten()
jet4px = t.AsMatrix(["jet4_px_obs"]).flatten()
jet5px = t.AsMatrix(["jet5_px_obs"]).flatten()

jet1py = t.AsMatrix(["jet1_py_obs"]).flatten()
jet2py = t.AsMatrix(["jet2_py_obs"]).flatten()
jet3py = t.AsMatrix(["jet3_py_obs"]).flatten()
jet4py = t.AsMatrix(["jet4_py_obs"]).flatten()
jet5py = t.AsMatrix(["jet5_py_obs"]).flatten()

jet1pz = t.AsMatrix(["jet1_pz_obs"]).flatten()
jet2pz = t.AsMatrix(["jet2_pz_obs"]).flatten()
jet3pz = t.AsMatrix(["jet3_pz_obs"]).flatten()
jet4pz = t.AsMatrix(["jet4_pz_obs"]).flatten()
jet5pz = t.AsMatrix(["jet5_pz_obs"]).flatten()

jet1p = np.stack([jet1px, jet1py, jet1pz], axis=1)
jet2p = np.stack([jet2px, jet2py, jet2pz], axis=1)
jet3p = np.stack([jet3px, jet3py, jet3pz], axis=1)
jet4p = np.stack([jet4px, jet4py, jet4pz], axis=1)
jet5p = np.stack([jet5px, jet5py, jet5pz], axis=1)

jet1m = t.AsMatrix(["jet1_m_obs"]).flatten()
jet2m = t.AsMatrix(["jet2_m_obs"]).flatten()
jet3m = t.AsMatrix(["jet3_m_obs"]).flatten()
jet4m = t.AsMatrix(["jet4_m_obs"]).flatten()
jet5m = t.AsMatrix(["jet5_m_obs"]).flatten()

jet1btag = t.AsMatrix(["jet1_btag_obs"]).flatten()
jet2btag = t.AsMatrix(["jet2_btag_obs"]).flatten()
jet3btag = t.AsMatrix(["jet3_btag_obs"]).flatten()
jet4btag = t.AsMatrix(["jet4_btag_obs"]).flatten()
jet5btag = t.AsMatrix(["jet5_btag_obs"]).flatten()

hadWpt = t.AsMatrix(["W_had_pt_true"]).flatten()
hadbpt = t.AsMatrix(["b_had_pt_true"]).flatten()
lepbpt = t.AsMatrix(["b_lep_pt_true"]).flatten()

n_events = t.GetEntries()

jets = []
for i in range(n_events):
    jet1 = MakeP4(jet1p[i], jet1m[i], representation)
    jet2 = MakeP4(jet2p[i], jet2m[i], representation)
    jet3 = MakeP4(jet3p[i], jet3m[i], representation)
    jet4 = MakeP4(jet4p[i], jet4m[i], representation)
    jet5 = MakeP4(jet5p[i], jet5m[i], representation)
    jets.append([jet1, jet2, jet3, jet4, jet5])
jets = np.array(jets)

jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)

################################################################################
# FILL HISTS WITH LEADING B-TAGGED AND NON B-TAGGED JETS 

# jet1,2,3,4,5 are already sorted in order of leading jet based on how the jets were stored when data was generated
# jet1 is first leading jet, jet2 the second, etc...

# btag type: 0,1,2,3
# 12, 13, 23, 1
matches = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]

for i in range(n_events):
    if ((i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    # get arrays of b-tagged and non-b-tagged jets
    # remove 0 entries in both arrays
    nonbtag_jets = np.delete(jets[i], np.where(jets_btag[i] != 0))
    btag_jets = np.delete(jets[i], np.where(jets_btag[i] == 0))

    if btag_jets.size and btag_jets[0].Pt() != 0: # skip events where the leading jet has 0 pT
        hists['had_b_Pt'].Fill( hadbpt[i] )
        hists['lep_b_Pt'].Fill( lepbpt[i] )
        hists['jet1_b_Pt'].Fill(btag_jets[0].Pt())
        hists['jet1_b_m'].Fill(btag_jets[0].M())

    b_tag_type = int(sum(jets_btag[i]))
    if b_tag_type >= 3:
        b_tag_type = 3

    # first look at 1+2 leading jets, then 1+3 or 2+3
    # and condition comes first so there isn't an error when trying to index nonbtag_jets[0]
    if nonbtag_jets.size and (nonbtag_jets[0].Pt() != 0): 
        # add first and second leading jet if there are 2 or more non-btagged jets
        if nonbtag_jets.size > 1:
            obs_jet_Pt = (nonbtag_jets[0] + nonbtag_jets[1]).Pt()
            obs_jet_m = (nonbtag_jets[0] + nonbtag_jets[1]).M()
            match = 0
        # otherwise, just use the Pt and mass fo the leading jet
        else:
            obs_jet_Pt = nonbtag_jets[0].Pt()
            obs_jet_m = nonbtag_jets[0].M()
            match = 3

        if (nonbtag_jets.size > 2) and (nonbtag_jets[2].Pt() != 0): # implies Pt of [0] and [1] are also non zero
            if (obs_jet_m < W_had_m_cutoff[0]) or (obs_jet_m > W_had_m_cutoff[1]):
                obs_jet_Pt = (nonbtag_jets[0] + nonbtag_jets[2]).Pt()
                obs_jet_m = (nonbtag_jets[0] + nonbtag_jets[2]).M()
                match = 1

            if (obs_jet_m < W_had_m_cutoff[0]) or (obs_jet_m > W_had_m_cutoff[1]):
                obs_jet_Pt = (nonbtag_jets[1] + nonbtag_jets[2]).Pt()
                obs_jet_m = (nonbtag_jets[1] + nonbtag_jets[2]).M()
                match = 2

        hadW_Pt = hadWpt[i]
        matches[b_tag_type][match] +=1

        # if obs_jet_m > W_had_m_cutoff[0] and obs_jet_m < W_had_m_cutoff[1]:
        hists['obs_jet_had_W_Pt_diff'].Fill( obs_jet_Pt - hadW_Pt )
        hists['obs_jet_had_W_Pt_{}_diff'.format(b_tag_type)].Fill( obs_jet_Pt - hadW_Pt )
        
        hists['obs_jet_Pt'].Fill( obs_jet_Pt )
        hists['obs_jet_Pt_{}'.format(b_tag_type)].Fill( obs_jet_Pt )
        hists['obs_jet_m'].Fill( obs_jet_m )
        hists['obs_jet_m_{}'.format(b_tag_type)].Fill( obs_jet_m )

        hists['had_W_Pt'].Fill( hadW_Pt )
        hists['had_W_Pt_{}'.format(b_tag_type)].Fill( hadW_Pt )

################################################################################

# get number of events in each b-tagging category
num_btag = np.sum(jets_btag, axis = 1)
btag0 = len(np.where(num_btag==0)[0])
btag1 = len(np.where(num_btag==1)[0])
btag2 = len(np.where(num_btag==2)[0])
btag3 = len(np.where(num_btag==3)[0])

print("number of events: {} \n 2 b-tagged jets: {}, {}% \n 1 b-tagged jets: {}, {}% \n 0 b-tagged jets: {}, {}% \n 3 or more b-tagged jets: {}, {}%".format(
        n_events, btag2, (float(btag2)/n_events)*100.0, btag1, (float(btag1)/n_events)*100.0, btag0, (float(btag0)/n_events)*100.0, btag3, (float(btag3)/n_events)*100.0
))

for i in range(len(matches)):
    print('btag jets: {}'.format(i))
    for j in range(len(matches[i])):
        if j == 0:
            match_type = '12'    
        elif j == 1:
            math_type = '13'
        elif j == 2:
            match_type = '23'
        elif j == 3:
            match_type = '1'
        print("num {} jet matches: {}, {}% of total 2 b-tagged jet events".format(match_type, matches[i][j], float(matches[i][j])/float(sum(matches[i]))*100.0 ))

for histname in hists:
    hists[histname].Write(histname)

ofile.Write()
ofile.Close()

# output_dir = sys.argv[2]
# ofilename = "{}/jet_filter_hists.root".format(output_dir)
infile = TFile.Open(ofilename)

def plot_hist(h, obs):

    c = TCanvas()
    h.Draw("H")

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    
    c.SaveAs("{0}/img/{1}.png".format(output_dir, obs))
    # pad0.Close()
    c.Close()

def plot_observables(h_jet, h_quark, wb, j=0, i = "12"):
    # i is the ith order leading jet, j is the number of b-tagged jets

    if wb == "W":
        label = "Hadronic W"
        caption = '{}+{}'.format(i[0], i[1])+ ' Leading jet p_{T} vs Hadronic W p_{T}'
        obs = 'jet_v_{}_{}_Pt'.format(wb, i)
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


jet1_b_m = infile.Get("jet1_b_m")
plot_hist(jet1_b_m, "jet1_b_m")

hist_pt_diff = infile.Get('obs_jet_had_W_Pt_diff')
plot_hist(hist_pt_diff, 'obs_jet_had_W_Pt_diff')
obs_jet_mass = infile.Get('obs_jet_m')
plot_hist(obs_jet_mass, 'obs_jet_had_W_m')
for j in range(4):
    hist_pt_diff = infile.Get('obs_jet_had_W_Pt_{}_diff'.format(j))
    plot_hist(hist_pt_diff, 'obs_jet_had_W_Pt_{}_diff'.format(j))
    obs_jet_mass = infile.Get('obs_jet_m_{}'.format(j))
    plot_hist(obs_jet_mass, 'obs_jet_had_W_m_{}'.format(j))

# plot in a separate loop because the style changes and I haven't figured out how to reverse it yet
from AngryTops.Plotting.PlottingHelper import *

hist_obs_jet = infile.Get('obs_jet_Pt')
hist_jet_b = infile.Get('jet1_b_Pt')
Normalize(hist_obs_jet)
Normalize(hist_jet_b)
Normalize(hist_W)
Normalize(hist_b_had)
Normalize(hist_b_lep)

plot_observables(hist_obs_jet, hist_W, 'W')
plot_observables(hist_jet_b, hist_b_had, 'had_b')
plot_observables(hist_jet_b, hist_b_lep, 'lep_b')

for j in range(4):
    hist_obs_jet_btag = infile.Get('obs_jet_Pt_{}'.format(j))
    hist_W_btag = infile.Get('had_W_Pt_{}'.format(j))
    Normalize(hist_obs_jet_btag)
    Normalize(hist_W_btag)
    plot_observables(hist_obs_jet_btag, hist_W_btag, 'W', str(j))
