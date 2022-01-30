import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, plot_corr, find_dist, Normalize
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

ofilename = "{}/jet_filter_b_hists.root".format(output_dir)
# Open output file
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

################################################################################
# HISTOGRAMS
hists = {}

hists['true_obs_had_t_m_diff'] = TH1F("true_obs_had_t_m_diff","(rad)", 50, 0, 250)
hists['true_obs_had_t_m_diff'].SetTitle("True vs Observed Hadronic t Mass Diff;mass (GeV);A.U.")

hists['obs_had_Wb_dist'] = TH1F("obs_had_Wb_dist","(rad)", 50, 0, 6)
hists['obs_had_Wb_dist'].SetTitle("Observed Hadronic W and Closest b-tagged Jet dist; (rad);A.U.")

# had b

hists['true_had_b_Pt'] = TH1F("true_had_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['obs_had_b_Pt'] = TH1F("obs_had_b_Pt","p_{T} (GeV)", 50, 0, 500)

hists['obs_had_b_m'] = TH1F("obs_had_b_m","mass (GeV)", 50, 0, 50)
hists['obs_had_b_m'].SetTitle("Observed Hadronic b Quark mass;mass (GeV);A.U.")

hists['obs_had_t_m'] = TH1F("obs_had_t_m","mass (GeV)", 50, 0, 500)
hists['obs_had_t_m'].SetTitle("Observed Hadronic t Quark mass;mass (GeV);A.U.")

hists['true_obs_had_b_dist'] = TH1F("true_obs_had_b_dist","(rad)", 50, 0, 3)
hists['true_obs_had_b_dist'].SetTitle("True vs Observed Hadronic b dist; (rad);A.U.")

hists['true_obs_had_b_pT_diff'] = TH1F("true_obs_had_b_Pt_diff","(rad)", 50, -200, 200)
hists['true_obs_had_b_pT_diff'].SetTitle("Hadronic b True p_{T} Observed p_{T}; (GeV);A.U.")

# lep b

hists['true_lep_b_Pt'] = TH1F("true_lep_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['obs_lep_b_Pt'] = TH1F("obs_lep_b_Pt","p_{T} (GeV)", 50, 0, 500)

hists['obs_lep_b_m'] = TH1F("obs_lep_b_m","mass (GeV)", 50, 0, 50)
hists['obs_lep_b_m'].SetTitle("Observed Leptonic b Quark mass;mass (GeV);A.U.")

hists['true_obs_lep_b_dist'] = TH1F("true_obs_lep_b_dist","(rad)", 50, 0, 3)
hists['true_obs_lep_b_dist'].SetTitle("True vs Observed Leptonic b dist; (rad);A.U.")

hists['true_obs_lep_b_pT_diff'] = TH1F("true_obs_lep_b_Pt_diff","(rad)", 50, -500, 500)
hists['true_obs_lep_b_pT_diff'].SetTitle("Leptonic b True p_{T} Observed p_{T}; (GeV);A.U.")

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

jet1pt = t.AsMatrix(["jet1_pt_obs"]).flatten()
jet2pt = t.AsMatrix(["jet2_pt_obs"]).flatten()
jet3pt = t.AsMatrix(["jet3_pt_obs"]).flatten()
jet4pt = t.AsMatrix(["jet4_pt_obs"]).flatten()
jet5pt = t.AsMatrix(["jet5_pt_obs"]).flatten()

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

hadbpt = t.AsMatrix(["b_had_pt_true"]).flatten()
hadbpx = t.AsMatrix(["b_had_px_true"]).flatten()
hadbpy = t.AsMatrix(["b_had_py_true"]).flatten()
hadbpz = t.AsMatrix(['b_had_pz_true']).flatten()
hadbm = t.AsMatrix(['b_had_m_true']).flatten()

lepbpt = t.AsMatrix(["b_lep_pt_true"]).flatten()
lepbpx = t.AsMatrix(["b_lep_px_true"]).flatten()
lepbpy = t.AsMatrix(["b_lep_py_true"]).flatten()
lepbpz = t.AsMatrix(['b_lep_pz_true']).flatten()
lepbm = t.AsMatrix(['b_lep_m_true']).flatten()

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

# counter 
b_events = 0.0

for i in range(n_events):
    if ((i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    # get arrays of b-tagged and non-b-tagged jets
    # remove 0 entries in both arrays
    nonbtag_jets = np.delete(jets[i], np.where(jets_btag[i] != 0))
    btag_jets = np.delete(jets[i], np.where(jets_btag[i] == 0))

    # only look at events with 2 b-tagged jets
    b_tag_type = int(sum(jets_btag[i]))
    if b_tag_type != 2 :
        continue

    # find hadronic W
    if nonbtag_jets.size and nonbtag_jets[0].Pt() != 0: 
        # add first and second leading jet if there are 2 or more non-btagged jets
        if nonbtag_jets.size > 1:
            jet12_Pt = (nonbtag_jets[0] + nonbtag_jets[1]).Pt()
            jet12_m = (nonbtag_jets[0] + nonbtag_jets[1]).M()
            hadW_obs = nonbtag_jets[0] + nonbtag_jets[1]
        # otherwise, just use the Pt and mass fo the leading jet
        else:
            jet12_Pt = nonbtag_jets[0].Pt()
            jet12_m = nonbtag_jets[0].M()
            hadW_obs = nonbtag_jets[0]

    # find hadronic b based on shortest eta-phi distance to had W
    bW_dist = [find_dist(hadW_obs, btag_jets[0]), find_dist(hadW_obs, btag_jets[1])]
    hadb_i = bW_dist.index(min(bW_dist))
    lepb_i = bW_dist.index(max(bW_dist))

    if i<= 10:
        print(hadb_i, lepb_i)

    hadb_obs = btag_jets[hadb_i]
    lepb_obs = btag_jets[lepb_i]
    hists['obs_had_Wb_dist'].Fill(min(bW_dist))

    # dist = 1000
    # for j in range(len(btag_jets)):
    #     bjet = btag_jets[j]
    #     obs_dist = find_dist(hadW_obs, bjet)
    #     if obs_dist < dist:
    #         dist = obs_dist
    #         hadb_obs = bjet
    #         bjet_ind = j

    # lepb_obs = btag_jets[(len(btag_jets)-1)-bjet_ind]
    # hists['obs_had_Wb_dist'].Fill(dist)

    # match based on top quark mass
    # diff = 1000
    # for j in range(len(btag_jets)):
    #     bjet = btag_jets[j]
    #     obs_diff = np.abs(m_t - (hadW_obs + bjet).M())
    #     if obs_diff < diff:
    #         diff = obs_diff
    #         hadb_obs = bjet
    #         bjet_ind = j
    # lepb_obs = btag_jets[(len(btag_jets)-1)-bjet_ind]

    # hists['true_obs_had_t_m_diff'].Fill(diff)

    hadt_obs = (hadW_obs + hadb_obs)
    hadb_true = MakeP4([hadbpx[i], hadbpy[i], hadbpz[i]], hadbm[i], representation)
    lepb_true = MakeP4([lepbpx[i], lepbpy[i], lepbpz[i]], lepbm[i], representation)

    hadb_dist = find_dist(hadb_true, hadb_obs)
    hadb_pt_diff = hadb_true.Pt() - hadb_obs.Pt()
    lepb_dist = find_dist(lepb_true, lepb_obs)
    lepb_pt_diff = lepb_true.Pt() - lepb_obs.Pt()

    hists['obs_had_t_m'].Fill(hadt_obs.M())

    hists['true_had_b_Pt'].Fill( hadbpt[i] )
    hists['obs_had_b_Pt'].Fill( hadb_obs.Pt() )
    hists['obs_had_b_m'].Fill( hadb_obs.M())
    hists['true_obs_had_b_dist'].Fill( hadb_dist)
    hists['true_obs_had_b_pT_diff'].Fill( hadb_pt_diff)

    hists['true_lep_b_Pt'].Fill( lepbpt[i] )
    hists['obs_lep_b_Pt'].Fill( lepb_obs.Pt())
    hists['obs_lep_b_m'].Fill( lepb_obs.M())
    hists['true_obs_lep_b_dist'].Fill( lepb_dist)
    hists['true_obs_lep_b_pT_diff'].Fill( lepb_pt_diff)

    b_events += 1.0

################################################################################

print("number of events counted: {} out of {}, {} %".format(b_events, n_events, b_events/float(n_events)))

for histname in hists:
    hists[histname].Write(histname)

ofile.Write()
ofile.Close()

# output_dir = sys.argv[2]
# ofilename = "{}/jet_filter_hists.root".format(output_dir)
infile = TFile.Open(ofilename)

def plot_hist(h, filenames):

    c = TCanvas()
    h.Draw("H")

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    
    c.SaveAs("{0}/img/{1}.png".format(output_dir, filenames))
    # pad0.Close()
    c.Close()

def plot_observables(true, obs, hadlep):

    fname = 'true_obs_{}_b_Pt'.format(hadlep)
    if hadlep == 'had':
        caption = 'Hadronic b Quark P_{T}'
    else:
        caption = 'Leptonic b Quark P_{T}'

    # Axis titles
    xtitle = true.GetXaxis().GetTitle()
    ytitle = true.GetYaxis().SetTitle("A.U.")
    Normalize(true)
    Normalize(obs)

    # Set Style
    SetTH1FStyle( true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( obs, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    true.Draw("h")
    obs.Draw("h same")
    hmax = 1.5 * max( [ true.GetMaximum(), obs.GetMaximum() ] )
    obs.SetMaximum( hmax )
    true.SetMaximum( hmax )
    obs.SetMinimum( 0. )
    true.SetMinimum( 0. )

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( true, "True", "f" )
    leg.AddEntry( obs, "Obs", "f" )
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
    frame, tot_unc, ratio = DrawRatio(true, obs, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/img/{1}.png".format(output_dir, fname))
    pad0.Close()
    pad1.Close()
    c.Close()

try:
    os.mkdir('{}/img'.format(output_dir))
except Exception as e:
    print("Overwriting existing files")

true_had_b_Pt = infile.Get('true_had_b_Pt')
obs_had_b_Pt = infile.Get('obs_had_b_Pt')

true_lep_b_Pt = infile.Get('true_lep_b_Pt')
obs_lep_b_Pt = infile.Get('obs_lep_b_Pt')

gStyle.SetOptStat("emr")

for histname in hists:
    if 'Pt' not in histname:
        hist = infile.Get(histname)
        # Normalize(hist)
        plot_hist(hist, histname)

# plot in a separate loop because the style changes and I haven't figured out how to reverse it yet
from AngryTops.Plotting.PlottingHelper import *

plot_observables(true_had_b_Pt , obs_had_b_Pt, 'had')
plot_observables(true_lep_b_Pt , obs_lep_b_Pt, 'lep')


