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
subdir = 'b_img'
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

# truth
hists['true_had_b_Pt'] = TH1F("true_had_b_Pt", "p_{T} (GeV)", 50, 0, 500)
hists['true_lep_b_Pt'] = TH1F("true_lep_b_Pt", "p_{T} (GeV)", 50, 0, 500)

# hadronic matching
hists['h_obs_had_b_Pt'] = TH1F("h_obs_had_b_Pt", "p_{T} (GeV)", 50, 0, 500)
hists['h_obs_had_b_m'] = TH1F("h_obs_had_b_m", "Observed Hadronic b Quark mass;mass (GeV);A.U.", 50, 0, 50)
hists['h_obs_had_t_m'] = TH1F("h_obs_had_t_m", "Observed Hadronic t Quark mass;mass (GeV);A.U.", 50, 0, 500)

hists['h_obs_lep_b_Pt'] = TH1F("h_obs_lep_b_Pt", "p_{T} (GeV)", 50, 0, 500)
hists['h_obs_lep_b_m'] = TH1F("h_obs_lep_b_m", "Observed Leptonic b Quark mass;mass (GeV);A.U.", 50, 0, 50)

# leptonic matching
# hists['l_obs_had_b_Pt'] = TH1F("l_obs_had_b_Pt", "p_{T} (GeV)", 50, 0, 500)
# hists['l_obs_had_b_m'] = TH1F("l_obs_had_b_m", "Observed Hadronic b Quark mass;mass (GeV);A.U.", 50, 0, 50)

# hists['l_obs_lep_b_Pt'] = TH1F("l_obs_lep_b_Pt", "p_{T} (GeV)", 50, 0, 500)
# hists['l_obs_lep_b_m'] = TH1F("l_obs_lep_b_m", "Observed Leptonic b Quark mass;mass (GeV);A.U.", 50, 0, 50)
# hists['l_obs_lep_t_m'] = TH1F("l_obs_lep_t_m", "Observed Leptonic t Quark mass;mass (GeV);A.U.", 50, 0, 500)

# correlations
# hists['corr_hadlep_obs_t'] = TH2F( "corr_had_lep_obs_t",   ";Observed Hadronic t mass [GeV];Observed Leptonic t mass [GeV]", 50, 50., 500., 50, 50., 500. )

################################################################################
# GET VALUES FROM TREE

n_events = t.GetEntries()

# jets, lepton, neutrino
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

mupx = t.AsMatrix(["jetmu_px_obs"]).flatten()
mupy = t.AsMatrix(["jetmu_py_obs"]).flatten()
mupz = t.AsMatrix(["jetmu_pz_obs"]).flatten()
muT0 = t.AsMatrix(["jetmu_T0_obs"])
lepEt = t.AsMatrix(["jetlep_ET_obs"]).flatten()
lepphi = t.AsMatrix(["jetlep_phi_obs"]).flatten()

# jets = np.array(jets)
jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)

# truth values
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

################################################################################
# FILL HISTS WITH LEADING B-TAGGED AND NON B-TAGGED JETS 

# counter 
b_events = 0.0
matches = [0.0,0.0,0.0,0.0]

for i in range(n_events):
    if ((i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    jet1 = MakeP4( [jet1px[i], jet1py[i], jet1pz[i]] , jet1m[i], representation)
    jet2 = MakeP4( [jet2px[i], jet2py[i], jet2pz[i]] , jet2m[i], representation)
    jet3 = MakeP4( [jet3px[i], jet3py[i], jet3pz[i]] , jet3m[i], representation)
    jet4 = MakeP4( [jet4px[i], jet4py[i], jet4pz[i]] , jet4m[i], representation)
    jet5 = MakeP4( [jet5px[i], jet5py[i], jet5pz[i]] , jet5m[i], representation)
    jets = [jet1, jet2, jet3, jet4, jet5]
    mu = MakeP4( [mupx[i], mupy[i], mupz[i]] , 0.10566, representation)#muT0[i], representation) 105.66 
    lep = MakeP4( [lepEt[i]*np.cos(lepphi[i]), lepEt[i]*np.sin(lepphi[i]), 0], 0 , representation)
    leptons = lep + mu
    hadb_true = MakeP4([hadbpx[i], hadbpy[i], hadbpz[i]], hadbm[i], representation)
    lepb_true = MakeP4([lepbpx[i], lepbpy[i], lepbpz[i]], lepbm[i], representation)

    # get arrays of b-tagged and non-b-tagged jets
    # remove 0 entries in both arrays
    nonbtag_jets = np.delete(jets, np.where(jets_btag[i] != 0))
    btag_jets = np.delete(jets, np.where(jets_btag[i] == 0))

    # only look at events with 2 b-tagged jets
    b_tag_type = int(sum(jets_btag[i]))
    if b_tag_type != 2 :
        continue

    
    # hadronic matching
    # first match jets to Had W
    # add first and second leading jet if there are 2 or more non-btagged jets
    if nonbtag_jets.size > 1:
        hadW_obs = nonbtag_jets[0] + nonbtag_jets[1]
        match = 0
    # otherwise, just use the Pt and mass fo the leading jet
    else:
        hadW_obs = nonbtag_jets[0]
        match = 3
    # check than 3rd leading jet is also non-zero
    if (nonbtag_jets.size > 2) and (nonbtag_jets[2].Pt() != 0):
        if (hadW_obs.M() < W_had_m_cutoff[0]) or (hadW_obs.M() > W_had_m_cutoff[1]):
            hadW_obs = nonbtag_jets[0] + nonbtag_jets[2]
            match = 1

        if (hadW_obs.M() < W_had_m_cutoff[0]) or (hadW_obs.M() > W_had_m_cutoff[1]):
            hadW_obs = nonbtag_jets[1] + nonbtag_jets[2]
            match = 2
    
    # keep track of the number of each match type
    matches[match] +=1.0
    # find hadronic b based on shortest eta-phi distance to had W
    bW_dist = [find_dist(hadW_obs, btag_jets[0]), find_dist(hadW_obs, btag_jets[1])]
    hadb_i = bW_dist.index(min(bW_dist))
    lepb_i = 1-hadb_i
    hadb_obs = btag_jets[hadb_i]
    lepb_obs = btag_jets[lepb_i]
    hadt_obs = hadW_obs + hadb_obs

    hists['h_obs_had_t_m'].Fill( hadt_obs.M() )
    hists['h_obs_had_b_Pt'].Fill( hadb_obs.Pt() )
    hists['h_obs_had_b_m'].Fill( hadb_obs.M())
    hists['h_obs_lep_b_Pt'].Fill( lepb_obs.Pt())
    hists['h_obs_lep_b_m'].Fill( lepb_obs.M())

    
    # leptonic matching
    # find mass difference between top and leptons+b
    '''
    lept_m_diff = [np.abs(m_t - (leptons + btag_jets[0]).M()) , np.abs(m_t - (leptons + btag_jets[1]).M())]
    lepb_i = lept_m_diff.index(min(lept_m_diff))
    hadb_i = 1-lepb_i
    hadb_obs = btag_jets[hadb_i]
    lepb_obs = btag_jets[lepb_i]
    lept_obs = leptons + lepb_obs

    hists['l_obs_lep_t_m'].Fill( lept_obs.M() )
    hists['l_obs_had_b_Pt'].Fill( hadb_obs.Pt() )
    hists['l_obs_had_b_m'].Fill( hadb_obs.M())
    hists['l_obs_lep_b_Pt'].Fill( lepb_obs.Pt())
    hists['l_obs_lep_b_m'].Fill( lepb_obs.M())
    '''
    # hists['corr_hadlep_obs_t'].Fill( hadt_obs.M(), lept_obs.M(), 1.0 )
    # truth
    hists['true_lep_b_Pt'].Fill( lepbpt[i] )
    hists['true_had_b_Pt'].Fill( hadbpt[i] )
    
    # b_events += 1.0


################################################################################

print("number of events counted: {} out of {}, {} %".format(b_events, n_events, b_events/float(n_events)))

for j in range(len(matches)):
    if j == 0:
        match_type = '12'    
    elif j == 1:
        math_type = '13'
    elif j == 2:
        match_type = '23'
    elif j == 3:
        match_type = '1'
    # print("num {} jet matches to Hadronic W: {}, {}% of total 2 b-tagged jet events".format(match_type, matches[j], float(matches[j])/float(sum(matches))*100.0 ))
print('testing')

for histname in hists:
    hists[histname].Write(histname)

ofile.Write()
ofile.Close()

# output_dir = sys.argv[2]
# ofilename = "{}/jet_filter_b_hists.root".format(output_dir)
# list of histograms to be formatted in the first style
hists = ['h_obs_had_b_m','h_obs_had_t_m','h_obs_lep_b_Pt']#,'l_obs_had_b_m', 'l_obs_had_b_m','l_obs_lep_t_m']

infile = TFile.Open(ofilename)

print(ofilename)

def plot_hist(h, filenames):

    c = TCanvas()
    h.Draw("H")

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    
    c.SaveAs("{0}/b_img/{1}.png".format(output_dir, filenames))
    # pad0.Close()
    c.Close()

def plot_observables(true, obs, fname):

    if 'had_b' in fname:
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

    c.SaveAs("{0}/b_img/{1}.png".format(output_dir, fname))
    pad0.Close()
    pad1.Close()
    c.Close()

def plot_correlations(hist, fname, caption = None):
    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    #pad0.SetTopMargin( 0.14 )
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

    c.SaveAs("{0}/{1}/{2}.png".format(output_dir, subdir, fname))
    pad0.Close()
    c.Close()

try:
    os.mkdir('{}/{}'.format(output_dir, subdir))
except Exception as e:
    print("Overwriting existing files")

true_had_b_Pt = infile.Get('true_had_b_Pt')
true_lep_b_Pt = infile.Get('true_lep_b_Pt')

h_obs_had_b_Pt = infile.Get('h_obs_had_b_Pt')
h_obs_lep_b_Pt = infile.Get('h_obs_lep_b_Pt')

# l_obs_had_b_Pt = infile.Get('l_obs_had_b_Pt')
# l_obs_lep_b_Pt = infile.Get('l_obs_lep_b_Pt')

# corr_hadlep_obs_t = infile.Get('corr_hadlep_obs_t')

gStyle.SetOptStat("emr")

for histname in hists:
    hist = infile.Get(histname)
    # Normalize(hist)
    plot_hist(hist, histname)

# plot in a separate loop because the style changes and I haven't figured out how to reverse it yet
from AngryTops.Plotting.PlottingHelper import *

plot_observables(true_had_b_Pt , h_obs_had_b_Pt, 'h_true_obs_had_b_Pt')
plot_observables(true_lep_b_Pt , h_obs_lep_b_Pt, 'h_true_obs_lep_b_Pt')
# plot_observables(true_had_b_Pt , l_obs_had_b_Pt, 'l_true_obs_had_b_Pt')
# plot_observables(true_lep_b_Pt , l_obs_lep_b_Pt, 'l_true_obs_lep_b_Pt')

gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

# plot_correlations(corr_hadlep_obs_t, 'corr_hadlep_obs_t')


