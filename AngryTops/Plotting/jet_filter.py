import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, plot_corr
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

hists['jet12_had_W_Pt_diff'] = TH1F("jet12_had_W_Pt_diff","p_{T} (GeV)", 50, -300, 300)
hists['jet12_had_W_Pt_diff'].SetTitle("1+2 Leading Jet p_{T} - Had W p_{T}; p_{T} (GeV);A.U.")

hists['jet12_Pt'] = TH1F("jet12_Pt","p_{T} (GeV)", 50, 0, 500)
hists['jet12_Pt'].SetTitle("1+2 Leading Jet p_{T}; p_{T} (GeV);A.U.")

hists['jet12_m'] = TH1F("jet12_m","mass (GeV)", 50, 0, 250)
hists['jet12_m'].SetTitle("1+2 Leading Jet mass; mass (GeV);A.U.")

# leading b tagged jet
hists['jet1_b_Pt'] = TH1F("jet1_b_Pt","p_{T} (GeV)", 50, 0, 500)
hists['jet1_b_m'] = TH1F("jet1_b_m","mass (GeV)", 50, 0, 50)
hists['jet1_b_m'].SetTitle("1 b-tagged Leading Jet mass;mass (GeV);A.U.")

# b-tagging categories
# had_W_Pt_3 includes events with 3 or more b-tagged jets
for j in range(4):
    hists['had_W_Pt_{}'.format(j)] = TH1F("had_W_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    
    hists['jet12_Pt_{}'.format(j)] = TH1F("jet12_Pt_{}".format(j),"p_{T} (GeV)", 50, 0, 500)
    hists['jet12_Pt_{}'.format(j)].SetTitle("1+2 Leading Jet p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")
    
    hists['jet12_m_{}'.format(j)] = TH1F("jet12_m_{}".format(j),"m (GeV)", 50, 0, 250)
    hists['jet12_m_{}'.format(j)].SetTitle("1+2 Leading Jet mass, " + "{} b-tagged jets".format(j) + "; mass (GeV);A.U.")

    hists['jet12_had_W_Pt_{}_diff'.format(j)] = TH1F("jet12_had_W_Pt_{}_diff".format(j),"p_{T} (GeV)", 50, -300, 300)
    hists['jet12_had_W_Pt_{}_diff'.format(j)].SetTitle("1+2 Leading Jet p_{T} - Had W p_{T}, " + "{} b-tagged jets".format(j) + "; p_{T} (GeV);A.U.")

hists['corr_W_had_pt']    = TH2F( "corr_W_had_pt",      ";True Hadronic W p_{T} [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
hists['corr_W_had_px']    = TH2F( "corr_W_had_px",      ";True Hadronic W p_{x} [GeV];Predicted Hadronic W p_{x} [GeV]", 50, -300., 300., 50, -300., 300. )
hists['corr_W_had_py']    = TH2F( "corr_W_had_py",      ";True Hadronic W p_{y} [GeV];Predicted Hadronic W p_{y} [GeV]", 50, -300., 300., 50, -300., 300. )
hists['corr_W_had_pz']    = TH2F( "corr_W_had_pz",      ";True Hadronic W p_{z} [GeV];Predicted Hadronic W p_{z} [GeV]", 50, -400., 400., 50, -400., 400. )
hists['corr_W_had_y']     = TH2F( "corr_W_had_y",       ";True Hadronic W y;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
hists['corr_W_had_phi']   = TH2F( "corr_W_had_phi",     ";True Hadronic W #phi;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
hists['corr_W_had_E']     = TH2F( "corr_W_had_E",       ";True Hadronic W E [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )
hists['corr_W_had_m']     = TH2F( "corr_W_had_m",       ";True Hadronic W m [GeV];Predicted Hadronic top m [GeV]", 25, 170., 175., 20, 150., 250. )

hists['corr_W_had_pt_1']    = TH2F( "corr_W_had_pt_1",      ";True Hadronic W p_{T} 1-btag jet [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
hists['corr_W_had_y_1']     = TH2F( "corr_W_had_y_1",       ";True Hadronic W y 1-btag jet;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
hists['corr_W_had_phi_1']   = TH2F( "corr_W_had_phi_1",     ";True Hadronic W #phi 1-btag jet ;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
hists['corr_W_had_E_1']     = TH2F( "corr_W_had_E_1",       ";True Hadronic W E 1-btag jet [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )

hists['corr_W_had_pt_2']    = TH2F( "corr_W_had_pt_2",      ";True Hadronic W p_{T} 2-btag jets [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
hists['corr_W_had_y_2']     = TH2F( "corr_W_had_y_2",       ";True Hadronic W y 2-btag jets ;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
hists['corr_W_had_phi_2']   = TH2F( "corr_W_had_phi_2",     ";True Hadronic W #phi 2-btag jets ;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
hists['corr_W_had_E_2']     = TH2F( "corr_W_had_E_2",       ";True Hadronic W E 2-btag jets [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )

hists['corr_W_had_pt_0']    = TH2F( "corr_W_had_pt_0",      ";True Hadronic W p_{T} 0-btag jet [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
hists['corr_W_had_y_0']     = TH2F( "corr_W_had_y_0",       ";True Hadronic W y 0-btag jet ;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
hists['corr_W_had_phi_0']   = TH2F( "corr_W_had_phi_0",     ";True Hadronic W #phi 0-btag jet ;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
hists['corr_W_had_E_0']     = TH2F( "corr_W_had_E_0",       ";True Hadronic W E 0-btag jet [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )

hists['corr_W_had_pt_3']    = TH2F( "corr_W_had_pt_3",      ";True Hadronic W p_{T} 3-btag jets [GeV];Predicted Hadronic W p_{T} [GeV]", 50, 0., 300., 50, 0., 300. )
hists['corr_W_had_y_3']     = TH2F( "corr_W_had_y_3",       ";True Hadronic W y 3-btag jets ;Predicted Hadronic top y", 25, -5., 5., 25, -5., 5. )
hists['corr_W_had_phi_3']   = TH2F( "corr_W_had_phi_3",     ";True Hadronic W #phi 3-btag jets ;Predicted Hadronic top #phi", 16, -3.2, 3.2, 16, -3.2, 3.2 )
hists['corr_W_had_E_3']     = TH2F( "corr_W_had_E_3",       ";True Hadronic W E 3-btag jets [GeV];Predicted Hadronic top E [GeV]", 50, 70., 400., 50, 70., 400. )

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

hadWpt = t.AsMatrix(["W_had_pt_true"]).flatten()
hadbpt = t.AsMatrix(["b_had_pt_true"]).flatten()
lepbpt = t.AsMatrix(["b_lep_pt_true"]).flatten()

hadWpx = t.AsMatrix(["W_had_px_true"]).flatten()
hadWpy = t.AsMatrix(["W_had_py_true"]).flatten()
hadWpz = t.AsMatrix(['W_had_pz_true']).flatten()
hadWE = t.AsMatrix(['W_had_E_true']).flatten()
hadWm = t.AsMatrix(['W_had_m_true']).flatten()
hadWy = t.AsMatrix(['W_had_y_true']).flatten()
hadWphi = t.AsMatrix(['W_had_phi_true']).flatten()

hadWpx_fit = t.AsMatrix(["W_had_px_fitted"]).flatten()
hadWpy_fit = t.AsMatrix(["W_had_py_fitted"]).flatten()
hadWpz_fit = t.AsMatrix(['W_had_pz_fitted']).flatten()
hadWE_fit = t.AsMatrix(['W_had_E_fitted']).flatten()
hadWpt_fit = t.AsMatrix(['W_had_pt_fitted']).flatten()
hadWm_fit = t.AsMatrix(['W_had_m_fitted']).flatten()
hadWy_fit = t.AsMatrix(['W_had_y_fitted']).flatten()
hadWphi_fit = t.AsMatrix(['W_had_phi_fitted']).flatten()

n_events = t.GetEntries()

jets = []
for i in range(n_events):
    jet1 = MakeP4(jet1p[i], jet1m[i], representation)
    jet2 = MakeP4(jet2p[i], jet2m[i], representation)
    jet3 = MakeP4(jet3p[i], jet3m[i], representation)
    jet4 = MakeP4(jet4p[i], jet4m[i], representation)
    jet5 = MakeP4(jet5p[i], jet5m[i], representation)
    jets.append([jet1, jet2, jet3, jet4, jet5])

jets_btag = np.stack([jet1btag, jet2btag, jet3btag, jet4btag, jet5btag], axis = 1)

################################################################################
# FIND B-TAGGED AND NON B-TAGGED ROWS

# jet1,2,3,4,5 are already sorted in order of leading jet based on how the jets were stored when data was generated
# jet1 is first leading jet, jet2 the second, etc...

# as a check, run:
# np.where(jet1pt < jet2pt)
# np.where(jet2pt < jet3pt)
# np.where(jet3pt < jet4pt)
# np.where(jet4pt < jet5pt)

# get arrays of b-tagged and non-b-tagged jets
jets_nonb = np.array(jets)
jets_b = np.array(jets)

for i in range(n_events):
    if ((i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

    # remove 0 entries in both arrays
    nonbtag_jets = np.delete(jets_nonb[i], np.where(jets_btag[i] != 0))
    btag_jets = np.delete(jets_b[i], np.where(jets_btag[i] == 0))

    # leading jet = nonbtag_jets[0]
    # second leading jet = nonbtag_jets[1]

    if btag_jets.size and btag_jets[0].Pt() != 0: # skip events where the leading jet has 0 pT
        hists['had_b_Pt'].Fill( hadbpt[i] )
        hists['lep_b_Pt'].Fill( lepbpt[i] )
        hists['jet1_b_Pt'].Fill(btag_jets[0].Pt())
        hists['jet1_b_m'].Fill(btag_jets[0].M())

    if nonbtag_jets.size and nonbtag_jets[0].Pt() != 0: 
        if nonbtag_jets.size > 1:
            jet12_Pt = (nonbtag_jets[0] + nonbtag_jets[1]).Pt()
            jet12_m = (nonbtag_jets[0] + nonbtag_jets[1]).M()
        else:
            jet12_Pt = nonbtag_jets[0].Pt()
            jet12_m = nonbtag_jets[0].M()

        b_tag_type = int(sum(jets_btag[i]))
        if b_tag_type >= 3: # set to 3 if greater or equal to 3 for easier indexing 
            b_tag_type = 3

        hadW_Pt = hadWpt[i]

        # if jet12_m > W_had_m_cutoff[0] and jet12_m < W_had_m_cutoff[1]:
        hists['jet12_had_W_Pt_diff'].Fill( jet12_Pt - hadW_Pt )
        hists['jet12_had_W_Pt_{}_diff'.format(b_tag_type)].Fill( jet12_Pt - hadW_Pt )
        
        hists['jet12_Pt'].Fill( jet12_Pt )
        hists['jet12_Pt_{}'.format(b_tag_type)].Fill( jet12_Pt )
        hists['jet12_m'].Fill( jet12_m )
        hists['jet12_m_{}'.format(b_tag_type)].Fill( jet12_m )

        hists['had_W_Pt'].Fill( hadW_Pt )
        hists['had_W_Pt_{}'.format(b_tag_type)].Fill( hadW_Pt )

        # plot predicted
        # if jet12_m > W_had_m_cutoff[0] and jet12_m < W_had_m_cutoff[1]:
        w = 1
        hists['corr_W_had_pt'].Fill(  hadWpt[i],  hadWpt_fit[i], w)
        hists['corr_W_had_px'].Fill(  hadWpx[i],  hadWpx_fit[i], w)
        hists['corr_W_had_py'].Fill(  hadWpy[i],  hadWpy_fit[i],  w )
        hists['corr_W_had_pz'].Fill(  hadWpz[i],  hadWpz_fit[i],  w )
        hists['corr_W_had_y'].Fill(   hadWy[i],  hadWy_fit[i], w )
        hists['corr_W_had_phi'].Fill( hadWphi[i],  hadWphi_fit[i], w )
        hists['corr_W_had_E'].Fill(   hadWE[i],  hadWE_fit[i],   w )
        hists['corr_W_had_m'].Fill(   hadWm[i],  hadWm_fit[i],   w )

        hists['corr_W_had_pt_{}'.format(b_tag_type)].Fill(  hadWpt[i],  hadWpt_fit[i], w)
        hists['corr_W_had_y_{}'.format(b_tag_type)].Fill(   hadWy[i],  hadWy_fit[i], w )
        hists['corr_W_had_phi_{}'.format(b_tag_type)].Fill( hadWphi[i],  hadWphi_fit[i], w )
        hists['corr_W_had_E_{}'.format(b_tag_type)].Fill(   hadWE[i],  hadWE_fit[i],   w )

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


jet1_b_m = infile.Get("jet1_b_m")
plot_hist(jet1_b_m, "jet1_b_m")

hist_pt_diff = infile.Get('jet12_had_W_Pt_diff')
plot_hist(hist_pt_diff, 'jet12_had_W_Pt_diff')
jet12_mass = infile.Get('jet12_m')
plot_hist(jet12_mass, 'jet12_had_W_m')
for j in range(4):
    hist_pt_diff = infile.Get('jet12_had_W_Pt_{}_diff'.format(j))
    plot_hist(hist_pt_diff, 'jet12_had_W_Pt_{}_diff'.format(j))
    jet12_mass = infile.Get('jet12_m_{}'.format(j))
    plot_hist(jet12_mass, 'jet12_had_W_m_{}'.format(j))

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
    plot_observables(hist_jet12_btag, hist_W_btag, 'W', str(j))

gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

for name in hists:
    if 'corr' in name:
        corr = infile.Get(name)
        plot_corr(name, corr, output_dir+"/img/")