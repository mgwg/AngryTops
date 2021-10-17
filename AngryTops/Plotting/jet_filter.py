import os, sys
import numpy as np
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4
from AngryTops.features import *

################################################################################
# CONSTANTS
training_dir = sys.argv[1]
output_dir = sys.argv[2]
representation = sys.argv[3]
scaling = True              # whether the dataset has been passed through a scaling function or not

m_t = 172.5
m_W = 80.4
m_b = 4.95

ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE   

################################################################################
# load data

# print("INFO: fitting ttbar decay chain...")
# predictions = np.load(training_dir + 'predictions.npz')
# jets = predictions['input']
# true = predictions['true']

# particles_shape = (true.shape[1], true.shape[2])
# print("jets shape", jets.shape)
# print("b tagging option", b_tagging)
# if scaling:
#     scaler_filename = training_dir + "scalers.pkl"
#     with open( scaler_filename, "rb" ) as file_scaler:
#         jets_scalar = pickle.load(file_scaler)
#         lep_scalar = pickle.load(file_scaler)
#         output_scalar = pickle.load(file_scaler)
#         # Rescale the truth array
#         true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
#         true = output_scalar.inverse_transform(true)
#         true = true.reshape(true.shape[0], particles_shape[0], particles_shape[1])
#         # Rescale the jets array
#         jets_lep = jets[:,:6]
#         jets_jets = jets[:,6:] # remove muon column
#         jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array

#         # Remove the b-tagging states and put them into a new array to be re-appended later.
#         b_tags = jets_jets[:,:,5]
#         jets_jets = np.delete(jets_jets, 5, 2) # delete the b-tagging states

#         jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) # reshape into 25 element long array
#         jets_lep = lep_scalar.inverse_transform(jets_lep)
#         jets_jets = jets_scalar.inverse_transform(jets_jets) # scale values ... ?
#         #I think this is the final 6x6 array the arxiv paper was talking about - 5 x 5 array containing jets (1 per row) and corresponding px, py, pz, E, m
#         jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))
#         # Re-append the b-tagging states as a column at the end of jets_jets 
#         jets_jets = np.append(jets_jets, np.expand_dims(b_tags, 2), 2)

# if not scaling:
#     jets_lep = jets[:,:6]
#     jets_jets = jets[:,6:]
#     jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
#     jets_jets = np.delete(jets_jets, 5, 2)

# # jets
# jet_mu = jets_lep
# # First jet for every event
# jet_1 = jets_jets[:,0]
# # Second jet for every event
# jet_2 = jets_jets[:,1]
# jet_3 = jets_jets[:,2]
# jet_4 = jets_jets[:,3]
# jet_5 = jets_jets[:,4]
# # Create an array with each jet's arrays for accessing b-tagging states later.
# jet_list = np.stack([jet_1, jet_2, jet_3, jet_4, jet_5]) 

# # truth
# y_true_W_had = true[:,0,:]
# y_true_W_lep = true[:,1,:]
# y_true_b_had = true[:,2,:]
# y_true_b_lep = true[:,3,:]
# y_true_t_had = true[:,4,:]
# y_true_t_lep = true[:,5,:]

# # store number of events as a separate variable for clarity
# n_events = true.shape[0]
# w = 1
# print("INFO ...done")

# ################################################################################
# # MAKE ROOT FILE

ofilename = "{}/jet_filter_hists".format(output_dir)
# # Open output file
# ofile = TFile.Open( ofilename, "recreate" )
# ofile.cd()

# ################################################################################
# # HISTOGRAMS
# hists = {}
# # hists['max_jet_Pt'] = TH1F("max_jet_Pt","p_{T} (GeV)", 50, 0, 500)
# # hists['jet_had_W_Pt_diff'] = TH1F("jet_had_W_Pt_diff","p_{T} (GeV)", 50, -300, 300)
# hists['had_W_Pt'] = TH1F("had_W_Pt","p_{T} (GeV)", 50, 0, 500)
# hists['jet12_had_W_Pt_diff'] = TH1F("jet12_had_W_Pt_diff", "p_{T} (GeV)", 50, -300, 300)
# hists['jet12_Pt'] = TH1F("jet12_Pt", "p_{T} (GeV)", 50, 0, 500)

# for i in range(5):
#     hists['max_jet_{}_Pt'.format(i)] = TH1F("max_jet_{}_Pt'.format(i)","p_{T} (GeV)", 50, 0, 500)
#     hists['jet_{}_had_W_Pt_diff'.format(i)] = TH1F("jet_{}_had_W_Pt_diff".format(i),"p_{T} (GeV)", 50, -300, 300)
#     hists['jet_{}_had_W_Pt_diff'.format(i)].SetTitle("Leading Jet p_{T} - Had W p_{T}; p_{T} (GeV);A.U.")

# ################################################################################
# # POPULATE TREE

# print("INFO: starting event loop. Found %i events" % n_events)

# # Print out example
# for i in range(n_events):
#     if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
#         perc = 100. * i / float(n_events)
#         print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

#     # make jets
        
#     W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
#     W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)
#     b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
#     b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
#     t_had_true   = MakeP4( y_true_t_had[i], m_t , representation)
#     t_lep_true   = MakeP4( y_true_t_lep[i], m_t , representation)

#     jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4], representation)

#     jet_1_vect = MakeP4(jet_1[i], jet_1[i][4], representation)
#     jet_2_vect = MakeP4(jet_2[i], jet_2[i][4], representation)
#     jet_3_vect = MakeP4(jet_3[i], jet_3[i][4], representation)
#     jet_4_vect = MakeP4(jet_4[i], jet_4[i][4], representation)
#     jet_5_vect = MakeP4(jet_5[i], jet_5[i][4], representation)
    
#     # add list containing jets of correspoonding event
#     jets = [jet_1_vect, jet_2_vect, jet_3_vect, jet_4_vect]
#     jets_Pt = [jet_1_vect.Pt(), jet_2_vect.Pt(), jet_3_vect.Pt(), jet_4_vect.Pt()]
#     # sort in decreasing order
#     jets_Pt.sort(reverse = True)

#     # If there is no fifth jet, do not append it to list of jets to avoid counting it
#     if not np.all(jet_5[i] == 0.):
#         jets.append(jet_5_vect)
#         jets_Pt.append(jet_5_vect.Pt())

#     # if W_had_true.Pt() < 20:
#     #     continue

#     hists['had_W_Pt'].Fill(W_had_true.Pt())
#     for i in range(len(jets_Pt)):
#         hists['max_jet_{}_Pt'.format(i)].Fill(jets_Pt[i])
#         hists['jet_{}_had_W_Pt_diff'.format(i)].Fill( jets_Pt[i]-W_had_true.Pt() )
#     hists['jet12_had_W_Pt_diff'].Fill(jets_Pt[0]+jets_Pt[-1]-W_had_true.Pt())
#     hists['jet12_Pt'].Fill(jets_Pt[0]+jets_Pt[-1])

#     # hists['jet5_had_b_Pt_diff'].Fill(jets_Pt[1]+jets_Pt[2]-W_had_true.Pt())
#     # hists['jet5_Pt'].Fill(jets_Pt[1]+jets_Pt[2])

# for histname in hists:
#     hists[histname].Write(histname)

# ofile.Write()
# ofile.Close()

infile = TFile.Open(ofilename)

def plot_hist(h, obs, caption):

    c = TCanvas()
    h.Draw()

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

hist_W = infile.Get('had_W_Pt')
hist_jet12_W_pt_diff = infile.Get('jet12_had_W_Pt_diff')
plot_hist(hist_jet12_W_pt_diff, 'jet12_had_W_Pt_diff', "Leading Jet 1 + Jet 3 p_{T} - Hadronic W p_{T}")
for i in range(5):
    hist_pt_diff = infile.Get('jet_{}_had_W_Pt_diff'.format(i))
    hist_jet = infile.Get('max_jet_{}_Pt'.format(i))
    plot_hist(hist_jet, 'jet_{}_Pt'.format(i), str(i+1)+' Leading Jet p_{T}')
    plot_hist(hist_pt_diff, 'jet_{}_W_Pt_diff'.format(i), str(i+1)+' Leading Jet p_{T} - Hadronic W p_{T}')


# infilename_truth = "{}/histograms.root".format("../CheckPoints/May27")
# infile_truth = TFile.Open(infilename_truth)

# # Make a plot for each observable
# for obs in attributes:
#     if "_pt" in obs:
#         plot_hist(obs)


from AngryTops.Plotting.PlottingHelper import *

hist_jet12_W = infile.Get('jet12_Pt')
plot_observables(hist_jet12_W, hist_W, 'jet12_v_W_Pt', "Leading Jet 1 + Jet 3 p_{T} vs Hadronic W p_{T}")
for i in range(5):
    hist_jet = infile.Get('max_jet_{}_Pt'.format(i))
    plot_observables(hist_jet, hist_W, 'jet_{}_v_W_Pt'.format(i), str(i+1)+' Leading Jet p_{T} vs Hadronic W p_{T}')
