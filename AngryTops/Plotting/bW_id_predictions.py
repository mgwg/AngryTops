# -- output num events meeting cut for predictions v true
# -- output std of true test event

import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

outputdir = sys.argv[1]
dir_name = ''
if len(sys.argv) > 2:
    dir_name = sys.argv[2]

subdir = '/closejets_img_predictions{}/'.format(dir_name)
scaling = True # whether the dataset has been passed through a scaling function or not
m_t = 172.5
m_W = 80.4
m_b = 4.95

################################################################################
# Read in input file
infile = TFile.Open( infilename )
tree   = infile.Get( "nominal")

################################################################################
# MAKE HISTOGRAMS

# make histograms to be filled
hists = {}

# Leptonic W
# Pred vs. true
hists['lep_W_dist_pred_v_true'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_W_dist_pred_v_true'].SetTitle("W Leptonic #phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
hists['lep_W_true_pred_pT_diff'] = TH1F("h_pT_W_lep_diff","W Leptonic p_{T} diffs, True - Predicted", 50, -400, 400)
hists['lep_W_true_pred_pT_diff'].SetTitle("W Leptonic p_{T} diffs, True - Predicted; Leptonic (GeV); A.U.")

# Hadronic W
# Pred vs. true
hists['had_W_dist_pred_v_true'] = TH1F("h_W_had_pred","W Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_W_dist_pred_v_true'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth; Hadronic (radians);A.U.")
hists['had_W_true_pred_pT_diff'] = TH1F("h_pT_W_had_diff","W Hadronic p_{T} diffs, True - Predicted", 50, -400, 400)
hists['had_W_true_pred_pT_diff'].SetTitle("W Hadronic p_{T} diffs, True - Predicted; Hadronic (GeV); A.U.")

# Leptonic b
# Pred vs. true
hists['lep_b_dist_pred_v_true'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_b_dist_pred_v_true'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
hists['lep_b_true_pred_pT_diff'] = TH1F("h_pT_b_lep_diff","b Leptonic p_{T} diffs, True - Predicted", 80, -400, 400)
hists['lep_b_true_pred_pT_diff'].SetTitle("b Leptonic p_{T} diffs, True - Predicted; Leptonic (GeV); A.U.")

# Hadronic b
# Pred vs. true
hists['had_b_dist_pred_v_true'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_b_dist_pred_v_true'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth;Hadronic (radians);A.U.")
hists['had_b_true_pred_pT_diff'] = TH1F("h_pT_b_had_diff","b Hadronic p_{T} diffs, True - Predicted", 80, -400, 400)
hists['had_b_true_pred_pT_diff'].SetTitle("b Hadronic p_{T} diffs, True - Predicted; Hadronic (GeV); A.U.")

################################################################################
# DEFINE COUNTERS

# Counters to make tally number of events that pass cuts
W_had_total_cuts = 0.
W_had_m_cuts = 0.
W_had_pT_cuts = 0.
W_had_dist_cuts = 0.

W_lep_total_cuts = 0.
W_lep_pT_cuts = 0.
W_lep_dist_cuts = 0.

b_had_pT_cuts = 0.
b_had_dist_cuts = 0.
b_had_total_cuts = 0.

b_lep_pT_cuts = 0.
b_lep_dist_cuts = 0.
b_lep_total_cuts = 0.

good_event = 0.

################################################################################
# POPULATE HISTOGRAMS
n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)

for i in event_index: # loop through every event
    if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
        perc = 100. * i / float(n_events)
        print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
    
    tree.GetEntry(i)

    w = tree.weight_mc

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


    ################################################# true vs predicted #################################################
    b_lep_R = find_dist(b_lep_true, b_lep_fitted)
    b_had_R = find_dist(b_had_true, b_had_fitted)
    b_had_true_pred_pT_diff = b_had_true.Pt() - b_had_fitted.Pt()
    b_lep_true_pred_pT_diff = b_lep_true.Pt() - b_lep_fitted.Pt()
    
    W_lep_R = find_dist(W_lep_true, W_lep_fitted)
    W_had_R = find_dist(W_had_true, W_had_fitted)
    W_had_true_pred_pT_diff = W_had_true.Pt() - W_had_fitted.Pt()
    W_lep_true_pred_pT_diff = W_lep_true.Pt() - W_lep_fitted.Pt()
    
    ############################################## check whether each event passes cuts #################################################
    # counter for hadronic W
    # Update tally for which jet combination is the closest
    W_had_m_cut = (W_had_fitted.M() >= W_had_m_cutoff[0] and W_had_fitted.M() <= W_had_m_cutoff[1])
    W_had_pT_cut = (W_had_true_pred_pT_diff >= W_had_pT_cutoff[0] and W_had_true_pred_pT_diff <= W_had_pT_cutoff[1])
    W_had_dist_cut = (W_had_R <= W_had_dist_cutoff[1]) 
    # All W_had cuts must be satisfied simultaneously.
    good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)

    W_had_total_cuts += good_W_had
    W_had_m_cuts += W_had_m_cut
    W_had_pT_cuts += W_had_pT_cut
    W_had_dist_cuts += W_had_dist_cut

    # counter for lep W
    W_lep_pT_cut = (W_lep_true_pred_pT_diff >= W_lep_pT_cutoff[0] and W_lep_true_pred_pT_diff <= W_lep_pT_cutoff[1])
    W_lep_dist_cut = (W_lep_R <= W_lep_dist_cutoff[1]) 
    good_W_lep = (W_lep_pT_cut and W_lep_dist_cut)

    W_lep_total_cuts += good_W_lep
    W_lep_pT_cuts += W_lep_pT_cut
    W_lep_dist_cuts += W_lep_dist_cut

    # counter for hadronic b
    b_had_pT_cut = (b_had_true_pred_pT_diff >= b_had_pT_cutoff[0] and b_had_true_pred_pT_diff <= b_had_pT_cutoff[1])
    b_had_dist_cut = (b_had_R <= b_had_dist_cutoff[1]) 
    good_b_had = (b_had_pT_cut and b_had_dist_cut)

    b_had_total_cuts += good_b_had
    b_had_pT_cuts += b_had_pT_cut
    b_had_dist_cuts += b_had_dist_cut

    # counter for leptonic b
    b_lep_pT_cut = (b_lep_true_pred_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_pred_pT_diff <= b_lep_pT_cutoff[1])
    b_lep_dist_cut = (b_lep_R <= b_lep_dist_cutoff[1]) 
    good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

    b_lep_total_cuts += good_b_lep
    b_lep_pT_cuts += b_lep_pT_cut
    b_lep_dist_cuts += b_lep_dist_cut

    # Good events must pass cuts on all partons.
    good_event += (good_b_had and good_b_lep and good_W_had and good_W_lep)

    ################################################# populate histograms #################################################

    # Populate histograms if all events are to be plotted or we are only dealing with a good event 
    # if (good_b_had and good_b_lep and good_W_had and good_W_lep):

    # Leptonic b
    hists['lep_b_dist_pred_v_true'].Fill(np.float(b_lep_R))
    hists['lep_b_true_pred_pT_diff'].Fill(b_lep_true_pred_pT_diff)

    # Hadronic b
    hists['had_b_dist_pred_v_true'].Fill(np.float(b_had_R))
    hists['had_b_true_pred_pT_diff'].Fill(b_had_true_pred_pT_diff) 

    # Leptonic W
    hists['lep_W_dist_pred_v_true'].Fill(np.float(W_lep_R))
    hists['lep_W_true_pred_pT_diff'].Fill(W_lep_true_pred_pT_diff) 

    # Hadronic W
    hists['had_W_dist_pred_v_true'].Fill(np.float(W_had_R))
    hists['had_W_true_pred_pT_diff'].Fill(np.float(W_had_true_pred_pT_diff))

# Print data regarding percentage of each class of event
print('Total number of events: {} \n'.format(n_events))
print('NOTE: some percentages do not reach 100%, as events where no Hadronic W can be matched after removing the b-tagged jets are skipped (all jets are b-tagged)')
print('\n==================================================================\n')
print('Cut Criteria')
print('Hadronic W, mass: {}, pT: {}, distance: {}'.format(W_had_m_cutoff, W_had_pT_cutoff, W_had_dist_cutoff))
print('Leptonic W, E_T: {}, dist: {}'.format(W_lep_pT_cutoff, W_lep_dist_cutoff))
print('Hadronic b, pT: {}, distance: {}'.format(b_had_pT_cutoff, b_had_dist_cutoff))
print('Leptonic b, pT: {}, distance: {}'.format(b_lep_pT_cutoff, b_lep_dist_cutoff))
print('\n==================================================================\n')

print("Number of events satisfying all hadronic W cut criteria:")
print('{}%, {} events'.format(100.*W_had_total_cuts/n_events, int(W_had_total_cuts)))
print("Number of events satisfying hadronic W mass cut criteria")
print('{}%, {} events'.format(100.*W_had_m_cuts/n_events, int(W_had_m_cuts)))
print("Number of events satisfying hadronic W pT cut criteria")
print('{}%, {} events'.format(100.*W_had_pT_cuts/n_events, int(W_had_pT_cuts)))
print("Number of events satisfying hadronic W distance cut criteria")
print('{}%, {} events'.format(100.*W_had_dist_cuts/n_events, int(W_had_dist_cuts)))
print('\n==================================================================\n')
print("Number of events satisfying all leptonic W cut criteria")
print('{}% , {} events\n'.format(100.*W_lep_total_cuts/n_events, int(W_lep_total_cuts)))
print("Number of events satisfying leptonic W ET cut criteria")
print('{}%, {} events'.format(100.*W_lep_pT_cuts/n_events, int(W_lep_pT_cuts)))
print("Number of events satisfying leptonic W distance cut criteria")
print('{}%, {} events'.format(100.*W_lep_dist_cuts/n_events, int(W_lep_dist_cuts)))
print('\n==================================================================\n')
print("Number of events satisfying all hadronic b cut criteria")
print('{}% , {} events\n'.format(100.*b_had_total_cuts/n_events, int(b_had_total_cuts)))
print("Number of events satisfying hadronic b pT cut criteria")
print('{}%, {} events'.format(100.*b_had_pT_cuts/n_events, int(b_had_pT_cuts)))
print("Number of events satisfying hadronic b distance cut criteria")
print('{}%, {} events'.format(100.*b_had_dist_cuts/n_events, int(b_had_dist_cuts)))
print('\n==================================================================\n')
print("Number of events satisfying all leptonic b cut criteria")
print('{}% , {} events\n'.format(100.*b_lep_total_cuts/n_events, int(b_lep_total_cuts)))
print("Number of events satisfying leptonic b pT cut criteria")
print('{}%, {} events'.format(100.*b_lep_pT_cuts/n_events, int(b_lep_pT_cuts)))
print("Number of events satisfying leptonic b distance cut criteria")
print('{}%, {} events'.format(100.*b_lep_dist_cuts/n_events, int(b_lep_dist_cuts)))
print('\n==================================================================\n')
print("Events satisfying cut all cut criteria for all partons")
print('{}%, {} events'.format(100.*good_event/n_events, int(good_event)))


# Helper function to output and save the correlation plots
def plot_corr(key, hist, outputdir):

    SetTH1FStyle(hist,  color=kGray+2, fillstyle=6)

    c = TCanvas()
    c.cd()

    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 )
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.18 )
    pad0.SetTopMargin( 0.07 )
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()
    pad0.cd()

    hist.Draw("colz")

    corr = hist.GetCorrelationFactor()
    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack)
    legend.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

    gPad.RedrawAxis()

    caption = hist.GetName()
    newpad = TPad("newpad","a caption",0.1,0,1,1)
    newpad.SetFillStyle(4000)
    newpad.Draw()
    newpad.cd()
    title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
    title.SetFillColor(16)
    title.SetTextFont(52)
    if 'pT' or 'ET' in key:
        title.SetTextSize(0.8)
    title.Draw()

    c.cd()
    c.SaveAs(outputdir + key +'.png')
    pad0.Close()
    c.Close()

# Run the two helper functions above   
if __name__ == "__main__":
    try:
        os.mkdir('{}/{}'.format(outputdir, subdir))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()

    hists_key = []
    corr_key = []
    for key in hists:
        if 'corr' not in key:
            hists_key.append(key)
        else:
            corr_key.append(key)

    for key in hists_key:
        plot_hists(key, hists[key], outputdir+subdir)

    # The following few lines must be run only once for all correlation plots, 
    #  so the correlation plots must be separated out from the other histograms.   
    from AngryTops.features import *
    from AngryTops.Plotting.PlottingHelper import *
    gStyle.SetPalette(kGreyScale)
    gROOT.GetColor(52).InvertPalette()

    for key in corr_key:
        plot_corr(key, hists[key], outputdir+subdir)