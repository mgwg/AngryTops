import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists, plot_corr

training_dir = sys.argv[1]
representation = sys.argv[2]
plot_type = sys.argv[3] # good, bad, all
dir_name = ''
if len(dir_name) > 4:
    date = dir_name[4]

subdir = '/closejets_img{}/'.format(dir_name)
scaling = True # whether the dataset has been passed through a scaling function or not
m_t = 172.5
m_W = 80.4
m_b = 4.95

################################################################################
# Read in input file
infilename = "{}/fitted_{}.root".format(training_dir, plot_type)
infile = TFile.Open( infilename )
print(infilename)
tree   = infile.Get( "nominal")

ofilename = "{}/histograms_{}.root".format(training_dir, plot_type)
ofile = TFile.Open( ofilename, "recreate" )
ofile.cd()

################################################################################
# MAKE HISTOGRAMS

hists = {}

# Leptonic W
# True vs. obs
hists['lep_W_dist_true_v_obs'] = TH1F("h_W_lep_true","W Leptonic Distances, True vs Observed", 50, 0, 3)
hists['lep_W_dist_true_v_obs'].SetTitle("W Leptonic #phi distances, True vs Observed; Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_W_dist_pred_v_true'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_W_dist_pred_v_true'].SetTitle("W Leptonic #phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_W_dist_pred_v_obs'] = TH1F("h_W_lep_d","W Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_W_dist_pred_v_obs'].SetTitle("W Leptonic #phi distances, Predicted vs Observed; Leptonic (radians);A.U.")
# transverse mass and energy
hists['lep_W_transverse_energy_diff'] = TH1F("W_lep_ET_d","W Leptonic Transverse Energy Difference, Truth - Observed", 50, -120, 120)
hists['lep_W_transverse_energy_diff'].SetTitle("W Leptonic Transverse Energy Difference, Truth - Observed;Leptonic (GeV);A.U.")
# Matching correlation plots
hists['lep_W_corr_ET_diff_dist_true_v_obs'] = TH2F("W Leptonic E_{T} Diffs vs. #eta-#phi Distances", ";W Leptonic #eta-#phi Distances, True vs Observed; W Leptonic E_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
# ET distributions
hists['lep_W_transverse_energy_obs'] = TH1F("W_lep_ET","W Leptonic Transverse Energy, Observed", 80, 0, 400)
hists['lep_W_transverse_energy_obs'].SetTitle("W Leptonic Transverse Energy, Observed;Leptonic (GeV);A.U.")
hists['lep_W_transverse_energy_diff'] = TH1F("W_lep_ET_d","W Leptonic Transverse Energy Difference, Truth - Observed", 50, -120, 120)
hists['lep_W_transverse_energy_diff'].SetTitle("W Leptonic Transverse Energy Difference, Truth - Observed;Leptonic (GeV);A.U.")
# Closest ET difference vs. ET
hists['lep_W_corr_ET_diff_ET_obs'] = TH2F("W Leptonic E_{T} Diffs vs. Observed W Leptonic E_{T}", ";Observed W Leptonic E_{T} [GeV]; W Leptonic E_{T} Diff [GeV]", 50, 0, 200, 50, -200, 200)

# Hadronic W
# True vs. obs
hists['had_W_dist_true_v_obs'] = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist_true_v_obs'].SetTitle("W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
hists['had_W_1_dist'] = TH1F("h_1_W_had_true","1 Jet W Hadronic Distances, True - Observed", 50, 0, 3 )
hists['had_W_1_dist'].SetTitle("1 Jet W Hadronic #eta-#phi distances, True - Observed; Hadronic (radians); A.U.")
hists['had_W_3_dist'] = TH1F("h_3_W_had_true","3 Jet W Hadronic Distances, True - Observed", 50, 0, 3 )
hists['had_W_3_dist'].SetTitle("3 Jet W Hadronic #eta-#phi distances, True - Observed; Hadronic (radians); A.U.")
hists['had_W_2_dist'] = TH1F("h_2_W_had_true","2 Jet W Hadronic Distances, True - Observed", 50, 0, 3 )
hists['had_W_2_dist'].SetTitle("2 Jet W Hadronic #eta-#phi distances, True - Observed; Hadronic (radians); A.U.")
# Pred vs. true
hists['had_W_dist_pred_v_true'] = TH1F("h_W_had_pred","W Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_W_dist_pred_v_true'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth; Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_W_dist_pred_v_obs'] = TH1F("h_W_had_d","W Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_W_dist_pred_v_obs'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['had_W_obs_1_mass'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_1_mass'].SetTitle("1 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_1_mass_log'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_1_mass_log'].SetTitle("1 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_2_mass'] = TH1F("W_had_m","2 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_2_mass'].SetTitle("2 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_3_mass'] = TH1F("W_had_m","3 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_3_mass'].SetTitle("3 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_W_obs_mass'].SetTitle("W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching transverse momentum distributions
hists['had_W_true_obs_pT'] = TH1F("h_pT_W_had", "W Hadronic p_{T}, Observed", 50, 0, 400)
hists['had_W_true_obs_pT'].SetTitle("W Hadronic p_{T}, Observed; Hadronic (GeV); A.U.")
hists['had_W_true_obs_pT_diff'] = TH1F("h_pT_W_had_diff","W Hadronic p_{T} diffs, True - Observed", 50, -400, 400)
hists['had_W_true_obs_pT_diff'].SetTitle("W Hadronic p_{T} diffs, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_1_pT_diff'] = TH1F("h_pT_W_had_true","1 Jet W Hadronic p_{T} Diff, True - Observed", 50, -300, 300. )
hists['had_W_true_1_pT_diff'].SetTitle("1 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_3_pT_diff'] = TH1F("h_pT_W_had_true","3 Jet W Hadronic p_{T} Diff, True - Observed", 50, -300, 300. )
hists['had_W_true_3_pT_diff'].SetTitle("3 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_2_pT_diff'] = TH1F("h_pT_W_had_true","2 Jet W Hadronic p_{T} Diff, True - Observed", 50, -300, 300. )
hists['had_W_true_2_pT_diff'].SetTitle("2 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
# Jet matching criteria correlation plots
# invariant mass vs eta-phi dist
hists['had_W_corr_1_mass_dist_true_v_obs'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0, 120 , 50, 0, 3.2  )
hists['had_W_corr_2_mass_dist_true_v_obs'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic #eta-#phi Distances [rad]", 50, 10, 300 , 50, 0, 3.2  )
hists['had_W_corr_3_mass_dist_true_v_obs'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic #eta-#phi Distances [rad]", 50, 40, 350 , 50, 0, 3.2  )
hists['had_W_corr_mass_dist_true_v_obs'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic #eta-#phi Distances [rad]", 50, 0, 350 , 50, 0, 3.2  )
# invariant mass vs Pt difference
hists['had_W_corr_1_mass_Pt_true_v_obs'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 80 , 50, -200, 200  )
hists['had_W_corr_2_mass_Pt_true_v_obs'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 10, 300 , 50, -200, 200  )
hists['had_W_corr_3_mass_Pt_true_v_obs'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic p_{T} diff [GeV]", 50, 40, 350 , 50, -200, 200  )
hists['had_W_corr_mass_Pt_true_v_obs'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic p_{T} diff [GeV]", 50, 0, 350 , 50, -200, 200  )
# eta-phi dist vs. Pt difference
hists['had_W_corr_1_dist_Pt_true_v_obs'] = TH2F( "W_had_corr_d",   ";1 Jet W Hadronic #eta-#phi Distances [rad];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_2_dist_Pt_true_v_obs'] = TH2F( "W_had_corr_d",   ";2 Jet W Hadronic #eta-#phi Distances [rad];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_3_dist_Pt_true_v_obs'] = TH2F( "W_had_corr_d",   ";3 Jet W Hadronic #eta-#phi Distances [rad];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_dist_Pt_true_v_obs'] = TH2F( "W_had_corr_d",   ";W Hadronic #eta-#phi Distances [rad];W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )

# Leptonic b
# True vs. obs
hists['lep_b_dist_true_v_obs'] = TH1F("h_b_lep_true","b Leptonic Distances, True vs Observed", 50, 0, 3)
hists['lep_b_dist_true_v_obs'].SetTitle("b Leptonic #eta-#phi distances, True vs Observed;Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_b_dist_pred_v_true'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_b_dist_pred_v_true'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_b_dist_pred_v_obs'] = TH1F("h_b_lep_d","b Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_b_dist_pred_v_obs'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Observed; Leptonic (radians);A.U.")
# Jet matching invariant mass distributions
hists['lep_b_obs_mass'] = TH1F("b_lep_m","b Leptonic Invariant Mass, Observed", 60, 0., 50. )
hists['lep_b_obs_mass'].SetTitle("b Leptonic Invariant Mass, Observed; Leptonic (GeV); A.U.")
# Jet matching transverse momentum distributions
hists['lep_b_true_obs_pT'] = TH1F("h_pT_b_lep","b Leptonic p_{T}, Observed", 80, 0, 400)
hists['lep_b_true_obs_pT'].SetTitle("b Leptonic p_{T}, Observed; Leptonic (GeV); A.U.")
hists['lep_b_true_obs_pT_diff'] = TH1F("h_pT_b_lep_diff","b Leptonic p_{T} diffs, True - Observed", 80, -400, 400)
hists['lep_b_true_obs_pT_diff'].SetTitle("b Leptonic p_{T} diffs, True - Observed; Leptonic (GeV); A.U.")
# Closest PT difference vs. PT
hists['lep_b_corr_pT_diff_pT_obs'] = TH2F("b Leptonic p_{T} Diffs vs. Observed b Leptonic p_{T}", ";Observed b Leptonic p_{T} [GeV]; b Leptonic p_{T} Diff [GeV]", 50, 0, 200, 50, -200, 200)
# Jet matching criteria correlation plots
hists['lep_b_corr_dist_true_v_obs_mass'] = TH2F("b Leptonic #eta-#phi Distances vs. Invariant Mass", ";b Leptonic Invariant Mass [GeV]; b Leptonic #eta-#phi Distances, True vs Observed", 50, 0, 50, 50, 0, 3.2)
hists['lep_b_corr_pT_diff_true_v_obs_mass'] = TH2F("b Leptonic p_{T} Diffs vs. Invariant Mass", ";b Leptonic Invariant Mass [GeV]; b Leptonic p_{T} Diff, True - Observed [GeV]", 50, 0, 50, 50, -100, 100)
hists['lep_b_corr_pT_diff_dist_true_v_obs'] = TH2F("b Leptonic p_{T} Diffs vs. #eta-#phi Distances", ";b Leptonic #eta-#phi Distances, True vs Observed; b Leptonic p_{T} Diff [GeV]", 50, 0, 3.2, 50, -100, 100)

# Hadronic b
# True vs. obs
hists['had_b_dist_true_v_obs'] = TH1F("h_b_had_true","b Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_b_dist_true_v_obs'].SetTitle("b Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_b_dist_pred_v_true'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_b_dist_pred_v_true'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth;Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_b_dist_pred_v_obs'] = TH1F("h_b_had_d","b Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_b_dist_pred_v_obs'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['had_b_obs_mass'] = TH1F("b_had_m","b Hadronic Invariant Mass, Observed", 60, 0., 50. )
hists['had_b_obs_mass'].SetTitle("b Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching transverse momentum distributions
hists['had_b_true_obs_pT'] = TH1F("h_pT_b_had","b Hadronic p_{T}, Observed", 80, 0, 400)
hists['had_b_true_obs_pT'].SetTitle("b Hadronic p_{T}, Observed; Hadronic (GeV); A.U.")
hists['had_b_true_obs_pT_diff'] = TH1F("h_pT_b_had_diff","b Hadronic p_{T} diffs, True - Observed", 80, -400, 400)
hists['had_b_true_obs_pT_diff'].SetTitle("b Hadronic p_{T} diffs, True - Observed; Hadronic (GeV); A.U.")
# Closest PT difference vs. PT
hists['had_b_corr_pT_diff_pT_obs'] = TH2F("b Hadronic p_{T} Diffs vs. Observed b Hadronic p_{T}", ";Observed b Hadronic p_{T} [GeV]; b Hadronic p_{T} Diff [GeV]", 50, 0, 200, 50, -200, 200)
# Jet matching criteria correlation plots
hists['had_b_corr_dist_true_v_obs_mass'] = TH2F("b Hadronic #eta-#phi Distances vs. Invariant Mass", ";b Hadronic Invariant Mass [GeV]; b Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 50, 50, 0, 3.2)
hists['had_b_corr_pT_diff_true_v_obs_mass'] = TH2F("b Hadronic p_{T} Diffs vs. Invariant Mass", ";b Hadronic Invariant Mass [GeV]; b Hadronic p_{T} Diff, True - Observed [GeV]", 50, 0, 50, 50, -100, 100)
hists['had_b_corr_pT_diff_dist_true_v_obs'] = TH2F("b Hadronic p_{T} Diffs vs. #eta-#phi Distances", ";b Hadronic #eta-#phi Distances, True vs Observed; b Hadronic p_{T} Diff [GeV]", 50, 0, 3.2, 50, -100, 100)

# Leptonic t
# True vs. obs
hists['lep_t_dist_true_v_obs'] = TH1F("h_t_lep_true","t Leptonic Distances, True vs Observed", 50, 0, 3)
hists['lep_t_dist_true_v_obs'].SetTitle("t Leptonic #phi distances, True vs Observed;Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_t_dist_pred_v_true'] = TH1F("t_lep_d","t Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_t_dist_pred_v_true'].SetTitle("t Leptonic #phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_t_dist_pred_v_obs'] = TH1F("h_t_lep_d","t Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_t_dist_pred_v_obs'].SetTitle("t Leptonic #phi distances, Predicted vs Observed; Leptonic (radians);A.U.")
# transverse mass and energy
hists['lep_t_transverse_energy_diff'] = TH1F("t_lep_ET_d","t Leptonic Transverse Energy Difference, Truth - Observed", 50, -200, 200)
hists['lep_t_transverse_energy_diff'].SetTitle("t Leptonic Transverse Energy Difference, Truth - Observed;Leptonic (GeV);A.U.")
# Matching correlation plots
hists['lep_t_corr_ET_diff_dist_true_v_obs'] = TH2F("t Leptonic E_{T} Diffs vs. #eta-#phi Distances", ";t Leptonic #eta-#phi Distances, True vs Observed; t Leptonic E_{T} Diff [GeV]", 50, 0, 3.2, 50, -400, 400)

# Hadronic t
# True vs. obs
hists['had_t_dist_true_v_obs'] = TH1F("h_t_had_true","t Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_t_dist_true_v_obs'].SetTitle("t Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_t_dist_pred_v_true'] = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_t_dist_pred_v_true'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth;Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_t_dist_pred_v_obs'] = TH1F("h_t_had_d","t Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_t_dist_pred_v_obs'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['had_t_obs_mass'] = TH1F("t_had_m","t Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['had_t_obs_mass'].SetTitle("t Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching momentum distributions
hists['had_t_true_obs_pT_diff'] = TH1F("h_pT_t_had_diff","t Hadronic p_{T} diffs, True - Observed", 80, -400, 400)
hists['had_t_true_obs_pT_diff'].SetTitle("t Hadronic p_{T} diffs, True - Observed; Hadronic (GeV); A.U.")
# Jet matching criteria correlation plots
hists['had_t_corr_dist_true_v_obs_mass'] = TH2F("t Hadronic #eta-#phi Distances vs. Invariant Mass", ";t Hadronic Invariant Mass [GeV]; t Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['had_t_corr_pT_diff_true_v_obs_mass'] = TH2F("t Hadronic p_{T} Diffs vs. Invariant Mass", ";t Hadronic Invariant Mass [GeV]; t Hadronic p_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -400, 400)
hists['had_t_corr_pT_diff_dist_true_v_obs'] = TH2F("t Hadronic p_{T} Diffs vs. #eta-#phi Distances", ";t Hadronic #eta-#phi Distances, True vs Observed; t Hadronic p_{T} Diff [GeV]", 50, 0, 3.2, 50, -400, 400)

################################################################################
# POPULATE HISTOGRAMS

n_events = tree.GetEntries()

print("INFO: starting event loop. Found %i events" % n_events)

for i in range(n_events): # loop through every event
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

    W_had_obs = TLorentzVector(tree.W_had_px_obs , tree.W_had_py_obs , tree.W_had_pz_obs , tree.W_had_E_obs )
    b_had_obs = TLorentzVector(tree.b_had_px_obs , tree.b_had_py_obs , tree.b_had_pz_obs , tree.b_had_E_obs )
    t_had_obs = TLorentzVector(tree.t_had_px_obs , tree.t_had_py_obs , tree.t_had_pz_obs , tree.t_had_E_obs )
    b_lep_obs = TLorentzVector(tree.b_lep_px_obs , tree.b_lep_py_obs , tree.b_lep_pz_obs , tree.b_lep_E_obs)
    W_lep_px_obs   = tree.W_lep_px_obs
    W_lep_py_obs   = tree.W_lep_py_obs
    W_lep_Et_obs  = tree.W_lep_Et_obs
    W_lep_phi_obs  = tree.W_lep_phi_obs 
    t_lep_px_obs   = tree.t_lep_px_obs
    t_lep_py_obs   = tree.t_lep_py_obs
    t_lep_pt_obs  = tree.t_lep_pt_obs
    t_lep_phi_obs  = tree.t_lep_phi_obs    

    num_jets = tree.W_had_num_jets#s


    ################################################# true vs observed ################################################# 

    b_had_dist = find_dist(b_had_true, b_had_obs)
    b_lep_dist = find_dist(b_lep_true, b_lep_obs)
    
    W_had_pT_diff = W_had_true.Pt() - W_had_obs.Pt()
    W_had_dist = find_dist(W_had_true, W_had_obs)
    
    W_lep_dist = np.abs( min( np.abs(W_lep_true.Phi()-W_lep_phi_obs), 2*np.pi-np.abs(W_lep_true.Phi()-W_lep_phi_obs) ) )
    # Calculate transverse energy assuming daughter particles are massless
    W_lep_ET_observed = np.sqrt( W_lep_px_obs**2 + W_lep_py_obs**2)
    W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed

    t_had_dist = find_dist(t_had_true, t_had_obs)
    t_had_pT_diff = t_had_true.Pt() - t_had_obs.Pt()

    t_lep_dist = np.abs( min( np.abs(t_lep_true.Phi()-t_lep_phi_obs), 2*np.pi-np.abs(t_lep_true.Phi() - t_lep_phi_obs) ) )
    t_lep_pT_diff = t_lep_true.Et() - t_lep_pt_obs

    # b quark calculations
    b_had_pT_diff = b_had_true.Pt() - b_lep_obs.Pt()
    b_lep_pT_diff = b_lep_true.Pt() - b_lep_obs.Pt()

    ################################################# predicted vs observed #################################################

    # Once the optimal jets have been matched in the previous section, 
    #  the eta-phi distances can be calculated between the predicted and observed variables
    #  with no further work.

    # Leptonic W
    # Calculate the distance between predicted and observed phi. 
    # No eta distance for comparison with truth vs. obs and pred vs. true
    W_lep_dphi_po = np.abs( min( np.abs(W_lep_fitted.Phi()-W_lep_phi_obs), 2*np.pi-np.abs(W_lep_fitted.Phi()-W_lep_phi_obs) ) )
    W_lep_R_po = np.sqrt(W_lep_dphi_po**2)
    # Hadronic W
    W_had_R_po = find_dist( W_had_fitted, W_had_obs )

    # Leptonic b
    b_lep_R_po = find_dist( b_lep_fitted, b_lep_obs )
    # Hadronic b
    b_had_R_po = find_dist( b_had_fitted, b_lep_obs )

    # Leptonic t
    t_lep_dphi_po = min(np.abs(t_lep_fitted.Phi()-t_lep_phi_obs), 2*np.pi-np.abs(t_lep_fitted.Phi()-t_lep_phi_obs))
    t_lep_R_po = np.sqrt(t_lep_dphi_po**2) # Again, no eta
    # Hadronic t
    t_had_R_po = find_dist( t_had_fitted, t_had_obs )

    ################################################# populate histograms #################################################

    # Leptonic b
    hists['lep_b_dist_true_v_obs'].Fill(np.float(b_lep_dist))
    hists['lep_b_dist_pred_v_obs'].Fill(np.float(b_lep_R_po))
    # Invariant mass:
    hists['lep_b_obs_mass'].Fill(b_lep_obs.M())
    # Jet matching criteria correlation plots
    hists['lep_b_corr_dist_true_v_obs_mass'].Fill(b_lep_obs.M(), b_lep_dist) 
    hists['lep_b_corr_pT_diff_true_v_obs_mass'].Fill(b_lep_obs.M(), b_lep_pT_diff) 
    hists['lep_b_corr_pT_diff_dist_true_v_obs'].Fill(b_lep_dist, b_lep_pT_diff)
    # Closest PT difference vs. PT
    hists['lep_b_true_obs_pT'].Fill(b_lep_obs.Pt())
    hists['lep_b_true_obs_pT_diff'].Fill(b_lep_pT_diff)
    hists['lep_b_corr_pT_diff_pT_obs'].Fill(b_lep_obs.Pt(), b_lep_pT_diff) 

    # Hadronic b
    hists['had_b_dist_true_v_obs'].Fill(np.float(b_had_dist))
    hists['had_b_dist_pred_v_obs'].Fill(np.float(b_had_R_po))
    # Invariant mass:
    hists['had_b_obs_mass'].Fill(b_had_obs.M())
    # Jet matching criteria correlation plots
    hists['had_b_corr_dist_true_v_obs_mass'].Fill(b_had_obs.M(), b_had_dist) 
    hists['had_b_corr_pT_diff_true_v_obs_mass'].Fill(b_had_obs.M(), b_had_pT_diff) 
    hists['had_b_corr_pT_diff_dist_true_v_obs'].Fill(b_had_dist, b_had_pT_diff)
    # Closest PT difference vs. PT
    hists['had_b_true_obs_pT'].Fill(b_had_obs.Pt())
    hists['had_b_true_obs_pT_diff'].Fill(b_had_pT_diff)
    hists['had_b_corr_pT_diff_pT_obs'].Fill(b_had_obs.Pt(), b_had_pT_diff) 

    # Leptonic t
    hists['lep_t_dist_true_v_obs'].Fill(np.float(t_lep_dist))
    hists['lep_t_dist_pred_v_obs'].Fill(np.float(t_lep_R_po))
    hists['lep_t_transverse_energy_diff'].Fill(np.float(t_lep_pT_diff))
    hists['lep_t_corr_ET_diff_dist_true_v_obs'].Fill(t_lep_dist, t_lep_pT_diff)
    # Hadronic t
    hists['had_t_dist_true_v_obs'].Fill(np.float(t_had_dist))
    hists['had_t_dist_pred_v_obs'].Fill(np.float(t_had_R_po))
    hists['had_t_obs_mass'].Fill(t_had_obs.M())
    hists['had_t_true_obs_pT_diff'].Fill(t_had_pT_diff)
    hists['had_t_corr_dist_true_v_obs_mass'].Fill(t_had_obs.M(), t_had_dist)
    hists['had_t_corr_pT_diff_true_v_obs_mass'].Fill(t_had_obs.M(), t_had_pT_diff)
    hists['had_t_corr_pT_diff_dist_true_v_obs'].Fill(t_had_dist, t_had_pT_diff)

    # Leptonic W
    hists['lep_W_dist_true_v_obs'].Fill(np.float(W_lep_dist))
    hists['lep_W_dist_pred_v_obs'].Fill(np.float(W_lep_R_po))
    # Closest ET difference vs. ET
    hists['lep_W_transverse_energy_obs'].Fill(np.float(W_lep_ET_observed))
    hists['lep_W_transverse_energy_diff'].Fill(np.float(W_lep_ET_diff))
    hists['lep_W_corr_ET_diff_ET_obs'].Fill(W_lep_ET_observed, W_lep_ET_diff)
    hists['lep_W_corr_ET_diff_dist_true_v_obs'].Fill(W_lep_dist, W_lep_ET_diff) 

    # Hadronic W
    hists['had_W_dist_true_v_obs'].Fill(np.float(W_had_dist))
    hists['had_W_dist_pred_v_obs'].Fill(np.float(W_had_R_po))
    # Invariant mass:
    hists['had_W_obs_mass'].Fill(W_had_obs.M())
    # Jet matching criteria correlation plots
    hists['had_W_corr_mass_dist_true_v_obs'].Fill(W_had_obs.M(), W_had_dist) 
    hists['had_W_corr_mass_Pt_true_v_obs'].Fill(W_had_obs.M(), W_had_pT_diff) 
    hists['had_W_corr_dist_Pt_true_v_obs'].Fill(W_had_dist, W_had_pT_diff)  
    # Closest pT difference vs. pT
    hists['had_W_true_obs_pT'].Fill(np.float(W_had_obs.Pt()))
    hists['had_W_true_obs_pT_diff'].Fill(np.float(W_had_pT_diff))
    # Plots that depend on whether a 1,2, or 3-jet sum is the best match to truth:
    if num_jets == 0:
        hists['had_W_true_1_pT_diff'].Fill(np.float(W_had_pT_diff))
        hists['had_W_obs_1_mass'].Fill(W_had_obs.M())
        hists['had_W_obs_1_mass_log'].Fill(W_had_obs.M())
        hists['had_W_corr_1_mass_dist_true_v_obs'].Fill(W_had_obs.M(), W_had_dist)
        hists['had_W_corr_1_mass_Pt_true_v_obs'].Fill(W_had_obs.M(), W_had_pT_diff)
        hists['had_W_corr_1_dist_Pt_true_v_obs'].Fill(W_had_dist, W_had_pT_diff)
        hists['had_W_1_dist'].Fill(np.float(W_had_dist))
    elif num_jets == 1:
        hists['had_W_true_2_pT_diff'].Fill(np.float(W_had_pT_diff))
        hists['had_W_obs_2_mass'].Fill(W_had_obs.M())
        hists['had_W_corr_2_mass_dist_true_v_obs'].Fill(W_had_obs.M(), W_had_dist)
        hists['had_W_corr_2_mass_Pt_true_v_obs'].Fill(W_had_obs.M(), W_had_pT_diff)
        hists['had_W_corr_2_dist_Pt_true_v_obs'].Fill(W_had_dist, W_had_pT_diff)
        hists['had_W_2_dist'].Fill(np.float(W_had_dist))
    elif num_jets == 2:
        hists['had_W_true_3_pT_diff'].Fill(np.float(W_had_pT_diff))
        hists['had_W_obs_3_mass'].Fill(W_had_obs.M())
        hists['had_W_corr_3_mass_dist_true_v_obs'].Fill(W_had_obs.M(), W_had_dist)
        hists['had_W_corr_3_mass_Pt_true_v_obs'].Fill(W_had_obs.M(), W_had_pT_diff)
        hists['had_W_corr_3_dist_Pt_true_v_obs'].Fill(W_had_dist, W_had_pT_diff)
        hists['had_W_3_dist'].Fill(np.float(W_had_dist))

for histname in hists:
    hists[histname].Write(histname)
print("Finished. Saved output file:", ofilename)

################################################################################
# PLOT
# reopen outfile to plot
ofile = TFile.Open(ofilename)

for key in hists:
    if 'corr' not in key:
        ofile.Get(hist_name)
        plot_hists(key, hists[key], outputdir+subdir)

# The following few lines must be run only once for all correlation plots, 
#  so the correlation plots must be separated out from the other histograms.   
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *
gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

for key in hists:
    if 'corr' in key:
        ofile.Get(hist_name)
        plot_corr(key, hists[key], outputdir+subdir)