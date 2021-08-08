import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

outputdir = sys.argv[1]
representation = sys.argv[2]
date = ''
if len(sys.argv) > 3:
    date = sys.argv[3]
event_type = 0
if len(sys.argv) > 4:
    event_type = sys.argv[4]

subdir = '/closejets_img{}/'.format(date)
scaling = True # whether the dataset has been passed through a scaling function or not
m_t = 172.5
m_W = 80.4
m_b = 4.95
ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE   # 0/All: Consider all jets, both b-tagged and not b-tagged
                # 1/None: Do not consider any b-tagged jets.
                # 2/Only: Consider only b-tagged jets
# For plot_cuts:
# True if you want to plot only the events that pass the cuts
# False to include events for which no combo of 1,2,3 jets pass cuts.                
plot_cuts = True
if plot_cuts:
    subdir = '/closejets_img_cuts{}/'.format(date)

# Cut ranges for the partons
# Cut ranges for the partons
W_had_m_cutoff = (30, 130)
W_had_pT_cutoff = (-100, 100)
W_had_dist_cutoff = (0, 0.8)

W_lep_ET_cutoff = (-100, 120)
W_lep_dist_cutoff = (0, 1.0)

b_had_pT_cutoff = (-80, 100)
b_had_dist_cutoff = (0, 0.8)

b_lep_pT_cutoff = (-80, 100)
b_lep_dist_cutoff = (0, 0.8)

# load data
predictions = np.load(outputdir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']

particles_shape = (true.shape[1], true.shape[2])
print("jets shape", jets.shape)
print("b tagging option", b_tagging)
if scaling:
    scaler_filename = outputdir + "scalers.pkl"
    with open( scaler_filename, "rb" ) as file_scaler:
        jets_scalar = pickle.load(file_scaler)
        lep_scalar = pickle.load(file_scaler)
        output_scalar = pickle.load(file_scaler)
        # Rescale the truth array
        true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
        true = output_scalar.inverse_transform(true)
        true = true.reshape(true.shape[0], particles_shape[0], particles_shape[1])
        # Rescale the fitted array
        fitted = fitted.reshape(fitted.shape[0], fitted.shape[1]*fitted.shape[2])
        fitted = output_scalar.inverse_transform(fitted)
        fitted = fitted.reshape(fitted.shape[0], particles_shape[0], particles_shape[1])
        # Rescale the jets array
        jets_lep = jets[:,:6]
        jets_jets = jets[:,6:] # remove muon column
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array

        # Remove the b-tagging states and put them into a new array to be re-appended later.
        b_tags = jets_jets[:,:,5]
        jets_jets = np.delete(jets_jets, 5, 2) # delete the b-tagging states

        jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) # reshape into 25 element long array
        jets_lep = lep_scalar.inverse_transform(jets_lep)
        jets_jets = jets_scalar.inverse_transform(jets_jets) # scale values ... ?
        #I think this is the final 6x6 array the arxiv paper was talking about - 5 x 5 array containing jets (1 per row) and corresponding px, py, pz, E, m
        jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))
        # Re-append the b-tagging states as a column at the end of jets_jets 
        jets_jets = np.append(jets_jets, np.expand_dims(b_tags, 2), 2)

if not scaling:
    jets_lep = jets[:,:6]
    jets_jets = jets[:,6:]
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
    jets_jets = np.delete(jets_jets, 5, 2)

# jets
jet_mu = jets_lep
# First jet for every event
jet_1 = jets_jets[:,0]
# Second jet for every event
jet_2 = jets_jets[:,1]
jet_3 = jets_jets[:,2]
jet_4 = jets_jets[:,3]
jet_5 = jets_jets[:,4]
# Create an array with each jet's arrays for accessing b-tagging states later.
jet_list = np.stack([jet_1, jet_2, jet_3, jet_4, jet_5]) 

# truth
y_true_W_had = true[:,0,:]
y_true_W_lep = true[:,1,:]
y_true_b_had = true[:,2,:]
y_true_b_lep = true[:,3,:]
y_true_t_had = true[:,4,:]
y_true_t_lep = true[:,5,:]

# fitted
y_fitted_W_had = fitted[:,0,:]
y_fitted_W_lep = fitted[:,1,:]
y_fitted_b_had = fitted[:,2,:]
y_fitted_b_lep = fitted[:,3,:]
y_fitted_t_had = fitted[:,4,:]
y_fitted_t_lep = fitted[:,5,:]

# store number of events as a separate variable for clarity
n_events = true.shape[0]

# define indices
event_index = range(n_events)
if event_type == "4":
    event_index = np.where(jet_5 == 0)
    event_index = np.unique(event_index[0])
elif event_type == "5":
    event_index = np.nonzero(jet_5)
    event_index = np.unique(event_index[0])

# make histograms to be filled
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
hists['lep_W_transverse_mass_obs'] = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Observed", 50, 0, 250)#120)
hists['lep_W_transverse_mass_obs'].SetTitle("W Leptonic Transverse Mass, Observed;Leptonic (GeV);A.U.")
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

# Function to make histograms
def make_histograms():
    
    # list of number of events best matched to 1,2,3 jets respectively.
    W_had_jets, W_had_total_cuts = [0., 0., 0.] , [0., 0., 0.] 

    # Counters to make tally number of events that pass cuts
    W_had_m_cuts, W_had_pT_cuts, W_had_dist_cuts = [0., 0., 0.] , [0., 0., 0.]  , [0., 0., 0.] 
    W_lep_total_cuts, W_lep_ET_cuts, W_lep_dist_cuts = 0., 0., 0.
    b_had_pT_cuts, b_had_dist_cuts, b_had_total_cuts = 0., 0., 0.
    b_lep_pT_cuts, b_lep_dist_cuts, b_lep_total_cuts = 0., 0., 0.

    good_event = 0.

    for i in event_index: # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W, representation)

        W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)
        W_lep_fitted = MakeP4( y_fitted_W_lep[i],  m_W, representation)

        b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
        b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b , representation)

        b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
        b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b, representation)

        t_had_true   = MakeP4( y_true_t_had[i], m_t , representation)
        t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t , representation)

        t_lep_true   = MakeP4( y_true_t_lep[i], m_t , representation)
        t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t, representation)

        jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4], representation)

        jet_1_vect = MakeP4(jet_1[i], jet_1[i][4], representation)
        jet_2_vect = MakeP4(jet_2[i], jet_2[i][4], representation)
        jet_3_vect = MakeP4(jet_3[i], jet_3[i][4], representation)
        jet_4_vect = MakeP4(jet_4[i], jet_4[i][4], representation)
        jet_5_vect = MakeP4(jet_5[i], jet_5[i][4], representation)
        
        jets = []
        # add list containing jets of correspoonding event
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
        if not np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)
        
        ################################################# true vs predicted #################################################
        b_lep_R = find_dist(b_lep_true, b_lep_fitted)
        b_had_R = find_dist(b_had_true, b_had_fitted)

        t_lep_dphi = min(np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()), 2*np.pi-np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()))
        t_lep_R = np.sqrt(t_lep_dphi**2) # No eta distances for apples-to-apples comparison with true vs. observed leptonic t

        t_had_R = find_dist(t_had_true, t_had_fitted)
        
        W_lep_dphi = min(np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()), 2*np.pi-np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()))
        W_lep_R = np.sqrt(W_lep_dphi**2)

        W_had_R = find_dist(W_had_true, W_had_fitted)


        ################################################# true vs observed ################################################# 

        # Set initial distances to be large since we don't know what the minimum distance is yet 
        b_had_dist_true = b_lep_dist_true = W_had_dist_true = 1000000

        # Perform jet matching for the bs, all jets, b-tagged and not b-tagged should be considered.
        for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
            # For bs:
            b_had_d_true = find_dist(b_had_true, jets[k])
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
                closest_b_had = jets[k]
            b_lep_d_true = find_dist(b_lep_true, jets[k])
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
                closest_b_lep = jets[k]

        good_jets = jets[:]
        if (b_tagging > 0):
            for m in range(len(jets)):
                # if don't include any b tagged jets and jet is b tagged OR
                # if only considering b tagged jets and jet is not b tagged
                if (b_tagging == 1 and jet_list[m, i, 5]) or (b_tagging == 2 and not jet_list[m,i,5]):
                    good_jets.remove(jets[m])
        # If there are no jets remaining in good_jets, then skip this event. Don't populate histograms.
        if not good_jets:
            continue
        
        # Consider best two jets first.
        if (len(good_jets) >= 2):
            for k in range(len(good_jets)):
                # if good_jets only contains one element, loop is skipped since range would be (1,1)
                for j in range(k + 1, len(good_jets)):                  
                    sum_vect = good_jets[k] + good_jets[j] 
                    W_had_d_true = find_dist(W_had_true, sum_vect)
                    if W_had_d_true < W_had_dist_true:
                        W_had_dist_true = W_had_d_true
                        closest_W_had = sum_vect
            W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 1
        
        # If the best double jet doesn't pass cuts, then consider three jets.
        if (len(good_jets) >= 3) and (closest_W_had.M() <= W_had_m_cutoff[0] \
            or closest_W_had.M() >= W_had_m_cutoff[1] or W_had_true_obs_pT_diff <= W_had_pT_cutoff[0] \
            or W_had_true_obs_pT_diff >= W_had_pT_cutoff[1] \
            or W_had_dist_true >= W_had_dist_cutoff[1]):
            # Reset maximum eta-phi distance.
            W_had_dist_true = 1000000
            for k in range(len(good_jets)):
                for j in range(k + 1, len(good_jets)):     
                    for l in range(j+1, len(good_jets)):
                        sum_vect = good_jets[k] + good_jets[j] + good_jets[l]
                        # Calculate eta-phi distance for current jet combo.
                        W_had_d_true = find_dist(W_had_true, sum_vect)
                        # Compare current distance to current minimum distance and update if lower.
                        if W_had_d_true < W_had_dist_true:
                            W_had_dist_true = W_had_d_true
                            closest_W_had = sum_vect
            # Calculate true - observed pT difference for the best triple jet
            W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 2

        # if there is only one jet in the list or previous matches don't pass cutoff conditions, find a single jet match
        if (len(good_jets) == 1) or ((closest_W_had.M() <= W_had_m_cutoff[0] or closest_W_had.M() >= W_had_m_cutoff[1]) \
            or (W_had_true_obs_pT_diff <= W_had_pT_cutoff[0] or W_had_true_obs_pT_diff >= W_had_pT_cutoff[1])\
            or W_had_dist_true >= W_had_dist_cutoff[1]):
            W_had_dist_true = 1000000
            # Single jets
            for k in range(len(good_jets)):
                sum_vect = good_jets[k]    
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    closest_W_had = sum_vect
            # Only calculate difference for best single jet.
            W_had_true_obs_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 0

        # Special calculations for the observed leptonic W assuming massless daughters.  
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]]
        # Observed neutrino transverse momentum from missing energy as [x, y].
        nu_pT_obs = [jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] 
        W_lep_Px_observed = muon_pT_obs[0] + nu_pT_obs[0]
        W_lep_Py_observed = muon_pT_obs[1] + nu_pT_obs[1]
        
        # Calculate the distance between true and observed phi.
        lep_phi = np.arctan2(W_lep_Py_observed, W_lep_Px_observed)
        W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
        # Calculate transverse energy assuming daughter particles are massless
        W_lep_ET_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
        W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed
        # Calculate the transverse mass
        obs_daughter_angle = np.arccos( np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs) )
        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 

        # Compare hadronic t distances
        t_had_jets = closest_b_had + closest_W_had
        t_had_dist_true = find_dist(t_had_true, t_had_jets)
        t_had_true_obs_pT_diff = t_had_true.Pt() - t_had_jets.Pt()
        
        # Compare leptonic t distances
        t_lep_x = W_lep_Px_observed + closest_b_lep.Px()
        t_lep_y = W_lep_Py_observed + closest_b_lep.Py()
        obs_t_phi = np.arctan2(t_lep_y, t_lep_x)
        t_lep_dist_true = np.abs( min( np.abs(t_lep_true.Phi()-obs_t_phi), 2*np.pi-np.abs(t_lep_true.Phi() - obs_t_phi) ) )
        t_lep_pT_observed = np.sqrt( t_lep_x**2 + t_lep_y**2)
        t_lep_true_obs_pT_diff = t_lep_true.Et() - t_lep_pT_observed

        # b quark calculations
        b_had_true_obs_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
        b_lep_true_obs_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

        ################################################# predicted vs observed #################################################

        # Once the optimal jets have been matched in the previous section, 
        #  the eta-phi distances can be calculated between the predicted and observed variables
        #  with no further work.

        # Leptonic W
        # Calculate the distance between predicted and observed phi. 
        # No eta distance for comparison with truth vs. obs and pred vs. true
        W_lep_dphi_po = np.abs( min( np.abs(W_lep_fitted.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_fitted.Phi()-lep_phi) ) )
        W_lep_R_po = np.sqrt(W_lep_dphi_po**2)
        # Hadronic W
        W_had_R_po = find_dist( W_had_fitted, closest_W_had )

        # Leptonic b
        b_lep_R_po = find_dist( b_lep_fitted, closest_b_lep )
        # Hadronic b
        b_had_R_po = find_dist( b_had_fitted, closest_b_had )

        # Leptonic t
        t_lep_dphi_po = min(np.abs(t_lep_fitted.Phi()-obs_t_phi), 2*np.pi-np.abs(t_lep_fitted.Phi()-obs_t_phi))
        t_lep_R_po = np.sqrt(t_lep_dphi_po**2) # Again, no eta
        # Hadronic t
        t_had_R_po = find_dist( t_had_fitted, t_had_jets )

        ############################################## check whether each event passes cuts #################################################
        # counter for hadronic W
        # Update tally for which jet combination is the closest
        W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
        W_had_pT_cut = (W_had_true_obs_pT_diff >= W_had_pT_cutoff[0] and W_had_true_obs_pT_diff <= W_had_pT_cutoff[1])
        W_had_dist_cut = (W_had_dist_true <= W_had_dist_cutoff[1]) 
        # All W_had cuts must be satisfied simultaneously.
        good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)

        W_had_jets[jet_combo_index] += 1.
        W_had_total_cuts[jet_combo_index] += good_W_had
        W_had_m_cuts[jet_combo_index] += W_had_m_cut
        W_had_pT_cuts[jet_combo_index] += W_had_pT_cut
        W_had_dist_cuts[jet_combo_index] += W_had_dist_cut

        # counter for lep W
        W_lep_ET_cut = (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_ET_diff <= W_lep_ET_cutoff[1])
        W_lep_dist_cut = (W_lep_dist_true <= W_lep_dist_cutoff[1]) 
        good_W_lep = (W_lep_ET_cut and W_lep_dist_cut)

        W_lep_total_cuts += good_W_lep
        W_lep_ET_cuts += W_lep_ET_cut
        W_lep_dist_cuts += W_lep_dist_cut

        # counter for hadronic b
        b_had_pT_cut = (b_had_true_obs_pT_diff >= b_had_pT_cutoff[0] and b_had_true_obs_pT_diff <= b_had_pT_cutoff[1])
        b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
        good_b_had = (b_had_pT_cut and b_had_dist_cut)

        b_had_total_cuts += good_b_had
        b_had_pT_cuts += b_had_pT_cut
        b_had_dist_cuts += b_had_dist_cut

        # counter for leptonic b
        b_lep_pT_cut = (b_lep_true_obs_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_obs_pT_diff <= b_lep_pT_cutoff[1])
        b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
        good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

        b_lep_total_cuts += good_b_lep
        b_lep_pT_cuts += b_lep_pT_cut
        b_lep_dist_cuts += b_lep_dist_cut

        # Good events must pass cuts on all partons.
        good_event += (good_b_had and good_b_lep and good_W_had and good_W_lep)

        ################################################# populate histograms #################################################

        # Populate histograms if all events are to be plotted or we are only dealing with a good event 
        if (not plot_cuts) or (plot_cuts and (good_b_had and good_b_lep and good_W_had and good_W_lep)):

            # Leptonic b
            hists['lep_b_dist_true_v_obs'].Fill(np.float(b_lep_dist_true))
            hists['lep_b_dist_pred_v_true'].Fill(np.float(b_lep_R))
            hists['lep_b_dist_pred_v_obs'].Fill(np.float(b_lep_R_po))
            # Invariant mass:
            hists['lep_b_obs_mass'].Fill(closest_b_lep.M())
            # Jet matching criteria correlation plots
            hists['lep_b_corr_dist_true_v_obs_mass'].Fill(closest_b_lep.M(), b_lep_dist_true) 
            hists['lep_b_corr_pT_diff_true_v_obs_mass'].Fill(closest_b_lep.M(), b_lep_true_obs_pT_diff) 
            hists['lep_b_corr_pT_diff_dist_true_v_obs'].Fill(b_lep_dist_true, b_lep_true_obs_pT_diff)
            # Closest PT difference vs. PT
            hists['lep_b_true_obs_pT'].Fill(closest_b_lep.Pt())
            hists['lep_b_true_obs_pT_diff'].Fill(b_lep_true_obs_pT_diff)
            hists['lep_b_corr_pT_diff_pT_obs'].Fill(closest_b_lep.Pt(), b_lep_true_obs_pT_diff) 

            # Hadronic b
            hists['had_b_dist_true_v_obs'].Fill(np.float(b_had_dist_true))
            hists['had_b_dist_pred_v_true'].Fill(np.float(b_had_R))
            hists['had_b_dist_pred_v_obs'].Fill(np.float(b_had_R_po))
            # Invariant mass:
            hists['had_b_obs_mass'].Fill(closest_b_had.M())
            # Jet matching criteria correlation plots
            hists['had_b_corr_dist_true_v_obs_mass'].Fill(closest_b_had.M(), b_had_dist_true) 
            hists['had_b_corr_pT_diff_true_v_obs_mass'].Fill(closest_b_had.M(), b_had_true_obs_pT_diff) 
            hists['had_b_corr_pT_diff_dist_true_v_obs'].Fill(b_had_dist_true, b_had_true_obs_pT_diff)
            # Closest PT difference vs. PT
            hists['had_b_true_obs_pT'].Fill(closest_b_had.Pt())
            hists['had_b_true_obs_pT_diff'].Fill(b_had_true_obs_pT_diff)
            hists['had_b_corr_pT_diff_pT_obs'].Fill(closest_b_had.Pt(), b_had_true_obs_pT_diff) 

            # Leptonic t
            hists['lep_t_dist_true_v_obs'].Fill(np.float(t_lep_dist_true))
            hists['lep_t_dist_pred_v_true'].Fill(np.float(t_lep_R))
            hists['lep_t_dist_pred_v_obs'].Fill(np.float(t_lep_R_po))
            hists['lep_t_transverse_energy_diff'].Fill(np.float(t_lep_true_obs_pT_diff))
            hists['lep_t_corr_ET_diff_dist_true_v_obs'].Fill(t_lep_dist_true, t_lep_true_obs_pT_diff)
            # Hadronic t
            hists['had_t_dist_true_v_obs'].Fill(np.float(t_had_dist_true))
            hists['had_t_dist_pred_v_true'].Fill(np.float(t_had_R))
            hists['had_t_dist_pred_v_obs'].Fill(np.float(t_had_R_po))
            hists['had_t_obs_mass'].Fill(t_had_jets.M())
            hists['had_t_true_obs_pT_diff'].Fill(t_had_true_obs_pT_diff)
            hists['had_t_corr_dist_true_v_obs_mass'].Fill(t_had_jets.M(), t_had_dist_true)
            hists['had_t_corr_pT_diff_true_v_obs_mass'].Fill(t_had_jets.M(), t_had_true_obs_pT_diff)
            hists['had_t_corr_pT_diff_dist_true_v_obs'].Fill(t_had_dist_true, t_had_true_obs_pT_diff)

            # Leptonic W
            hists['lep_W_dist_true_v_obs'].Fill(np.float(W_lep_dist_true))
            hists['lep_W_dist_pred_v_true'].Fill(np.float(W_lep_R))
            hists['lep_W_dist_pred_v_obs'].Fill(np.float(W_lep_R_po))
            hists['lep_W_transverse_mass_obs'].Fill(np.float(met_obs))
            # Matching scatterplots
                # hists['lep_W_scat_ET_diff_dist_true_v_obs'].Fill(W_lep_dist_true, W_lep_ET_diff) 
            hists['lep_W_corr_ET_diff_dist_true_v_obs'].Fill(W_lep_dist_true, W_lep_ET_diff) 
            # Closest ET difference vs. ET
            hists['lep_W_transverse_energy_obs'].Fill(np.float(W_lep_ET_observed))
            hists['lep_W_transverse_energy_diff'].Fill(np.float(W_lep_ET_diff))
            hists['lep_W_corr_ET_diff_ET_obs'].Fill(W_lep_ET_observed, W_lep_ET_diff)

            # Hadronic W
            hists['had_W_dist_true_v_obs'].Fill(np.float(W_had_dist_true))
            hists['had_W_dist_pred_v_true'].Fill(np.float(W_had_R))
            hists['had_W_dist_pred_v_obs'].Fill(np.float(W_had_R_po))
            # Invariant mass:
            hists['had_W_obs_mass'].Fill(closest_W_had.M())
            # Jet matching criteria correlation plots
            hists['had_W_corr_mass_dist_true_v_obs'].Fill(closest_W_had.M(), W_had_dist_true) 
            hists['had_W_corr_mass_Pt_true_v_obs'].Fill(closest_W_had.M(), W_had_true_obs_pT_diff) 
            hists['had_W_corr_dist_Pt_true_v_obs'].Fill(W_had_dist_true, W_had_true_obs_pT_diff)  
            # Closest pT difference vs. pT
            hists['had_W_true_obs_pT'].Fill(np.float(closest_W_had.Pt()))
            hists['had_W_true_obs_pT_diff'].Fill(np.float(W_had_true_obs_pT_diff))
            # Plots that depend on whether a 1,2, or 3-jet sum is the best match to truth:
            if jet_combo_index == 0:
                hists['had_W_true_1_pT_diff'].Fill(np.float(W_had_true_obs_pT_diff))
                hists['had_W_obs_1_mass'].Fill(closest_W_had.M())
                hists['had_W_obs_1_mass_log'].Fill(closest_W_had.M())
                hists['had_W_corr_1_mass_dist_true_v_obs'].Fill(closest_W_had.M(), W_had_dist_true)
                hists['had_W_corr_1_mass_Pt_true_v_obs'].Fill(closest_W_had.M(), W_had_true_obs_pT_diff)
                hists['had_W_corr_1_dist_Pt_true_v_obs'].Fill(W_had_dist_true, W_had_true_obs_pT_diff)
                hists['had_W_1_dist'].Fill(np.float(W_had_dist_true))
            elif jet_combo_index == 1:
                hists['had_W_true_2_pT_diff'].Fill(np.float(W_had_true_obs_pT_diff))
                hists['had_W_obs_2_mass'].Fill(closest_W_had.M())
                hists['had_W_corr_2_mass_dist_true_v_obs'].Fill(closest_W_had.M(), W_had_dist_true)
                hists['had_W_corr_2_mass_Pt_true_v_obs'].Fill(closest_W_had.M(), W_had_true_obs_pT_diff)
                hists['had_W_corr_2_dist_Pt_true_v_obs'].Fill(W_had_dist_true, W_had_true_obs_pT_diff)
                hists['had_W_2_dist'].Fill(np.float(W_had_dist_true))
            elif jet_combo_index == 2:
                hists['had_W_true_3_pT_diff'].Fill(np.float(W_had_true_obs_pT_diff))
                hists['had_W_obs_3_mass'].Fill(closest_W_had.M())
                hists['had_W_corr_3_mass_dist_true_v_obs'].Fill(closest_W_had.M(), W_had_dist_true)
                hists['had_W_corr_3_mass_Pt_true_v_obs'].Fill(closest_W_had.M(), W_had_true_obs_pT_diff)
                hists['had_W_corr_3_dist_Pt_true_v_obs'].Fill(W_had_dist_true, W_had_true_obs_pT_diff)
                hists['had_W_3_dist'].Fill(np.float(W_had_dist_true))

    # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))
    print('NOTE: some percentages do not reach 100%, as events where no Hadronic W can be matched after removing the b-tagged jets are skipped (all jets are b-tagged)')
    print('\n==================================================================\n')
    print('Cut Criteria')
    print('Hadronic W, mass: {}, pT: {}, distance: {}'.format(W_had_m_cutoff, W_had_pT_cutoff, W_had_dist_cutoff))
    print('Leptonic W, E_T: {}, dist: {}'.format(W_lep_ET_cutoff, W_lep_dist_cutoff))
    print('Hadronic b, pT: {}, distance: {}'.format(b_had_pT_cutoff, b_had_dist_cutoff))
    print('Leptonic b, pT: {}, distance: {}'.format(b_lep_pT_cutoff, b_lep_dist_cutoff))
    print('\n==================================================================\n')

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets, before applying cuts on events matched to 1 jet:")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_jets[0]/n_events, int(W_had_jets[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_jets[1]/n_events, int(W_had_jets[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_jets[2]/n_events, int(W_had_jets[2])))

    print("Number of events satisfying all hadronic W cut criteria, as a percentage of their respective categories before applying cuts:")
    print('{}% Total Hadronic Ws within cuts, {} events'.format(100.*sum(W_had_total_cuts)/n_events, int(sum(W_had_total_cuts))))
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/W_had_jets[0], int(W_had_total_cuts[0]), int(W_had_jets[0])))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/W_had_jets[1], int(W_had_total_cuts[1]), int(W_had_jets[1])))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/W_had_jets[2], int(W_had_total_cuts[2]), int(W_had_jets[2])))

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets after cuts are applied: ")
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/sum(W_had_total_cuts), int(W_had_total_cuts[0]), int(sum(W_had_total_cuts))))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/sum(W_had_total_cuts), int(W_had_total_cuts[1]), int(sum(W_had_total_cuts))))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/sum(W_had_total_cuts), int(W_had_total_cuts[2]), int(sum(W_had_total_cuts))))

    print("Number of events satisfying hadronic W mass cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[0]/W_had_jets[0], int(W_had_m_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[1]/W_had_jets[1], int(W_had_m_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_m_cuts[2]/W_had_jets[2], int(W_had_m_cuts[2])))
    print("Number of events satisfying hadronic W pT cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[0]/W_had_jets[0], int(W_had_pT_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[1]/W_had_jets[1], int(W_had_pT_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_pT_cuts[2]/W_had_jets[2], int(W_had_pT_cuts[2])))
    print("Number of events satisfying hadronic W distance cut criteria")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[0]/W_had_jets[0], int(W_had_dist_cuts[0])))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[1]/W_had_jets[1], int(W_had_dist_cuts[1])))
    print('{}% 3 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[2]/W_had_jets[2], int(W_had_dist_cuts[2])))
    print('\n==================================================================\n')
    print("Number of events satisfying all leptonic W cut criteria")
    print('{}% , {} events\n'.format(100.*W_lep_total_cuts/n_events, int(W_lep_total_cuts)))
    print("Number of events satisfying leptonic W ET cut criteria")
    print('{}%, {} events'.format(100.*W_lep_ET_cuts/n_events, int(W_lep_ET_cuts)))
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

    # hists_key = []
    # corr_key = []
    # for key in hists:
    #     if 'corr' not in key:
    #         hists_key.append(key)
    #     else:
    #         corr_key.append(key)

    # for key in hists_key:
    #     plot_hists(key, hists[key], outputdir+subdir)

    # # The following few lines must be run only once for all correlation plots, 
    # #  so the correlation plots must be separated out from the other histograms.   
    # from AngryTops.features import *
    # from AngryTops.Plotting.PlottingHelper import *
    # gStyle.SetPalette(kGreyScale)
    # gROOT.GetColor(52).InvertPalette()

    # for key in corr_key:
    #     plot_corr(key, hists[key], outputdir+subdir)