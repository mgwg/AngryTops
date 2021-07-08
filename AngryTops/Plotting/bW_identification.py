import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle

representation = sys.argv[2]
outputdir = sys.argv[1]
event_type = 0
if len(sys.argv) > 3:
    event_type = sys.argv[3]
date = ''
if len(sys.argv) > 4:
    date = sys.argv[4]


subdir = '/closejets_img_btag_none/'
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = "None" # "None": Do not consider any b-tagged jets.
                   # "All": Consider all jets, both b-tagged and not b-tagged
                   # "Only": Consider only b-tagged jets

# Helper function to create histograms of eta-phi distance distributions
def MakeP4(y, m):
    """
    Form the momentum vector.
    """
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[2]
    # Construction of momentum vector depends on the representation of the input
    if representation == "pxpypzE":
        E  = y[3]
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypzM":
        M  = y[3]
        E = np.sqrt(p0**2 + p1**2 + p2**2 + M**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "pxpypz" or representation == "pxpypzEM":
        E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
        p4.SetPxPyPzE(p0, p1, p2, E)
    elif representation == "ptetaphiE":
        E  = y[3]
        p4.SetPtEtaPhiE(p0, p1, p2, E)
    elif representation == "ptetaphiM":
        M  = y[3]
        p4.SetPtEtaPhiM(p0, p1, p2, M)
    elif representation == "ptetaphi" or representation == "ptetaphiEM":
        p4.SetPtEtaPhiM(p0, p1, p2, m)
    else:
        raise Exception("Invalid Representation Given: {}".format(representation))
    return p4

def find_dist(a, b):
    '''
    a, b are both TLorentz Vectors
    returns the eta-phi distances between true and sum_vect
    '''
    dphi_true = min(np.abs(a.Phi() - b.Phi()), 2*np.pi-np.abs(a.Phi() - b.Phi()))
    deta_true = a.Eta() - b.Eta()
    d_true = np.sqrt(dphi_true**2 + deta_true**2)
    return d_true

# load data
predictions = np.load(outputdir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']

particles_shape = (true.shape[1], true.shape[2])
print("jets shape", jets.shape)
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

        # Retain b-tagging states depending on value of b-tagging
        if b_tagging != "All":
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
        else:
            # Don't care about b-tagging states
            # Rescale the jets array
            jets_lep = jets[:,:6]
            jets_jets = jets[:,6:] # remove muon column
            jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array
            jets_jets = np.delete(jets_jets, 5, 2) # delete the b-tagging states
            jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) # reshape into 25 element long array
            jets_lep = lep_scalar.inverse_transform(jets_lep)
            jets_jets = jets_scalar.inverse_transform(jets_jets) # scale values ... ?
            jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))#I think this is the final 6x6 array the arxiv paper was talking about - 5 x 5 array containing jets (1 per row) and corresponding px, py, pz, E, m

if not scaling:
    jets_lep = jets[:,:6]
    jets_jets = jets[:,6:]
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
    jets_jets = np.delete(jets_jets, 5, 2)
    jets_jets = jets_jets.reshape((jets_jets.shape[0], 25))
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))

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
jet_list = np.stack([jet_1, jet_2, jet_3, jet_4, jet_5]) # I suppose you could use a dictionary here but the keys would just be indices.

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

# A meaningful name for the number of events
n_events = true.shape[0]

# define indices
event_index = range(n_events)
if event_type == "4":
    event_index = np.where(jet_5 == 0)
    event_index = np.unique(event_index[0])
    subdir += "_four" 
elif event_type == "5":
    event_index = np.nonzero(jet_5)
    event_index = np.unique(event_index[0])
    subdir += "_five"

# make histograms to be filled
hists = {}

# Leptonic W
# Miscellaneous histograms
hists['leptonic_W_transverse_mass_observed'] = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Observed", 50, 0, 250)#120)
hists['leptonic_W_transverse_mass_observed'].SetTitle("W Leptonic Transverse Mass, Observed;Leptonic (GeV);A.U.")
hists['leptonic_W_transverse_energy_diff'] = TH1F("W_lep_Et_d","W Leptonic Transverse Energy Difference, Observed - Truth", 50, -120, 120)
hists['leptonic_W_transverse_energy_diff'].SetTitle("W Leptonic Transverse Energy Difference, Observed - Truth;Leptonic (GeV);A.U.")
hists['leptonic_W_transverse_energy_diff_high'] = TH1F("W_lep_Et_d","W Leptonic Transverse Energy Difference (High Energy), Observed - Truth", 50, -120, 120)
hists['leptonic_W_transverse_energy_diff_high'].SetTitle("W Leptonic Transverse Energy Difference (High Energy), Observed - Truth;Leptonic (GeV);A.U.")
# True vs. obs
hists['leptonic_W_dist_true_v_obs'] = TH1F("h_W_lep_true","W Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_W_dist_true_v_obs'].SetTitle("W Leptonic #phi distances, True vs Observed; Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_W_dist_pred_v_true'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_W_dist_pred_v_true'].SetTitle("W Leptonic #phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_W_dist_pred_v_obs'] = TH1F("h_W_lep_d","W Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_W_dist_pred_v_obs'].SetTitle("W Leptonic #phi distances, Predicted vs Observed; Leptonic (radians);A.U.")

# Hadronic W
hists['hadronic_W_true_pT_diff'] = TH1F("h_pT_W_had_diff","W Hadronic {p}_{T} diffs, True - Observed", 80, -400, 400)
hists['hadronic_W_true_pT_diff'].SetTitle("W Hadronic {p}_{T} diffs, True - Observed; Hadronic (GeV); A.U.")
# True vs. obs
hists['hadronic_W_dist_true_v_obs'] = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_W_dist_true_v_obs'].SetTitle("W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_W_dist_pred_v_true'] = TH1F("h_W_had_pred","W Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_W_dist_pred_v_true'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth; Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_W_dist_pred_v_obs'] = TH1F("h_W_had_d","W Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_W_dist_pred_v_obs'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['hadronic_W_obs_1_mass'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['hadronic_W_obs_1_mass'].SetTitle("1 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['hadronic_W_obs_1_mass_log'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['hadronic_W_obs_1_mass_log'].SetTitle("1 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['hadronic_W_obs_2_mass'] = TH1F("W_had_m","2 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['hadronic_W_obs_2_mass'].SetTitle("2 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['hadronic_W_obs_3_mass'] = TH1F("W_had_m","3 Jet W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['hadronic_W_obs_3_mass'].SetTitle("3 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['hadronic_W_obs_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, Observed", 60, 0., 300. )
hists['hadronic_W_obs_mass'].SetTitle("W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching criteria scatterplots
# invariant mass vs eta-phi dist
hists['hadronic_W_1_scat_dist_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_1_scat_dist_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 1-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_2_scat_dist_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_2_scat_dist_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 2-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_3_scat_dist_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_3_scat_dist_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 3-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_scat_dist_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_scat_dist_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic #eta-#phi Distances vs. Invariant Mass")
# invariant mass vs Pt difference
hists['hadronic_W_1_scat_pT_diff_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_1_scat_pT_diff_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 1-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_2_scat_pT_diff_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_2_scat_pT_diff_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 2-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_3_scat_pT_diff_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_3_scat_pT_diff_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic 3-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_scat_pT_diff_true_v_obs_mass'] = TH2F("W_had_scat", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_scat_pT_diff_true_v_obs_mass'].SetTitle("Scatterplot of W Hadronic {p}_{T} Diffs vs. Invariant Mass") 
# eta-phi dist vs. Pt difference
hists['hadronic_W_1_scat_pT_diff_dist_true_v_obs'] = TH2F("W_had_scat", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_1_scat_pT_diff_dist_true_v_obs'].SetTitle("Scatterplot of W Hadronic 1-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_2_scat_pT_diff_dist_true_v_obs'] = TH2F("W_had_scat", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_2_scat_pT_diff_dist_true_v_obs'].SetTitle("Scatterplot of W Hadronic 2-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_3_scat_pT_diff_dist_true_v_obs'] = TH2F("W_had_scat", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_3_scat_pT_diff_dist_true_v_obs'].SetTitle("Scatterplot of W Hadronic 3-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_scat_pT_diff_dist_true_v_obs'] = TH2F("W_had_scat", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_scat_pT_diff_dist_true_v_obs'].SetTitle("Scatterplot of W Hadronic W Hadronic {p}_{T} Diffs vs. #eta-#phi Distances")
# Closest 1-jet Pt difference vs. closest 2-jet Pt difference
hists['hadronic_W_scat_1_pT_diff_v_2_pT_diff'] = TH2F("W_had_scat", ";W Hadronic Closest 1-jet {p}_{T} Diffs; W Hadronic Closest 2-jet {p}_{T} Diffs", 50, -200, 200, 50, -200, 200)
hists['hadronic_W_scat_1_pT_diff_v_2_pT_diff'].SetTitle("Scatterplot of W Hadronic Closest 1-jet {p}_{T} Diffs vs. W Hadronic Closest 2-jet {p}_{T} Diffs")
# Jet matching criteria correlation plots
# invariant mass vs eta-phi dist
hists['hadronic_W_1_corr_dist_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_1_corr_dist_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 1-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_2_corr_dist_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_2_corr_dist_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 2-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_3_corr_dist_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_3_corr_dist_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 3-jet #eta-#phi Distances vs. Invariant Mass")
hists['hadronic_W_corr_dist_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic #eta-#phi Distances, True vs Observed", 50, 0, 300, 50, 0, 3.2)
hists['hadronic_W_corr_dist_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic #eta-#phi Distances vs. Invariant Mass")
# invariant mass vs Pt difference
hists['hadronic_W_1_corr_pT_diff_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_1_corr_pT_diff_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 1-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_2_corr_pT_diff_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_2_corr_pT_diff_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 2-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_3_corr_pT_diff_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_3_corr_pT_diff_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic 3-jet {p}_{T} Diffs vs. Invariant Mass") 
hists['hadronic_W_corr_pT_diff_true_v_obs_mass'] = TH2F("W_had_corr", ";W Hadronic Invariant Mass [GeV]; W Hadronic {p}_{T} Diff, True - Observed [GeV]", 50, 0, 300, 50, -200, 200)
hists['hadronic_W_corr_pT_diff_true_v_obs_mass'].SetTitle("Correlation plot of W Hadronic {p}_{T} Diffs vs. Invariant Mass") 
# eta-phi dist vs. Pt difference
hists['hadronic_W_1_corr_pT_diff_dist_true_v_obs'] = TH2F("W_had_corr", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_1_corr_pT_diff_dist_true_v_obs'].SetTitle("Correlation plot of W Hadronic 1-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_2_corr_pT_diff_dist_true_v_obs'] = TH2F("W_had_corr", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_2_corr_pT_diff_dist_true_v_obs'].SetTitle("Correlation plot of W Hadronic 2-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_3_corr_pT_diff_dist_true_v_obs'] = TH2F("W_had_corr", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_3_corr_pT_diff_dist_true_v_obs'].SetTitle("Correlation plot of W Hadronic 3-jet {p}_{T} Diffs vs. #eta-#phi Distances")
hists['hadronic_W_corr_pT_diff_dist_true_v_obs'] = TH2F("W_had_corr", ";W Hadronic #eta-#phi Distances, True vs Observed; W Hadronic {p}_{T} Diff [GeV]", 50, 0, 3.2, 50, -200, 200)
hists['hadronic_W_corr_pT_diff_dist_true_v_obs'].SetTitle("Correlation plot of W Hadronic W Hadronic {p}_{T} Diffs vs. #eta-#phi Distances")
# Closest 1-jet Pt difference vs. closest 2-jet Pt difference
hists['hadronic_W_corr_1_pT_diff_v_2_pT_diff'] = TH2F("W_had_corr", ";W Hadronic Closest 1-jet {p}_{T} Diffs; W Hadronic Closest 2-jet {p}_{T} Diffs", 50, -200, 200, 50, -200, 200)
hists['hadronic_W_corr_1_pT_diff_v_2_pT_diff'].SetTitle("Correlation plot of W Hadronic Closest 1-jet {p}_{T} Diffs vs. W Hadronic Closest 2-jet {p}_{T} Diffs")

# Leptonic b
# True vs. obs
hists['leptonic_b_dist_true_v_obs'] = TH1F("h_b_lep_true","b Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_b_dist_true_v_obs'].SetTitle("b Leptonic #eta-#phi distances, True vs Observed;Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_b_dist_pred_v_true'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_b_dist_pred_v_true'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_b_dist_pred_v_obs'] = TH1F("h_b_lep_d","b Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_b_dist_pred_v_obs'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Observed; Leptonic (radians);A.U.")

# Hadronic b
# True vs. obs
hists['hadronic_b_dist_true_v_obs'] = TH1F("h_b_had_true","b Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_b_dist_true_v_obs'].SetTitle("b Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_b_dist_pred_v_true'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_b_dist_pred_v_true'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth;Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_b_dist_pred_v_obs'] = TH1F("h_b_had_d","b Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_b_dist_pred_v_obs'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")

# Leptonic t
# True vs. obs
hists['leptonic_t_dist_true_v_obs'] = TH1F("h_t_lep_true","t Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_t_dist_true_v_obs'].SetTitle("t Leptonic #phi distances, True vs Observed;Leptonic (radians);A.U.")
# Pred vs. true
hists['lep_t_dist_pred_v_true'] = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth", 50, 0, 3)
hists['lep_t_dist_pred_v_true'].SetTitle("t Leptonic #phi distances, Predicted vs Truth;Leptonic (radians);A.U.")
# Pred vs. obs
hists['lep_t_dist_pred_v_obs'] = TH1F("h_t_lep_d","t Leptonic Distances, Predicted vs Observed", 50, 0, 3)
hists['lep_t_dist_pred_v_obs'].SetTitle("t Leptonic #phi distances, Predicted vs Observed; Leptonic (radians);A.U.")

# Hadronic t
# True vs. obs
hists['hadronic_t_dist_true_v_obs'] = TH1F("h_t_had_true","t Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_t_dist_true_v_obs'].SetTitle("t Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_t_dist_pred_v_true'] = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_t_dist_pred_v_true'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth;Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_t_dist_pred_v_obs'] = TH1F("h_t_had_d","t Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_t_dist_pred_v_obs'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")

# Function to make histograms
def make_histograms():
    # define tolerance limits
    b_lep_dist_t_lim = 0.39
    b_had_dist_t_lim = 0.39
    t_lep_dist_t_lim = 0.80
    t_had_dist_t_lim = 0.80
    W_lep_dist_t_lim = 1.28
    W_had_dist_t_lim = 1.28

    good_b_lep = good_b_had = 0.
    good_W_lep = good_W_had = 0.
    bad_b_lep = bad_b_had = 0.
    bad_W_lep = bad_W_had = 0.

    high_E = 0.
    w_had_jets = [0., 0., 0.] # List of number of events best matched by 1,2,3 jets respectively.

    for i in event_index: # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W )
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W)

        W_lep_true   = MakeP4( y_true_W_lep[i], m_W )
        W_lep_fitted = MakeP4( y_fitted_W_lep[i],  m_W)

        b_had_true   = MakeP4( y_true_b_had[i], m_b )
        b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b )

        b_lep_true   = MakeP4( y_true_b_lep[i], m_b )
        b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b)

        t_had_true   = MakeP4( y_true_t_had[i], m_t )
        t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t )

        t_lep_true   = MakeP4( y_true_t_lep[i], m_t )
        t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t)

        jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4])

        jet_1_vect = MakeP4(jet_1[i], jet_1[i][4])
        jet_2_vect = MakeP4(jet_2[i], jet_2[i][4])
        jet_3_vect = MakeP4(jet_3[i], jet_3[i][4])
        jet_4_vect = MakeP4(jet_4[i], jet_4[i][4])
        jet_5_vect = MakeP4(jet_5[i], jet_5[i][4])
        
        jets = []
        # add list containing jets of correspoonding event
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
        if not np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)

        # Special calculations for the observed leptonic W

        # Observed transverse mass distribution is square root of 2* Etmiss 
        #  * Transverse angle between daughter particles, assuming that they are massless.
                
        # First, find the transverse angle between the daughter particles, which is the angle 
        #  between the muon momentum vector and the missing Et vector. For the muon, we have 
        #  px and py, but for the missing Et, we have the vector in polar representation.
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]] # Observed transverse momentum of muon
        # Convert missing transverse energy to a momentum
        nu_pT_obs = [jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] # Observed neutrino transverse momentum from missing energy as [x, y].
        # Now, calculate the angle
        obs_daughter_angle = np.arccos( np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs) )
        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 

        # Calculate the transverse energy of the observed leptonic W summing rest mass and total transverse momentum.
        W_lep_Px_observed = muon_pT_obs[0] + nu_pT_obs[0]
        W_lep_Py_observed = muon_pT_obs[1] + nu_pT_obs[1]
        W_Et_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
        # difference
        W_lep_Et_diff = W_Et_observed - W_lep_true.Et()

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
        b_had_dist_true = 1000
        b_lep_dist_true = 1000
        t_had_dist_true = 1000
        t_lep_dist_true = 1000
        W_had_dist_true_start = 10000000
        W_had_dist_true = W_had_dist_true_start

        # Variables to hold minimum distances for 1 and 2-jets separately.
        W_had_dist_true_1 = 10000000
        W_had_dist_true_2 = 10000000

            # Perform jet matching for the bs and Ws
        for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
            # For bs:
            b_had_d_true = find_dist(b_had_true, jets[k])
            b_lep_d_true = find_dist(b_lep_true, jets[k])
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
                closest_b_had = jets[k]
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
                closest_b_lep = jets[k]

            # For hadronic Ws
            # If all jets are to be considered or b-tagged jets are not to be considered and jet k is not b-tagged 
            #  or only b-tagged jets are to be considered and jet k is b-tagged
            if (b_tagging == "All") \
                or (b_tagging == "None" and not jet_list[k,i,5]) \
                    or (b_tagging == "Only" and jet_list[k,i,5]): # k ranges from 0 to 4 or 5 depending on event type
                # Go through each 1,2,3 combination of jets that are not b-tagged and check their sum
                sum_vect = jets[k]    
                # Single jets
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
                    closest_W_had = sum_vect
                    w_jets = 0
                # Comparison for best single jet
                if W_had_d_true < W_had_dist_true_1:
                    W_had_true_pT_diff_1 = W_had_true.Pt() - sum_vect.Pt()

                # Dijets
                for j in range(k + 1, len(jets)):
                    # j ranges from k+1 to 4 or 5 depending on event type     
                    if (b_tagging == "All") \
                        or (b_tagging == "None" and not jet_list[j,i,5]) \
                            or (b_tagging == "Only" and jet_list[j,i,5]):
                        sum_vect = jets[k] + jets[j] 
                        W_had_d_true = find_dist(W_had_true, sum_vect)
                        if W_had_d_true < W_had_dist_true:
                            W_had_dist_true = W_had_d_true
                            W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
                            closest_W_had = sum_vect
                            w_jets = 1
                        # Comparison for best dijet
                        if W_had_d_true < W_had_dist_true_2:
                            W_had_true_pT_diff_2 = W_had_true.Pt() - sum_vect.Pt()

                # Trijets
                        for l in range(j+1, len(jets)):
                            # l ranges from j+k+1 to 4 or 5 depending on event type
                            if (b_tagging == "All") \
                                or (b_tagging == "None" and not jet_list[l,i,5]) \
                                    or (b_tagging == "Only" and jet_list[l,i,5]):
                                sum_vect = jets[k] + jets[j] + jets[l]
                                W_had_d_true = find_dist(W_had_true, sum_vect)
                                if W_had_d_true < W_had_dist_true:
                                    W_had_dist_true = W_had_d_true
                                    W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
                                    closest_W_had = sum_vect
                                    w_jets = 2

        # If b-tagged jets are not to be considered and all jets are b-tagged 
        #  or only b-tagged jets are to be considered and no jet is b-tagged, 
        #  skip the event.
        if W_had_dist_true == W_had_dist_true_start: # If there are no jets to be matched for this event,
            continue


        w_had_jets[w_jets] += 1    
        
        # Calculate W leptonic distances
        # Add muon transverse momentum components to missing momentum components
        lep_x = jet_mu[i][0] + nu_pT_obs[0]
        lep_y = jet_mu[i][1] + nu_pT_obs[1]
        # Calculate phi using definition in Kuunal's report 
        lep_phi = np.arctan2(lep_y, lep_x)
        # Calculate the distance between true and observed phi.
        W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )

        # Compare hadronic t distances
        t_had_jets = closest_b_had + closest_W_had
        t_had_dist_true = find_dist(t_had_true, t_had_jets)
        
        # Compare leptonic t distances
        t_lep_x = lep_x + closest_b_lep.Px()
        t_lep_y = lep_y + closest_b_lep.Py()
        obs_t_phi = np.arctan2(t_lep_y, t_lep_x)
        t_lep_dist_true = np.abs( min( np.abs(t_lep_true.Phi()-obs_t_phi), 2*np.pi-np.abs(t_lep_true.Phi() - obs_t_phi) ) )

        if (b_lep_dist_true <= b_lep_dist_t_lim): # if minimum distance is less than the tolearance limits, everything is ok
            good_b_lep += 1
        else:
            bad_b_lep += 1
        if (b_had_dist_true <= b_had_dist_t_lim):
            good_b_had += 1
        else:
            bad_b_had += 1
        if (W_lep_dist_true <= W_lep_dist_t_lim): # mismatch between W_lep_dist_true and good_W_lep
            good_W_lep += 1
        else:
            bad_W_lep += 1
        if (W_had_dist_true <= W_had_dist_t_lim):
            good_W_had += 1
        else:
            bad_W_had += 1


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

        ################################################# populate histograms and check percentages #################################################

        # Leptonic b
        hists['lep_b_dist_true_v_obs'].Fill(np.float(b_lep_dist_true))
        hists['lep_b_dist_pred_v_true'].Fill(np.float(b_lep_R))
        hists['lep_b_dist_pred_v_obs'].Fill(np.float(b_lep_R_po))
        # Hadronic b
        hists['had_b_dist_true_v_obs'].Fill(np.float(b_had_dist_true))
        hists['had_b_dist_pred_v_true'].Fill(np.float(b_had_R))
        hists['had_b_dist_pred_v_obs'].Fill(np.float(b_had_R_po))

        # Leptonic t
        hists['lep_t_dist_true_v_obs'].Fill(np.float(t_lep_dist_true))
        hists['lep_t_dist_pred_v_true'].Fill(np.float(t_lep_R))
        hists['lep_t_dist_pred_v_obs'].Fill(np.float(t_lep_R_po))
        # Hadronic t
        hists['had_t_dist_true_v_obs'].Fill(np.float(t_had_dist_true))
        hists['had_t_dist_pred_v_true'].Fill(np.float(t_had_R))
        hists['had_t_dist_pred_v_obs'].Fill(np.float(t_had_R_po))

        # Leptonic W
        hists['lep_W_dist_true_v_obs'].Fill(np.float(W_lep_dist_true))
        hists['lep_W_dist_pred_v_true'].Fill(np.float(W_lep_R))
        hists['lep_W_dist_pred_v_obs'].Fill(np.float(W_lep_R_po))
        hists['lep_W_transverse_mass_observed'].Fill(np.float(met_obs))
        hists['lep_W_transverse_energy_diff'].Fill(np.float(W_lep_Et_diff))
        # Hadronic W
        hists['hadronic_W_dist_true_v_obs'].Fill(np.float(W_had_dist_true))
        hists['hadronic_W_dist_pred_v_true'].Fill(np.float(W_had_R))
        hists['hadronic_W_dist_pred_v_obs'].Fill(np.float(W_had_R_po))
        hists['hadronic_W_true_pT_diff'].Fill(np.float(W_had_true_pT_diff))
        # Invariant mass:
        hists['hadronic_W_obs_mass'].Fill(closest_W_had.M())
        # Jet matching criteria scatterplots
        hists['hadronic_W_scat_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true) 
        hists['hadronic_W_scat_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff) 
        hists['hadronic_W_scat_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)  
        # Jet matching criteria correlation plots
        hists['hadronic_W_corr_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true) 
        hists['hadronic_W_corr_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff) 
        hists['hadronic_W_corr_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)  
        # Closest 1-jet Pt difference vs. closest 2-jet Pt difference
        hists['hadronic_W_scat_1_pT_diff_v_2_pT_diff'].Fill(W_had_true_pT_diff_1, W_had_true_pT_diff_2)
        hists['hadronic_W_corr_1_pT_diff_v_2_pT_diff'].Fill(W_had_true_pT_diff_1, W_had_true_pT_diff_2)
        # Plots that depend on whether a 1,2, or 3-jet sum is the best match to truth:
        if w_jets == 0:
            hists['hadronic_W_obs_1_mass'].Fill(closest_W_had.M())
            hists['hadronic_W_obs_1_mass_log'].Fill(closest_W_had.M())
            hists['hadronic_W_1_scat_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_1_scat_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_1_scat_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)
            hists['hadronic_W_1_corr_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_1_corr_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_1_corr_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)
        elif w_jets == 1:
            hists['hadronic_W_obs_2_mass'].Fill(closest_W_had.M())
            hists['hadronic_W_2_scat_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_2_scat_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_2_scat_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)
            hists['hadronic_W_2_corr_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_2_corr_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_2_corr_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)
        elif w_jets == 2:
            hists['hadronic_W_obs_3_mass'].Fill(closest_W_had.M())
            hists['hadronic_W_3_scat_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_3_scat_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_3_scat_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)
            hists['hadronic_W_3_corr_dist_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_dist_true)
            hists['hadronic_W_3_corr_pT_diff_true_v_obs_mass'].Fill(closest_W_had.M(), W_had_true_pT_diff)
            hists['hadronic_W_3_corr_pT_diff_dist_true_v_obs'].Fill(W_had_dist_true, W_had_true_pT_diff)


   # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))
    print('=================================================================')
    print('Number of events within limit: ')
    print('good_W_had', good_W_had, 'bad_W_had', bad_W_had)
    print('good_W_lep', good_W_lep, 'bad_W_lep', bad_W_lep)
    print('good_b_had', good_b_had, 'bad_b_had', bad_b_had)
    print('good_b_lep', good_b_lep, 'bad_b_lep', bad_b_lep)
    print('{}% good_W_had'.format(100*good_W_had/n_events))
    print('{}% good_W_lep'.format(100*good_W_lep/n_events))
    print('{}% good_b_had'.format(100*good_b_had/n_events))
    print('{}% good_b_lep'.format(100*good_b_lep/n_events))
    print('=================================================================')
    print('Jet matching percentages for hadronic W')
    print('{}% 1 jet Hadronic W, {} events'.format(100*w_had_jets[0]/n_events, w_had_jets[0]))
    print('{}% 2 jet Hadronic W, {} events'.format(100*w_had_jets[1]/n_events, w_had_jets[1]))
    print('{}% 3 jet Hadronic W, {} events'.format(100*w_had_jets[2]/n_events, w_had_jets[2]))
    print("Number of hadronic W's satisfying cut criteria")
    print('1 jet Hadronic Ws within cut')
    print('2 jet Hadronic Ws within cut')
    print('3 jet Hadronic Ws within cut')

# Helper function to output and save the histograms and scatterplots 
def plot_hists(key):
    c1 = TCanvas()
    hists[key].Draw()

    # Make log-log plots:
    if "log" in key:
        c1.SetLogy()

    legend = TLatex()
    legend.SetNDC()
    legend.SetTextFont(42)
    legend.SetTextColor(kBlack) 
    # Display bin width for all plots except scatterplots
    if "scat" not in key:
        binWidth = hists[key].GetBinWidth(0)
        legend.DrawLatex( 0.65, 0.70, "Bin Width: %.2f" % binWidth )
    
    c1.SaveAs(outputdir + subdir + key +'.png')
    c1.Close()


# Helper function to output and save the correlation plots
def plot_corr(key):
    hist = hists[key]

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
    title.Draw()

    c.cd()
    c.SaveAs(outputdir + subdir + key +'.png')
    pad0.Close()
    c.Close()
    

# Run the two helper functions above   
if __name__ == "__main__":
    try:
        os.mkdir('{}/{}'.format(outputdir, subdir))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()

    # gStyle.SetPalette(kGreyScale)
    # gROOT.GetColor(52).InvertPalette()

    # for key in hists:
    #     if 'corr' not in key:
    #         plot_hists(key)
    #     else:
    #         plot_corr(key)

    hists_key = []
    corr_key = []
    for key in hists:
        if 'corr' not in key:
            hists_key.append(key)
        else:
            corr_key.append(key)

    for key in hists_key:
        plot_hists(key)

    # The following few lines must be run only once for all correlation plots, 
    #  so the correlation plots must be separated out from the other histograms.   
    from AngryTops.features import *
    from AngryTops.Plotting.PlottingHelper import *
    gStyle.SetPalette(kGreyScale)
    gROOT.GetColor(52).InvertPalette()
    for key in corr_key:
        plot_corr(key)