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

subdir = '/closejets_img{}/'.format(date)
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = True
cut_mass = cut_pT = cut_dist = False
mass_cut = (25, 100)
pT_cut = (-50, 50)
dist_cut = (0, 0.5)

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
        if b_tagging:
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
jet_1 = jets_jets[:,0]
jet_2 = jets_jets[:,1]
jet_3 = jets_jets[:,2]
jet_4 = jets_jets[:,3]
jet_5 = jets_jets[:,4]

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

# define indixes
n_events = true.shape[0]
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

# Hadronic W
hists['had_W_true_pT_diff'] = TH1F("h_pT_W_had_true","W Hadronic p_T, True - Observed", 50, -300, 300)
hists['had_W_true_pT_diff'].SetTitle("W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_3_pT_diff'] = TH1F("h_pT_W_had_true","3 Jet W Hadronic p_{T} Diff, True - Observed", 30, -300, 300. )
hists['had_W_true_3_pT_diff'].SetTitle("3 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_2_pT_diff'] = TH1F("h_pT_W_had_true","2 Jet W Hadronic p_{T} Diff, True - Observed", 30, -300, 300. )
hists['had_W_true_2_pT_diff'].SetTitle("2 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
hists['had_W_true_1_pT_diff'] = TH1F("h_pT_W_had_true","1 Jet W Hadronic p_{T} Diff, True - Observed", 30, -300, 300. )
hists['had_W_true_1_pT_diff'].SetTitle("1 Jet W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
# True vs. obs
hists['had_W_dist_true_v_obs'] = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist_true_v_obs'].SetTitle("W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
hists['had_W_dist_3_true_v_obs'] = TH1F("h_W_had_true","3 Jet W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist_3_true_v_obs'].SetTitle("3 Jet W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
hists['had_W_dist_2_true_v_obs'] = TH1F("h_W_had_true","2 Jet W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist_2_true_v_obs'].SetTitle("2 Jet W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
hists['had_W_dist_1_true_v_obs'] = TH1F("h_W_had_true","1 Jet W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['had_W_dist_1_true_v_obs'].SetTitle("1 Jet W Hadronic #eta-#phi distances, True vs Observed;Hadronic (radians);A.U.")
# Pred vs. true
hists['had_W_dist_pred_v_true'] = TH1F("h_W_had_pred","W Hadronic Distances, Predicted vs Truth", 50, 0, 3)
hists['had_W_dist_pred_v_true'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth; Hadronic (radians);A.U.")
# Pred vs. obs
hists['had_W_dist_pred_v_obs'] = TH1F("h_W_had_d","W Hadronic Distances, Predicted vs Observed", 50, 0, 3)
hists['had_W_dist_pred_v_obs'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Observed; Hadronic (radians);A.U.")
# Jet matching invariant mass distributions
hists['had_W_obs_1_mass'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, Observed", 30, 0., 300. )
hists['had_W_obs_1_mass'].SetTitle("1 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_2_mass'] = TH1F("W_had_m","2 Jet W Hadronic Invariant Mass, Observed", 30, 0., 300. )
hists['had_W_obs_2_mass'].SetTitle("2 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_3_mass'] = TH1F("W_had_m","3 Jet W Hadronic Invariant Mass, Observed", 30, 0., 300. )
hists['had_W_obs_3_mass'].SetTitle("3 Jet W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
hists['had_W_obs_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, Observed", 30, 0., 300. )
hists['had_W_obs_mass'].SetTitle("W Hadronic Invariant Mass, Observed; Hadronic (GeV); A.U.")
# Jet matching criteria correlation plots
# invariant mass vs eta-phi dist
hists['had_W_corr_1_mass_dist'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0, 120 , 50, 0, 3.2  )
hists['had_W_corr_2_mass_dist'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic #eta-#phi Distances [rad]", 50, 10, 300 , 50, 0, 3.2  )
hists['had_W_corr_3_mass_dist'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic #eta-#phi Distances [rad]", 50, 40, 350 , 50, 0, 3.2  )
hists['had_W_corr_mass_dist'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic #eta-#phi Distances [rad]", 50, 0, 350 , 50, 0, 3.2  )
# invariant mass vs Pt difference
hists['had_W_corr_1_mass_Pt'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 80 , 50, -200, 200  )
hists['had_W_corr_2_mass_Pt'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 10, 300 , 50, -200, 200  )
hists['had_W_corr_3_mass_Pt'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic p_{T} diff [GeV]", 50, 40, 350 , 50, -200, 200  )
hists['had_W_corr_mass_Pt'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic p_{T} diff [GeV]", 50, 0, 350 , 50, -200, 200  )
# eta-phi dist vs. Pt difference
hists['had_W_corr_1_dist_Pt'] = TH2F( "W_had_corr_d",   ";1 Jet W Hadronic #eta-#phi Distances [rad];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_2_dist_Pt'] = TH2F( "W_had_corr_d",   ";2 Jet W Hadronic #eta-#phi Distances [rad];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_3_dist_Pt'] = TH2F( "W_had_corr_d",   ";3 Jet W Hadronic #eta-#phi Distances [rad];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
hists['had_W_corr_dist_Pt'] = TH2F( "W_had_corr_d",   ";W Hadronic #eta-#phi Distances [rad];W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )

if cut_mass:
    ################################# mass cut-offs ################################
    hists['had_W_mass_cut_dist'] = TH1F("W_had_m","W Hadronic Distances, Mass Cutoff", 50, 0., 3. )
    hists['had_W_mass_cut_dist'].SetTitle("W Hadronic #eta-#phi Distances, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))
    hists['had_W_mass_cut_pT'] = TH1F("W_had_m","W Hadronic p_T Diff, Mass Cutoff", 50, -300., 300. )
    hists['had_W_mass_cut_pT'].SetTitle("W Hadronic p_T Diff, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))

    hists['had_W_mass_cut_1_dist'] = TH1F("W_had_m","1 Jet W Hadronic Distances, Mass Cutoff", 50, 0., 3. )
    hists['had_W_mass_cut_1_dist'].SetTitle("1 Jet W Hadronic #eta-#phi Distances, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))
    hists['had_W_mass_cut_1_pT'] = TH1F("W_had_m","1 Jet W Hadronic p_T Diff, Mass Cutoff", 50, -300., 300. )
    hists['had_W_mass_cut_1_pT'].SetTitle("1 Jet W Hadronic p_T Diff, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))

    hists['had_W_mass_cut_2_dist'] = TH1F("W_had_m","2 Jet W Hadronic Distances, Mass Cutoff", 50, 0., 3. )
    hists['had_W_mass_cut_2_dist'].SetTitle("2 Jet W Hadronic #eta-#phi Distances, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))
    hists['had_W_mass_cut_2_pT'] = TH1F("W_had_m","2 Jet W Hadronic p_T Diff, Mass Cutoff", 50, -300., 300. )
    hists['had_W_mass_cut_2_pT'].SetTitle("2 Jet W Hadronic p_T Diff, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))

    hists['had_W_mass_cut_3_dist'] = TH1F("W_had_m","3 Jet W Hadronic Distances, Mass Cutoff", 50, 0., 3. )
    hists['had_W_mass_cut_3_dist'].SetTitle("3 Jet W Hadronic #eta-#phi Distances, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))
    hists['had_W_mass_cut_3_pT'] = TH1F("W_had_m","3 Jet W Hadronic p_T Diff, Mass Cutoff", 50, -300., 300. )
    hists['had_W_mass_cut_3_pT'].SetTitle("3 Jet W Hadronic p_T Diff, Mass Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(mass_cut[0], mass_cut[1]))

    # correlations
    # invariant mass vs eta-phi dist
    hists['had_W_mass_cut_corr_1_mass_dist'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic #eta-#phi Distances [rad]", 50, mass_cut[0], mass_cut[1] , 50, 0, 3.2  )
    hists['had_W_mass_cut_corr_2_mass_dist'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic #eta-#phi Distances [rad]", 50, mass_cut[0], mass_cut[1] , 50, 0, 3.2  )
    hists['had_W_mass_cut_corr_3_mass_dist'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic #eta-#phi Distances [rad]", 50, mass_cut[0], mass_cut[1] , 50, 0, 3.2  )
    hists['had_W_mass_cut_corr_mass_dist'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic #eta-#phi Distances [rad]", 50, mass_cut[0], mass_cut[1] , 50, 0, 3.2  )
    # invariant mass vs Pt difference
    hists['had_W_mass_cut_corr_1_mass_Pt'] = TH2F( "W_had_corr_m",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic p_{T} Diff [GeV]", 50, mass_cut[0], mass_cut[1] , 50, -200, 200  )
    hists['had_W_mass_cut_corr_2_mass_Pt'] = TH2F( "W_had_corr_m",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic p_{T} Diff [GeV]", 50, mass_cut[0], mass_cut[1] , 50, -200, 200  )
    hists['had_W_mass_cut_corr_3_mass_Pt'] = TH2F( "W_had_corr_m",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic p_{T} diff [GeV]", 50, mass_cut[0], mass_cut[1] , 50, -200, 200  )
    hists['had_W_mass_cut_corr_mass_Pt'] = TH2F( "W_had_corr_m",   ";W Hadronic Invariant Mass [GeV];W Hadronic p_{T} diff [GeV]", 50, mass_cut[0], mass_cut[1] , 50, -200, 200  )
    # eta-phi dist vs. Pt difference
    hists['had_W_mass_cut_corr_1_dist_Pt'] = TH2F( "W_had_corr_d",   ";1 Jet W Hadronic #eta-#phi Distances [rad];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
    hists['had_W_mass_cut_corr_2_dist_Pt'] = TH2F( "W_had_corr_d",   ";2 Jet W Hadronic #eta-#phi Distances [rad];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
    hists['had_W_mass_cut_corr_3_dist_Pt'] = TH2F( "W_had_corr_d",   ";3 Jet W Hadronic #eta-#phi Distances [rad];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
    hists['had_W_mass_cut_corr_dist_Pt'] = TH2F( "W_had_corr_d",   ";W Hadronic #eta-#phi Distances [rad];W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, -200, 200  )
if cut_pT:
    ################################ pT cut-offs #################################
    hists['had_W_pT_cut_dist'] = TH1F("W_had_m","W Hadronic Distances, pT Cutoff", 50, 0., 3. )
    hists['had_W_pT_cut_dist'].SetTitle("W Hadronic #eta-#phi Distances, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))
    hists['had_W_pT_cut_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, pT Cutoff", 50, 0., 300. )
    hists['had_W_pT_cut_mass'].SetTitle("W Hadronic Invariant Mass, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))

    hists['had_W_pT_cut_1_dist'] = TH1F("W_had_m","1 Jet W Hadronic Distances, pT Cutoff", 50, 0., 3. )
    hists['had_W_pT_cut_1_dist'].SetTitle("1 Jet W Hadronic #eta-#phi Distances, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))
    hists['had_W_pT_cut_1_mass'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, pT Cutoff", 50, 0., 300. )
    hists['had_W_pT_cut_1_mass'].SetTitle("1 Jet W Hadronic Invariant Mass, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))

    hists['had_W_pT_cut_2_dist'] = TH1F("W_had_m","2 Jet W Hadronic Distances, pT Cutoff", 50, 0., 3. )
    hists['had_W_pT_cut_2_dist'].SetTitle("2 Jet W Hadronic #eta-#phi Distances, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))
    hists['had_W_pT_cut_2_mass'] = TH1F("W_had_m","2 Jet W Hadronic Invariant Mass, pT Cutoff", 50, 0., 300. )
    hists['had_W_pT_cut_2_mass'].SetTitle("2 Jet W Hadronic Invariant Mass, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))

    hists['had_W_pT_cut_3_dist'] = TH1F("W_had_m","3 Jet W Hadronic Distances, pT Cutoff", 50, 0., 3. )
    hists['had_W_pT_cut_3_dist'].SetTitle("3 Jet W Hadronic #eta-#phi Distances, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))
    hists['had_W_pT_cut_3_mass'] = TH1F("W_had_m","3 Jet W Hadronic Invariant Mass, pT Cutoff", 50, 0., 300. )
    hists['had_W_pT_cut_3_mass'].SetTitle("3 Jet W Hadronic Invariant Mass, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(pT_cut[0], pT_cut[1]))

    # correlations
    # invariant mass vs eta-phi dist
    hists['had_W_pT_cut_corr_1_mass_dist'] = TH2F( "W_had_corr_pt",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, 0., 3. )
    hists['had_W_pT_cut_corr_2_mass_dist'] = TH2F( "W_had_corr_pt",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, 0., 3. )
    hists['had_W_pT_cut_corr_3_mass_dist'] = TH2F( "W_had_corr_pt",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, 0., 3. )
    hists['had_W_pT_cut_corr_mass_dist'] = TH2F( "W_had_corr_pt",   ";W Hadronic Invariant Mass [GeV];W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, 0., 3. )
    # invariant mass vs Pt difference
    hists['had_W_pT_cut_corr_1_mass_Pt'] = TH2F( "W_had_corr_pt",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0., 300., 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_2_mass_Pt'] = TH2F( "W_had_corr_pt",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0., 300., 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_3_mass_Pt'] = TH2F( "W_had_corr_pt",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0., 300., 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_mass_Pt'] = TH2F( "W_had_corr_pt",   ";W Hadronic Invariant Mass [GeV];W Hadronic p_{T} diff [GeV]", 50, 0., 300., 50, pT_cut[0], pT_cut[1] )
    # eta-phi dist vs. Pt difference
    hists['had_W_pT_cut_corr_1_dist_Pt'] = TH2F( "W_had_corr_pt",   ";1 Jet W Hadronic #eta-#phi Distances [rad];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_2_dist_Pt'] = TH2F( "W_had_corr_pt",   ";2 Jet W Hadronic #eta-#phi Distances [rad];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0, 3.2 , 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_3_dist_Pt'] = TH2F( "W_had_corr_pt",   ";3 Jet W Hadronic #eta-#phi Distances [rad];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, pT_cut[0], pT_cut[1] )
    hists['had_W_pT_cut_corr_dist_Pt'] = TH2F( "W_had_corr_pt",   ";W Hadronic #eta-#phi Distances [rad];W Hadronic p_{T} diff [GeV]", 50, 0, 3.2 , 50, pT_cut[0], pT_cut[1] )
if cut_dist:
    ################################ dist cut-offs #################################
    hists['had_W_dist_cut_pT'] = TH1F("W_had_m","W Hadronic pT Diff, pT Cutoff", 50, -200., 200. )
    hists['had_W_dist_cut_pT'].SetTitle("W Hadronic pT Diff, pT Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))
    hists['had_W_dist_cut_mass'] = TH1F("W_had_m","W Hadronic Invariant Mass, dist Cutoff", 50, 0., 300. )
    hists['had_W_dist_cut_mass'].SetTitle("W Hadronic Invariant Mass, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))

    hists['had_W_dist_cut_1_pT'] = TH1F("W_had_m","1 Jet W Hadronic pT Diff, dist Cutoff", 50, -200., 200. )
    hists['had_W_dist_cut_1_pT'].SetTitle("1 Jet W Hadronic pT Diff, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))
    hists['had_W_dist_cut_1_mass'] = TH1F("W_had_m","1 Jet W Hadronic Invariant Mass, dist Cutoff", 50, 0., 300. )
    hists['had_W_dist_cut_1_mass'].SetTitle("1 Jet W Hadronic Invariant Mass, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))

    hists['had_W_dist_cut_2_pT'] = TH1F("W_had_m","2 Jet W Hadronic pT Diff, dist Cutoff", 50, -200., 200. )
    hists['had_W_dist_cut_2_pT'].SetTitle("2 Jet W Hadronic pT Diff, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))
    hists['had_W_dist_cut_2_mass'] = TH1F("W_had_m","2 Jet W Hadronic Invariant Mass, dist Cutoff", 50, 0., 300. )
    hists['had_W_dist_cut_2_mass'].SetTitle("2 Jet W Hadronic Invariant Mass, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))

    hists['had_W_dist_cut_3_pT'] = TH1F("W_had_m","3 Jet W Hadronic pT Diff, dist Cutoff", 50, -200., 200. )
    hists['had_W_dist_cut_3_pT'].SetTitle("3 Jet W Hadronic pT Diff, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))
    hists['had_W_dist_cut_3_mass'] = TH1F("W_had_m","3 Jet W Hadronic Invariant Mass, dist Cutoff", 50, 0., 300. )
    hists['had_W_dist_cut_3_mass'].SetTitle("3 Jet W Hadronic Invariant Mass, dist Cutoff {} - {} GeV; Hadronic (radians); A.U.".format(dist_cut[0], dist_cut[1]))

    # correlations
    # invariant mass vs eta-phi dist
    hists['had_W_dist_cut_corr_1_mass_dist'] = TH2F( "W_had_corr_dist",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, dist_cut[0], dist_cut[1] )
    hists['had_W_dist_cut_corr_2_mass_dist'] = TH2F( "W_had_corr_dist",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, dist_cut[0], dist_cut[1] )
    hists['had_W_dist_cut_corr_3_mass_dist'] = TH2F( "W_had_corr_dist",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, dist_cut[0], dist_cut[1] )
    hists['had_W_dist_cut_corr_mass_dist'] = TH2F( "W_had_corr_dist",   ";W Hadronic Invariant Mass [GeV];W Hadronic #eta-#phi Distances [rad]", 50, 0., 300., 50, dist_cut[0], dist_cut[1] )
    # invariant mass vs Pt difference
    hists['had_W_dist_cut_corr_1_mass_Pt'] = TH2F( "W_had_corr_dist",   ";1 Jet W Hadronic Invariant Mass [GeV];1 Jet W Hadronic p_{T} Diff [GeV]", 50, 0., 300., 50,  -200., 200. )
    hists['had_W_dist_cut_corr_2_mass_Pt'] = TH2F( "W_had_corr_dist",   ";2 Jet W Hadronic Invariant Mass [GeV];2 Jet W Hadronic p_{T} Diff [GeV]", 50, 0., 300., 50,  -200., 200. )
    hists['had_W_dist_cut_corr_3_mass_Pt'] = TH2F( "W_had_corr_dist",   ";3 Jet W Hadronic Invariant Mass [GeV];3 Jet W Hadronic p_{T} diff [GeV]", 50, 0., 300., 50,  -200., 200. )
    hists['had_W_dist_cut_corr_mass_Pt'] = TH2F( "W_had_corr_dist",   ";W Hadronic Invariant Mass [GeV];W Hadronic p_{T} diff [GeV]", 50, 0., 300., 50,  -200., 200. )
    # eta-phi dist vs. Pt difference
    hists['had_W_dist_cut_corr_1_dist_Pt'] = TH2F( "W_had_corr_dist",   ";1 Jet W Hadronic #eta-#phi Distances [rad];1 Jet W Hadronic p_{T} Diff [GeV]", 50, dist_cut[0], dist_cut[1] , 50,  -200., 200. )
    hists['had_W_dist_cut_corr_2_dist_Pt'] = TH2F( "W_had_corr_dist",   ";2 Jet W Hadronic #eta-#phi Distances [rad];2 Jet W Hadronic p_{T} Diff [GeV]", 50, dist_cut[0], dist_cut[1] , 50,  -200., 200. )
    hists['had_W_dist_cut_corr_3_dist_Pt'] = TH2F( "W_had_corr_dist",   ";3 Jet W Hadronic #eta-#phi Distances [rad];3 Jet W Hadronic p_{T} diff [GeV]", 50, dist_cut[0], dist_cut[1] , 50,  -200., 200. )
    hists['had_W_dist_cut_corr_dist_Pt'] = TH2F( "W_had_corr_dist",   ";W Hadronic #eta-#phi Distances [rad];W Hadronic p_{T} diff [GeV]", 50, dist_cut[0], dist_cut[1], 50,  -200., 200. )

def make_histograms():
    # define tolerance limits
    W_had_dist_t_lim = 1.28

    full_recon_dist_true = part_recon_dist_true = un_recon_dist_true = 0.
    full_recon_phi_true = part_recon_phi_true = un_recon_phi_true = 0.
    full_recon_eta_true = part_recon_eta_true = un_recon_eta_true = 0.

    p_full_recon_t_full_dist = p_part_recon_t_full_dist = p_un_recon_t_full_dist = 0.
    p_full_recon_t_part_dist = p_part_recon_t_part_dist = p_un_recon_t_part_dist = 0.
    p_full_recon_t_un_dist = p_part_recon_t_un_dist = p_un_recon_t_un_dist = 0.

    W_had_dist_p_corr_t_full = 0.
    W_had_dist_p_corr_t_part = 0.
    good_W_had = 0.
    bad_W_had = 0.

    high_E = 0.
    w_had_jets = [0., 0., 0.]

    for i in event_index: # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W )
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W)

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
        if np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)

        ################################################# true vs predicted #################################################
        W_had_R = find_dist(W_had_true, W_had_fitted)
        # determine whether or not the jets were reconstructed
        W_had_phi_recon = W_had_eta_recon = W_had_R_recon = False

        if ((W_had_true.Phi() - 57.0/50.0) <= W_had_fitted.Phi()) and (W_had_fitted.Phi() <= (W_had_true.Phi() + 57.0/50.0)):
            W_had_phi_recon = True
        if ((W_had_true.Eta() - 4.0/5.0) <= W_had_fitted.Eta()) and (W_had_fitted.Eta() <= (W_had_true.Eta() + 4.0/5.0)):
            if (np.abs(W_had_true.Eta()) <= 1.8) and (np.abs(W_had_fitted.Eta()) <= 1.8):
                W_had_eta_recon = True
        if (W_had_phi_recon == True) and (W_had_eta_recon == True):
            W_had_R_recon = True
        elif W_had_R <= (W_had_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                W_had_R_recon = True
        
        ################################################# true vs observed ################################################# 
        W_had_dist_true = 10000000

        # loop through each of the jets to find the minimum distance for each particle
        for k in range(len(jets)): 
            b_had_d_true = find_dist(b_had_true, jets[k])
            b_lep_d_true = find_dist(b_lep_true, jets[k])
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
                closest_b_had = jets[k]
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
                closest_b_lep = jets[k]

            # For hadronic Ws
            # check 1, 2, 3 jet combinations for hadronic W

            # Single jets
            if (not b_tagging) or (not jet_list[k,i,5]): 
                # Check if jet k is not b-tagged, k ranges from 0 to 4 or 5 depending on event type
                sum_vect = jets[k]    
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
                    closest_W_had = sum_vect
                    w_jets = 0
            # Two jets
                for j in range(k + 1, len(jets)):
                    # Check if jet j is not b-tagged, j ranges from k+1 to 4 or 5 depending on event type     
                    if (not b_tagging) or (not jet_list[j,i,5]):
                        sum_vect = jets[k] + jets[j] 
                        W_had_d_true = find_dist(W_had_true, sum_vect)
                        if W_had_d_true < W_had_dist_true:
                            W_had_dist_true = W_had_d_true
                            W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
                            closest_W_had = sum_vect
                            w_jets = 1
                # Threejets
                        for l in range(j+1, len(jets)):
                            # Check if jet l is not b-tagged, l ranges from j+k+1 to 4 or 5 depending on event type
                            if (not b_tagging) or  (not jet_list[l,i,5]):
                                sum_vect = jets[k] + jets[j] + jets[l]
                                W_had_d_true = find_dist(W_had_true, sum_vect)
                                if W_had_d_true < W_had_dist_true:
                                    W_had_dist_true = W_had_d_true
                                    W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
                                    closest_W_had = sum_vect
                                    w_jets = 2

        w_had_jets[w_jets] += 1    

        if (W_had_dist_true <= W_had_dist_t_lim):
            good_W_had += 1
        else:
            bad_W_had += 1

        ################################################# predicted vs observed #################################################

        # Hadronic W
        W_had_R_po = find_dist( W_had_fitted, closest_W_had )

        ################################################# populate histograms and check percentages #################################################

        # Hadronic W
        hists['had_W_dist_true_v_obs'].Fill(np.float(W_had_dist_true))
        hists['had_W_dist_pred_v_true'].Fill(np.float(W_had_R))
        hists['had_W_dist_pred_v_obs'].Fill(np.float(W_had_R_po))
        # pT diff and Invariant mass:
        hists['had_W_true_pT_diff'].Fill(np.float(W_had_true_pT))
        hists['had_W_obs_mass'].Fill(np.float(closest_W_had.M()))
        # Jet matching criteria correlation plots
        hists['had_W_corr_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
        hists['had_W_corr_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
        hists['had_W_corr_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        # Mass Cutoff
        if cut_mass and closest_W_had.M() <= mass_cut[1] and closest_W_had.M() >= mass_cut[0]:
            hists['had_W_mass_cut_dist'].Fill(np.float(W_had_dist_true))
            hists['had_W_mass_cut_pT'].Fill(np.float(W_had_true_pT))
            hists['had_W_mass_cut_corr_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_mass_cut_corr_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_mass_cut_corr_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        # pT Cutoff
        if cut_pT and W_had_true_pT <= pT_cut[1] and W_had_true_pT >= pT_cut[0]:
            hists['had_W_pT_cut_dist'].Fill(np.float(W_had_dist_true))
            hists['had_W_pT_cut_mass'].Fill(np.float(closest_W_had.M()))
            hists['had_W_pT_cut_corr_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_pT_cut_corr_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_pT_cut_corr_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        if cut_dist and W_had_dist_true <= dist_cut[1] and W_had_dist_true >= dist_cut[0]:
            hists['had_W_dist_cut_pT'].Fill(np.float(W_had_true_pT))
            hists['had_W_dist_cut_mass'].Fill(np.float(closest_W_had.M()))
            hists['had_W_dist_cut_corr_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_dist_cut_corr_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_dist_cut_corr_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        # Plots that depend on whether a 1,2, or 3-jet sum is the best match to truth:
        if w_jets == 0:
            hists['had_W_obs_1_mass'].Fill(closest_W_had.M())
            hists['had_W_dist_1_true_v_obs'].Fill(np.float(W_had_dist_true))
            hists['had_W_corr_1_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_corr_1_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_corr_1_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            hists['had_W_true_1_pT_diff'].Fill(np.float(W_had_true_pT))
            if cut_mass and (closest_W_had.M() <= mass_cut[1] and closest_W_had.M() >= mass_cut[0]):
                hists['had_W_mass_cut_1_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_mass_cut_1_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_mass_cut_corr_1_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_mass_cut_corr_1_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_mass_cut_corr_1_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_pT and (W_had_true_pT <= pT_cut[1] and W_had_true_pT >= pT_cut[0]):
                hists['had_W_pT_cut_1_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_pT_cut_1_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_pT_cut_corr_1_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_pT_cut_corr_1_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_pT_cut_corr_1_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_dist and (W_had_dist_true <= dist_cut[1] and W_had_dist_true >= dist_cut[0]):
                hists['had_W_dist_cut_1_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_dist_cut_1_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_dist_cut_corr_1_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_dist_cut_corr_1_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_dist_cut_corr_1_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        if w_jets == 1:
            hists['had_W_obs_2_mass'].Fill(closest_W_had.M())
            hists['had_W_dist_2_true_v_obs'].Fill(np.float(W_had_dist_true))
            hists['had_W_corr_2_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_corr_2_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_corr_2_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            hists['had_W_true_2_pT_diff'].Fill(np.float(W_had_true_pT))
            if cut_mass and (closest_W_had.M() <= mass_cut[1] and closest_W_had.M() >= mass_cut[0]):
                hists['had_W_mass_cut_2_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_mass_cut_2_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_mass_cut_corr_2_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_mass_cut_corr_2_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_mass_cut_corr_2_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_pT and (W_had_true_pT <= pT_cut[1] and W_had_true_pT >= pT_cut[0]):
                hists['had_W_pT_cut_2_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_pT_cut_2_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_pT_cut_corr_2_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_pT_cut_corr_2_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_pT_cut_corr_2_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_dist and (W_had_dist_true <= dist_cut[1] and W_had_dist_true >= dist_cut[0]):
                hists['had_W_dist_cut_2_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_dist_cut_2_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_dist_cut_corr_2_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_dist_cut_corr_2_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_dist_cut_corr_2_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
        if w_jets == 2:
            hists['had_W_obs_3_mass'].Fill(closest_W_had.M())
            hists['had_W_dist_3_true_v_obs'].Fill(np.float(W_had_dist_true))
            hists['had_W_corr_3_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_corr_3_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
            hists['had_W_corr_3_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            hists['had_W_true_3_pT_diff'].Fill(np.float(W_had_true_pT))
            if cut_mass and (closest_W_had.M() <= mass_cut[1] and closest_W_had.M() >= mass_cut[0]):
                hists['had_W_mass_cut_3_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_mass_cut_3_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_mass_cut_corr_3_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_mass_cut_corr_3_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_mass_cut_corr_3_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_pT and (W_had_true_pT <= pT_cut[1] and W_had_true_pT >= pT_cut[0]):
                hists['had_W_pT_cut_3_dist'].Fill(np.float(W_had_dist_true))
                hists['had_W_pT_cut_3_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_pT_cut_corr_3_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_pT_cut_corr_3_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_pT_cut_corr_3_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)
            if cut_dist and (W_had_dist_true <= dist_cut[1] and W_had_dist_true >= dist_cut[0]):
                hists['had_W_dist_cut_3_pT'].Fill(np.float(W_had_true_pT))
                hists['had_W_dist_cut_3_mass'].Fill(np.float(closest_W_had.M()))
                hists['had_W_dist_cut_corr_3_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_dist_cut_corr_3_mass_Pt'].Fill(closest_W_had.M(), W_had_true_pT)
                hists['had_W_dist_cut_corr_3_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_true_pT)

   # Print data regarding percentage of each class of event
    print('good_W_had', good_W_had, 'bad_W_had', bad_W_had)

    print('=================================================================')
    print(100*good_W_had/n_events, ' good_W_had')
    print('=================================================================')
    print(100*w_had_jets[0]/n_events, '1 event Hadronic W')
    print(100*w_had_jets[1]/n_events, '2 event Hadronic W')
    print(100*w_had_jets[2]/n_events, '3 event Hadronic W')

# Helper function to output and save the plots 
def plot_hists(key):
    c1 = TCanvas()
    hists[key].Draw()

    # Display bin width
    binWidth = hists[key].GetBinWidth(0)
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.65, 0.70, "Bin Width: %.2f" % binWidth )

    c1.SaveAs(outputdir + subdir + key +'.png')
    c1.Close()

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
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )

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
    
    hists_key = []
    corr_key = []
    for key in hists:
        if 'corr' not in key:
            hists_key.append(key)
        else:
            corr_key.append(key)

    for key in hists_key:
        plot_hists(key)

    # The following two lines must be run only once for all correlation plots, 
    #  so the correlation plots must be separated out from the other histograms.    
    
    from AngryTops.features import *
    from AngryTops.Plotting.PlottingHelper import *
    gStyle.SetPalette(kGreyScale)
    gROOT.GetColor(52).InvertPalette()
    for key in corr_key:
        plot_corr(key)