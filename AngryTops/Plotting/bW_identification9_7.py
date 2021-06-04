<<<<<<< HEAD
import os, sys
import numpy as np
from scipy.spatial import distance
from ROOT import *
#import ROOT
import pickle

representation = sys.argv[2]
outputdir = sys.argv[1]
subdir = '/closejets_img/'
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95

def MakeLimit(tvo):
	pvt = ((np.pi)/8) * (np.sin(2*tvo - ((np.pi)/2)) + 1)
	pvt = float(round(pvt, 2))
	return pvt

def MakeP4_lep(y,m):
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[3]
    E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
    p4.SetPxPyPzE(p0, p1, p2, E)
    return p4

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
      jets_jets = jets[:,6:]
      jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
      jets_jets = np.delete(jets_jets, 5, 2)
      jets_jets = jets_jets.reshape((jets_jets.shape[0], 25))
      jets_lep = lep_scalar.inverse_transform(jets_lep)
      jets_jets = jets_scalar.inverse_transform(jets_jets)
      jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))#I think this is the final 6x6 array the arxiv paper was talking about

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

n_events = true.shape[0]
w = 1

def make_histograms():
    W_lep_met_d = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Observed", 50, 0, 120)
    W_lep_met_d.SetTitle("W Leptonic Transverse Mass, Observed;Leptonic (GeV);A.U.")
    W_lep_met_d_t = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Truth", 50, 0, 120)
    W_lep_met_d_t.SetTitle("W Leptonic Transverse Mass, Truth;Leptonic (GeV);A.U.")
    W_lep_met_d_p = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Predicted", 50, 0, 120)
    W_lep_met_d_p.SetTitle("W Leptonic Transverse Mass, Predicted;Leptonic (GeV);A.U.")

    W_lep_d = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    W_had_d = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Reconstructed", 50, 0, 5)
    W_lep_d.SetTitle("W Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (radians);A.U.")
    W_had_d.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (radians);A.U.")

    b_lep_d = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    b_had_d = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    b_lep_d.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
    b_had_d.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")

    t_lep_d = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    t_had_d = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    t_lep_d.SetTitle("t Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
    t_had_d.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")

    W_lep_d_part = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    W_had_d_part = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    W_lep_d_part.SetTitle("W Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (radians);A.U.")
    W_had_d_part.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (radians);A.U.")

    b_lep_d_part = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    b_had_d_part = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    b_lep_d_part.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
    b_had_d_part.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")

    t_lep_d_part = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    t_had_d_part = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    t_lep_d_part.SetTitle("t Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
    t_had_d_part.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")

    W_lep_d_un = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    W_had_d_un = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Not Reconstructed", 50, 0, 5)
    W_lep_d_un.SetTitle("W Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (radians);A.U.")
    W_had_d_un.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (radians);A.U.")

    b_lep_d_un = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    b_had_d_un = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    b_lep_d_un.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")
    b_had_d_un.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

    t_lep_d_un = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    t_had_d_un = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    t_lep_d_un.SetTitle("t Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")
    t_had_d_un.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

    h_W_had_true = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 5)
    h_pT_W_had_true = TH1F("h_pT_W_had_true","W Hadronic p_T, True vs Observed", 50, -500, 500)
    h_W_had_true.SetTitle("W Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    h_W_lep_true = TH1F("h_W_lep_true","W Leptonic Distances, True vs Observed", 50, 0, 5)
    h_W_lep_true.SetTitle("W Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")

    h_b_lep_true = TH1F("h_b_lep_true","b Leptonic Distances, True vs Observed", 50, 0, 5)
    h_b_had_true = TH1F("h_b_had_true","b Hadronic Distances, True vs Observed", 50, 0, 5)
    h_b_lep_true.SetTitle("b Leptonic #eta-#phi distances, True vs Observed;true leptonic (radians);A.U.")
    h_b_had_true.SetTitle("b Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    h_t_lep_true = TH1F("h_t_lep_true","t Leptonic Distances, True vs Observed", 50, 0, 5)
    h_t_had_true = TH1F("h_t_had_true","t Hadronic Distances, True vs Observed", 50, 0, 5)
    h_t_lep_true.SetTitle("t Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")
    h_t_had_true.SetTitle("t Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    jets = []

    b_lep_dist_t_lim = 0.39
    b_had_dist_t_lim = 0.39
    t_lep_dist_t_lim = 0.80
    t_had_dist_t_lim = 0.80
    W_lep_dist_t_lim = 0.82
    W_had_dist_t_lim = 1.28

    full_recon_dist_true = 0.
    part_recon_dist_true = 0.
    un_recon_dist_true = 0.
    full_recon_phi_true = 0.
    part_recon_phi_true = 0.
    un_recon_phi_true = 0.
    full_recon_eta_true = 0.
    part_recon_eta_true = 0.
    un_recon_eta_true = 0.

    p_full_recon_t_full_dist = 0.
    p_part_recon_t_full_dist = 0.
    p_un_recon_t_full_dist = 0.
    p_full_recon_t_part_dist = 0.
    p_part_recon_t_part_dist = 0.
    p_un_recon_t_part_dist = 0.
    p_full_recon_t_un_dist = 0.
    p_part_recon_t_un_dist = 0.
    p_un_recon_t_un_dist = 0.

    b_lep_dist_p_corr_t_full = 0.
    b_had_dist_p_corr_t_full = 0.
    t_lep_dist_p_corr_t_full = 0.
    t_had_dist_p_corr_t_full = 0.
    W_lep_dist_p_corr_t_full = 0.
    W_had_dist_p_corr_t_full = 0.

    b_lep_dist_p_corr_t_part = 0.
    b_had_dist_p_corr_t_part = 0.
    t_lep_dist_p_corr_t_part = 0.
    t_had_dist_p_corr_t_part = 0.
    W_lep_dist_p_corr_t_part = 0.
    W_had_dist_p_corr_t_part = 0.

    good_b_had = 0.
    good_b_lep = 0.
    good_W_had = 0.
    good_W_lep = 0.

    bad_b_had = 0.
    bad_b_lep = 0.
    bad_W_had = 0.
    bad_W_lep = 0.

    jump = 0

    h_b_lep_arr = []
    h_b_had_arr = []
    h_t_lep_arr = []
    h_t_had_arr = []
    h_W_lep_arr = []
    h_W_had_arr = []
    h_pT_W_had_arr = []
    h_W_lep_met_arr = []
    h_W_lep_met_arr_t = []
    h_W_lep_met_arr_p = []

    for i in range(n_events):
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
        
        jets.append([])
        jets[i].append(jet_1_vect)
        jets[i].append(jet_2_vect)
        jets[i].append(jet_3_vect)
        jets[i].append(jet_4_vect)
        jets[i].append(jet_5_vect)

        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(jet_mu[i][5])))
        met_true = np.sqrt(2*W_lep_true.Pt()*W_lep_true.Et()*(1 - np.cos(W_lep_true.Phi())))
        met_pred = np.sqrt(2*W_lep_fitted.Pt()*W_lep_fitted.Et()*(1 - np.cos(W_lep_fitted.Phi())))

        b_lep_dphi = min(np.abs(b_lep_true.Phi()-b_lep_fitted.Phi()), 2*np.pi-np.abs(b_lep_true.Phi()-b_lep_fitted.Phi()))
        b_lep_deta = b_lep_true.Eta()-b_lep_fitted.Eta()
        b_lep_R = np.sqrt(b_lep_dphi**2 + b_lep_deta**2)

        b_had_dphi = min(np.abs(b_had_true.Phi()-b_had_fitted.Phi()), 2*np.pi-np.abs(b_had_true.Phi()-b_had_fitted.Phi()))
        b_had_deta = b_had_true.Eta()-b_had_fitted.Eta()
        b_had_R = np.sqrt(b_had_dphi**2 + b_had_deta**2)

        t_lep_dphi = min(np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()), 2*np.pi-np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()))
        t_lep_R = np.sqrt(t_lep_dphi**2)

        t_had_dphi = min(np.abs(t_had_true.Phi()-t_had_fitted.Phi()), 2*np.pi-np.abs(t_had_true.Phi()-t_had_fitted.Phi()))
        t_had_deta = t_had_true.Eta()-t_had_fitted.Eta()
        t_had_R = np.sqrt(t_had_dphi**2 + t_had_deta**2)

        W_lep_dphi = min(np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()), 2*np.pi-np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()))
        W_lep_R = np.abs(W_lep_dphi**2)
        #W_lep_met = np.sqrt(2*W_lep_true.Et()*W_lep_fitted.Et()*(1 - np.cos(W_lep_R)))

        W_had_dphi = min(np.abs(W_had_true.Phi()-W_had_fitted.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-W_had_fitted.Phi()))
        W_had_deta = W_had_true.Eta()-W_had_fitted.Eta()
        W_had_R = np.sqrt(W_had_dphi**2 + W_had_deta**2)

        b_lep_phi_recon = False
        b_lep_eta_recon = False
        b_lep_R_recon = False
        b_had_phi_recon = False
        b_had_eta_recon = False
        b_had_R_recon = False
        W_lep_phi_recon = False
        W_lep_eta_recon = False
        W_lep_R_recon = False
        W_had_phi_recon = False
        W_had_eta_recon = False
        W_had_R_recon = False

        if ((b_lep_true.Phi() - 37.0/50.0) <= b_lep_fitted.Phi()) and (b_lep_fitted.Phi() <= (b_lep_true.Phi() + 37.0/50.0)):
            b_lep_phi_recon = True
        else:
            b_lep_phi_recon = False
        if ((b_lep_true.Eta() - 4.0/5.0) <= b_lep_fitted.Eta()) and (b_lep_fitted.Eta() <= (b_lep_true.Eta()*15.0/14.0 + 13.0/14.0)):
            if (np.abs(b_lep_true.Eta()) <= 1.8) and (b_lep_fitted.Eta() <= 2.0) and (-1.8 <= b_lep_fitted.Eta()):
                b_lep_eta_recon = True
        else:
            b_lep_eta_recon = False
        if (b_lep_phi_recon == True) and (b_lep_eta_recon == True):
            b_lep_R_recon = True
        elif (b_lep_phi_recon == False) and (b_lep_eta_recon == False):
            b_lep_R_recon = False
        else:
            if b_lep_R <= (b_lep_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                b_lep_R_recon = True
            else:
                b_lep_R_recon = False

        if ((b_had_true.Phi() - 37.0/50.0) <= b_had_fitted.Phi()) and (b_had_fitted.Phi() <= (b_had_true.Phi() + 37.0/50.0)):
            b_had_phi_recon = True
        else:
            b_had_phi_recon = False
        if ((b_had_true.Eta()*5.0/4.0 - 19.0/20.0) <= b_had_fitted.Eta()) and (b_had_fitted.Eta() <= (b_had_true.Eta()*5.0/4.0 + 19.0/20.0)):
            if (np.abs(b_had_true.Eta()) <= 2.2) and (np.abs(b_had_fitted.Eta()) <= 2.4):
                b_had_eta_recon = True
        else:
            b_had_eta_recon = False
        if (b_had_phi_recon == True) and (b_had_eta_recon == True):
            b_had_R_recon = True
        elif (b_had_phi_recon == False) and (b_had_eta_recon == False):
            b_had_R_recon = False
        else:
            if b_had_R <= (b_had_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                b_had_R_recon = True
            else:
                b_had_R_recon = False

        if ((W_lep_true.Phi() - 37.0/50.0) <= W_lep_fitted.Phi()) and (W_lep_fitted.Phi() <= (W_lep_true.Phi() + 37.0/50.0)):
            W_lep_R_recon = True
        elif W_lep_R <= (W_lep_dist_t_lim + 0.2):
            W_lep_R_recon = True
        else:
            W_lep_R_recon = False

        if ((W_had_true.Phi() - 57.0/50.0) <= W_had_fitted.Phi()) and (W_had_fitted.Phi() <= (W_had_true.Phi() + 57.0/50.0)):
            W_had_phi_recon = True
        else:
            W_had_phi_recon = False
        if ((W_had_true.Eta() - 4.0/5.0) <= W_had_fitted.Eta()) and (W_had_fitted.Eta() <= (W_had_true.Eta() + 4.0/5.0)):
            if (np.abs(W_had_true.Eta()) <= 1.8) and (np.abs(W_had_fitted.Eta()) <= 1.8):
                W_had_eta_recon = True
        else:
            W_had_eta_recon = False
        if (W_had_phi_recon == True) and (W_had_eta_recon == True):
            W_had_R_recon = True
        elif (W_had_phi_recon == False) and (W_had_eta_recon == False):
            W_had_R_recon = False
        else:
            if W_had_R <= (W_had_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                W_had_R_recon = True
            else:
                W_had_R_recon = False

        b_had_dist_true = 1000
        b_lep_dist_true = 1000
        t_had_dist_true = 1000
        t_lep_dist_true = 1000
        W_had_true_pT = 0
        W_had_dist_true = 10000000
        #W_lep_true_pT = 0
        #W_lep_dist_true = 10000000
        for k in range(len(jets[i])):
            b_had_dphi_true = min(np.abs(b_had_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(b_had_true.Phi()-jets[i][k].Phi()))
            b_had_deta_true = b_had_true.Eta()-jets[i][k].Eta()
            b_had_d_true = np.sqrt(b_had_dphi_true**2+b_had_deta_true**2)
            b_lep_dphi_true = min(np.abs(b_lep_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(b_lep_true.Phi()-jets[i][k].Phi()))
            b_lep_deta_true = b_lep_true.Eta()-jets[i][k].Eta()
            b_lep_d_true = np.sqrt(b_lep_dphi_true**2+b_lep_deta_true**2)
            t_had_dphi_true = min(np.abs(t_had_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(t_had_true.Phi()-jets[i][k].Phi()))
            t_had_deta_true = t_had_true.Eta()-jets[i][k].Eta()#CHECK A RUN TO SEE IF NOT CHECKING ETA FOR T WILL IMPROVE RESULTS
            t_had_d_true = np.sqrt(t_had_dphi_true**2+t_had_deta_true**2)
            t_lep_dphi_true = min(np.abs(t_lep_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(t_lep_true.Phi()-jets[i][k].Phi()))
            t_lep_d_true = np.sqrt(t_lep_dphi_true**2)
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
            if t_had_d_true < t_had_dist_true:
                t_had_dist_true = t_had_d_true
            if t_lep_d_true < t_lep_dist_true:
                t_lep_dist_true = t_lep_d_true
            for j in range(k + 1, len(jets[i])):
                sum_vect = jets[i][k] + jets[i][j] #W_lep_eta values commented out, if that doesn't work try to look into sum_vect since I think this adds a missing E_T jet
                W_had_dphi_true = min(np.abs(W_had_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-sum_vect.Phi()))
                W_had_deta_true = W_had_true.Eta()-sum_vect.Eta()
                W_had_d_true = np.sqrt(W_had_dphi_true**2+W_had_deta_true**2)
                #W_lep_dphi_true = min(np.abs(W_lep_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_lep_true.Phi()-sum_vect.Phi()))
                #W_lep_d_true = np.abs(W_lep_dphi_true)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
                #if W_lep_d_true < W_lep_dist_true:
                    #W_lep_dist_true = W_lep_d_true
                    #W_lep_true_pT = W_lep_true.Pt() - sum_vect.Pt()
        
        W_lep_dist_true = 10000000
        if i == 0:
            for k in range(9):
                W_lep_d_true = np.abs(min(np.abs(W_lep_true.Phi()-jet_mu[k][5]), np.pi-np.abs(W_lep_true.Phi()-jet_mu[k][5])))
                if W_lep_d_true < W_lep_dist_true:
                    W_lep_dist_true = W_lep_d_true
            c = k
        else:
            for k in range(10):
                try:
                    W_lep_d_true = np.abs(min(np.abs(W_lep_true.Phi()-jet_mu[c+k+i][5]), np.pi-np.abs(W_lep_true.Phi()-jet_mu[c+k+i][5])))
                    if W_lep_d_true < W_lep_dist_true:
                        W_lep_dist_true = W_lep_d_true
                except IndexError:
                    pass
            c = c + k

        corr_jets_dist = 0.
        corr_p_jets_dist = 0.

        if (b_lep_dist_true <= b_lep_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_b_lep = good_b_lep + 1
        else:
            bad_b_lep = bad_b_lep + 1
        if (b_had_dist_true <= b_had_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_b_had = good_b_had + 1
        else:
            bad_b_had = bad_b_had + 1
        if (W_lep_dist_true <= W_lep_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_W_lep = good_W_lep + 1
        else:
            bad_W_lep = bad_W_lep + 1
        if (W_had_dist_true <= W_had_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_W_had = good_W_had + 1
        else:
            bad_W_had = bad_W_had + 1

        if corr_jets_dist == 4:
            full_recon_dist_true = full_recon_dist_true + 1
            b_lep_d.Fill(b_lep_R)
            b_had_d.Fill(b_had_R)
            t_lep_d.Fill(t_lep_R)
            t_had_d.Fill(t_had_R)
            W_lep_d.Fill(W_lep_R)
            W_had_d.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_full_dist = p_full_recon_t_full_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_full_dist = p_part_recon_t_full_dist + 1
            else:
                p_un_recon_t_full_dist = p_un_recon_t_full_dist + 1
        elif corr_jets_dist == 3:
            part_recon_dist_true = part_recon_dist_true + 1
            b_lep_d_part.Fill(b_lep_R)
            b_had_d_part.Fill(b_had_R)
            t_lep_d_part.Fill(t_lep_R)
            t_had_d_part.Fill(t_had_R)
            W_lep_d_part.Fill(W_lep_R)
            W_had_d_part.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_part_dist = p_full_recon_t_part_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_part_dist = p_part_recon_t_part_dist + 1
            else:
                p_un_recon_t_part_dist = p_un_recon_t_part_dist + 1
        else:
            un_recon_dist_true = un_recon_dist_true + 1
            b_lep_d_un.Fill(b_lep_R)
            b_had_d_un.Fill(b_had_R)
            t_lep_d_un.Fill(t_lep_R)
            t_had_d_un.Fill(t_had_R)
            W_lep_d_un.Fill(W_lep_R)
            W_had_d_un.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_un_dist = p_full_recon_t_un_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_un_dist = p_part_recon_t_un_dist + 1
            else:
                p_un_recon_t_un_dist = p_un_recon_t_un_dist + 1
        h_W_lep_met_arr.append(np.float(met_obs))
        h_W_lep_met_arr_t.append(np.float(met_true))
        h_W_lep_met_arr_p.append(np.float(met_pred))
        h_b_lep_arr.append(np.float(b_lep_dist_true))
        h_b_had_arr.append(np.float(b_had_dist_true))
        h_t_lep_arr.append(np.float(t_lep_dist_true))
        h_t_had_arr.append(np.float(t_had_dist_true))
        h_W_lep_arr.append(np.float(W_lep_dist_true))
        h_W_had_arr.append(np.float(W_had_dist_true))
        h_pT_W_had_arr.append(np.float(W_had_true_pT))

    print('good_W_had', good_W_had, 'bad_W_had', bad_W_had)
    print('good_W_lep', good_W_lep, 'bad_W_lep', bad_W_lep)
    print('good_b_had', good_b_had, 'bad_b_had', bad_b_had)
    print('good_b_lep', good_b_lep, 'bad_b_lep', bad_b_lep)
    good_b_had = 0.
    good_b_lep = 0.
    good_W_had = 0.
    good_W_lep = 0.
    bad_b_had = 0.
    bad_b_lep = 0.
    bad_W_had = 0.
    bad_W_lep = 0.
    for i in range(n_events):
        if (h_b_lep_arr[i] <= b_lep_dist_t_lim):
            good_b_lep = good_b_lep + 1
        else:
            bad_b_lep = bad_b_lep + 1
        if (h_b_had_arr[i] <= b_had_dist_t_lim):
            good_b_had = good_b_had + 1
        else:
            bad_b_had = bad_b_had + 1
        if (h_W_lep_arr[i] <= W_lep_dist_t_lim):
            good_W_lep = good_W_lep + 1
        else:
            bad_W_lep = bad_W_lep + 1
        if (h_W_had_arr[i] <= W_had_dist_t_lim):
            good_W_had = good_W_had + 1
        else:
            bad_W_had = bad_W_had + 1
        h_b_lep_true.Fill(h_b_lep_arr[i])
        h_b_had_true.Fill(h_b_had_arr[i])
        h_t_had_true.Fill(h_t_had_arr[i])
        h_t_lep_true.Fill(h_t_lep_arr[i])
        h_W_had_true.Fill(h_W_had_arr[i])
        h_pT_W_had_true.Fill(h_pT_W_had_arr[i])
        h_W_lep_true.Fill(h_W_lep_arr[i])
        W_lep_met_d.Fill(h_W_lep_met_arr[i])
        W_lep_met_d_t.Fill(h_W_lep_met_arr_t[i])
        W_lep_met_d_p.Fill(h_W_lep_met_arr_p[i])
  
    print('Total number of events: ', n_events, '\n')
    print('Percentage of jets analysis for Distances, True vs Observed:')
    print('Number, Percentage of Truth events with fully reconstructable Jets:                       ', full_recon_dist_true, 'events, ', 100*full_recon_dist_true/n_events, '%')
    print('Number, Percentage of Truth events with partially reconstructable Jets:                   ', part_recon_dist_true, 'events, ', 100*part_recon_dist_true/n_events, '%')
    print('Number, Percentage of Truth events with unreconstructable Jets:                           ', un_recon_dist_true, ' events, ', 100*un_recon_dist_true/n_events, '%')
    print('SUM CHECK:                                                                                ', full_recon_dist_true + part_recon_dist_true + un_recon_dist_true, ' events, ', 100*(full_recon_dist_true + part_recon_dist_true + un_recon_dist_true)/n_events, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_full_dist, ' events, ', 100*p_full_recon_t_full_dist/full_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_full_dist, ' events, ', 100*p_part_recon_t_full_dist/full_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_full_dist, ' events, ', 100*p_un_recon_t_full_dist/full_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_full_dist + p_part_recon_t_full_dist + p_un_recon_t_full_dist, ' events, ', 100*(p_full_recon_t_full_dist + p_part_recon_t_full_dist + p_un_recon_t_full_dist)/full_recon_dist_true, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth Partially Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_part_dist, ' events, ', 100*p_full_recon_t_part_dist/part_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_part_dist, ' events, ', 100*p_part_recon_t_part_dist/part_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_part_dist, ' events, ', 100*p_un_recon_t_part_dist/part_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_part_dist + p_part_recon_t_part_dist + p_un_recon_t_part_dist, ' events, ', 100*(p_full_recon_t_part_dist + p_part_recon_t_part_dist + p_un_recon_t_part_dist)/part_recon_dist_true, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth not Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_un_dist, '  events, ', 100*p_full_recon_t_un_dist/un_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_un_dist, ' events, ', 100*p_part_recon_t_un_dist/un_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_un_dist, ' events, ', 100*p_un_recon_t_un_dist/un_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_un_dist + p_part_recon_t_un_dist + p_un_recon_t_un_dist, ' events, ', 100*(p_full_recon_t_un_dist + p_part_recon_t_un_dist + p_un_recon_t_un_dist)/un_recon_dist_true, '%')
    print('=================================================================')
    print('=================================================================')
    print(100*good_W_had/(bad_W_had + good_W_had), ' good_W_had')
    print(100*good_W_lep/(bad_W_lep + good_W_lep), ' good_W_lep')
    print(100*good_b_had/(bad_b_had + good_b_had), ' good_b_had')
    print(100*good_b_lep/(bad_b_lep + good_b_lep), ' good_b_lep')

    c1 = TCanvas()
    W_lep_d.Draw()
    c1.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_recon.png')
    c1.Close()

    c2 = TCanvas()
    W_had_d.Draw()
    c2.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_recon.png')
    c2.Close()

    c3 = TCanvas()
    b_lep_d.Draw()
    c3.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_recon.png')
    c3.Close()

    c4 = TCanvas()
    b_had_d.Draw()
    c4.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_recon.png')
    c4.Close()

    c5 = TCanvas()
    t_lep_d.Draw()
    c5.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_recon.png')
    c5.Close()

    c6 = TCanvas()
    t_had_d.Draw()
    c6.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_recon.png')
    c6.Close()

    c7 = TCanvas()
    W_lep_d_part.Draw()
    c7.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_part_recon.png')
    c7.Close()

    c8 = TCanvas()
    W_had_d_part.Draw()
    c8.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_part_recon.png')
    c8.Close()

    c9 = TCanvas()
    b_lep_d_part.Draw()
    c9.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_part_recon.png')
    c9.Close()

    c10 = TCanvas()
    b_had_d_part.Draw()
    c10.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_part_recon.png')
    c10.Close()

    c11 = TCanvas()
    t_lep_d_part.Draw()
    c11.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_part_recon.png')
    c11.Close()

    c12 = TCanvas()
    t_had_d_part.Draw()
    c12.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_part_recon.png')
    c12.Close()

    c13 = TCanvas()
    h_b_lep_true.Draw()
    c13.SaveAs(outputdir + subdir + 'leptonic_b_true_dist.png')
    c13.Close()

    c14 = TCanvas()
    h_b_had_true.Draw()
    c14.SaveAs(outputdir + subdir + 'hadronic_b_true_dist.png')
    c14.Close()

    c15 = TCanvas()
    h_t_lep_true.Draw()
    c15.SaveAs(outputdir + subdir + 'leptonic_t_true_dist.png')
    c15.Close()

    c16 = TCanvas()
    h_t_had_true.Draw()
    c16.SaveAs(outputdir + subdir + 'hadronic_t_true_dist.png')
    c16.Close()

    c17 = TCanvas()
    h_W_had_true.Draw()
    c17.SaveAs(outputdir + subdir + 'hadronic_W_true_dist.png')
    c17.Close()

    c18 = TCanvas()
    h_pT_W_had_true.Draw()
    c18.SaveAs(outputdir + subdir + 'hadronic_W_true_pT_dist.png')
    c18.Close()

    c19 = TCanvas()
    h_W_lep_true.Draw()
    c19.SaveAs(outputdir + subdir + 'leptonic_W_true_dist.png')
    c19.Close()

    #c20 = TCanvas()
    #h_pT_W_lep_true.Draw()
    #c20.SaveAs(outputdir + subdir + 'leptonic_W_true_pT_dist.png')
    #c20.Close()

    c21 = TCanvas()
    W_lep_d_un.Draw()
    c21.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_un_recon.png')
    c21.Close()

    c22 = TCanvas()
    W_had_d_un.Draw()
    c22.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_un_recon.png')
    c22.Close()

    c23 = TCanvas()
    b_lep_d_un.Draw()
    c23.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_un_recon.png')
    c23.Close()

    c24 = TCanvas()
    b_had_d_un.Draw()
    c24.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_un_recon.png')
    c24.Close()

    c25 = TCanvas()
    t_lep_d_un.Draw()
    c25.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_un_recon.png')
    c25.Close()

    c26 = TCanvas()
    t_had_d_un.Draw()
    c26.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_un_recon.png')
    c26.Close()

    c27 = TCanvas()
    W_lep_met_d.Draw()
    c27.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_observed.png')
    c27.Close()

    c28 = TCanvas()
    W_lep_met_d_t.Draw()
    c28.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_true.png')
    c28.Close()

    c29 = TCanvas()
    W_lep_met_d_p.Draw()
    c29.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_predicted.png')
    c29.Close()

def make_correlations():
    c_W_had_true = TH2F( "c_W_had_true", "c_W_had_true", 50, 0., 1000., 50, 0., 1000. )
    c_W_had_fitted = TH2F( "c_W_had_fitted", "c_W_had_fitted", 50, 0., 1000., 50, 0., 1000. )
    
    jets = []

    for i in range(n_events):
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W )
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W)
    
        jet_mu_vect = MakeP4(jet_mu[i], jet_mu[i][4])
        jet_1_vect = MakeP4(jet_1[i], jet_1[i][4])
        jet_2_vect = MakeP4(jet_2[i], jet_2[i][4])
        jet_3_vect = MakeP4(jet_3[i], jet_3[i][4])
        jet_4_vect = MakeP4(jet_4[i], jet_4[i][4])
        jet_5_vect = MakeP4(jet_5[i], jet_5[i][4])
        
        jets.append([])
        jets[i].append(jet_1_vect)
        jets[i].append(jet_2_vect)
        jets[i].append(jet_3_vect)
        jets[i].append(jet_4_vect)
        jets[i].append(jet_5_vect)
        
        W_had_dist_true = 10000000
        p_had_true_total = 0
        for k in range(len(jets[i])):
            for j in range(i, len(jets[i])):
                #px_total = jets[i][k].Px() + jets[i][j].Px()
                #py_total = jets[i][k].Py() + jets[i][j].Py()
                #pz_total = jets[i][k].Pz() + jets[i][j].Pz()
                #d = np.sqrt((W_had_true.Px()-px_total)**2+(W_had_true.Py()-py_total)**2+(W_had_true.Pz()-pz_total)**2)
                sum_vect = jets[i][k] + jets[i][j]
                dphi = min(np.abs(W_had_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-sum_vect.Phi()))
                d = np.sqrt(dphi**2+(W_had_true.Eta()-sum_vect.Eta())**2)
                if d < W_had_dist_true:
                    W_had_dist_true = d
                    p_had_true_total = sum_vect.Px()**2 + sum_vect.Py()**2 + sum_vect.Pz()**2
        c_W_had_true.Fill(W_had_true.Px()**2+W_had_true.Py()**2+W_had_true.Pz()**2, p_had_true_total)
    
        W_had_dist_fitted = 10000000
        p_had_fitted_total = 0
        for k in range(len(jets[i])):
            for j in range(i, len(jets[i])):
                #px_total = jets[i][k].Px() + jets[i][j].Px()
                #py_total = jets[i][k].Py() + jets[i][j].Py()
                #pz_total = jets[i][k].Pz() + jets[i][j].Pz()
                #d = np.sqrt((W_had_fitted.Px()-px_total)**2+(W_had_fitted.Py()-py_total)**2+(W_had_fitted.Pz()-pz_total)**2)
                sum_vect = jets[i][k] + jets[i][j]
                dphi = min(np.abs(W_had_fitted.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_fitted.Phi()-sum_vect.Phi()))
                d = np.sqrt(dphi**2+(W_had_fitted.Eta()-sum_vect.Eta())**2)
                if d < W_had_dist_fitted:
                    W_had_dist_fitted = d
                    p_had_fitted_total = sum_vect.Px()**2 + sum_vect.Py()**2 + sum_vect.Pz()**2
        c_W_had_fitted.Fill(W_had_fitted.Px()**2+W_had_fitted.Py()**2+W_had_fitted.Pz()**2, p_had_fitted_total)
        
    c11 = TCanvas()
    c_W_had_true.Draw()
    corr_had_true = c_W_had_true.GetCorrelationFactor()
    l3 = TLatex()
    l3.SetNDC()
    l3.SetTextFont(42)
    l3.SetTextColor(kBlack)
    l3.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr_had_true)
    c11.SaveAs(outputdir + subdir + 'hadronic_W_true_corr.png')
    c11.Close()
    
    c12 = TCanvas()
    c_W_had_fitted.Draw()
    corr_had_fitted = c_W_had_fitted.GetCorrelationFactor()
    l4 = TLatex()
    l4.SetNDC()
    l4.SetTextFont(42)
    l4.SetTextColor(kBlack)
    l4.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr_had_fitted)
    c11.SaveAs(outputdir + subdir + '/hadronic_W_fitted_corr.png')
    c11.Close()
    
if __name__ == "__main__":
    try:
        os.mkdir('{}/closejets_img'.format(outputdir))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()
    #make_correlations()
=======
import os, sys
import numpy as np
from scipy.spatial import distance
from ROOT import *
#import ROOT
import pickle

representation = sys.argv[2]
outputdir = sys.argv[1]
subdir = '/closejets_img/'
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95

def MakeLimit(tvo):
	pvt = ((np.pi)/8) * (np.sin(2*tvo - ((np.pi)/2)) + 1)
	pvt = float(round(pvt, 2))
	return pvt

def MakeP4_lep(y,m):
    p4 = TLorentzVector()
    p0 = y[0]
    p1 = y[1]
    p2 = y[3]
    E = np.sqrt(p0**2 + p1**2 + p2**2 + m**2)
    p4.SetPxPyPzE(p0, p1, p2, E)
    return p4

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
      jets_jets = jets[:,6:]
      jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6))
      jets_jets = np.delete(jets_jets, 5, 2)
      jets_jets = jets_jets.reshape((jets_jets.shape[0], 25))
      jets_lep = lep_scalar.inverse_transform(jets_lep)
      jets_jets = jets_scalar.inverse_transform(jets_jets)
      jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))#I think this is the final 6x6 array the arxiv paper was talking about

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

n_events = true.shape[0]
w = 1

def make_histograms():
    W_lep_met_d = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Observed", 50, 0, 120)
    W_lep_met_d.SetTitle("W Leptonic Transverse Mass, Observed;Leptonic (GeV);A.U.")
    W_lep_met_d_t = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Truth", 50, 0, 120)
    W_lep_met_d_t.SetTitle("W Leptonic Transverse Mass, Truth;Leptonic (GeV);A.U.")
    W_lep_met_d_p = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Predicted", 50, 0, 120)
    W_lep_met_d_p.SetTitle("W Leptonic Transverse Mass, Predicted;Leptonic (GeV);A.U.")

    W_lep_d = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    W_had_d = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Reconstructed", 50, 0, 5)
    W_lep_d.SetTitle("W Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (radians);A.U.")
    W_had_d.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (radians);A.U.")

    b_lep_d = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    b_had_d = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    b_lep_d.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
    b_had_d.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")

    t_lep_d = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    t_had_d = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 5)
    t_lep_d.SetTitle("t Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
    t_had_d.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")

    W_lep_d_part = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    W_had_d_part = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    W_lep_d_part.SetTitle("W Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (radians);A.U.")
    W_had_d_part.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (radians);A.U.")

    b_lep_d_part = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    b_had_d_part = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    b_lep_d_part.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
    b_had_d_part.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")

    t_lep_d_part = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    t_had_d_part = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 5)
    t_lep_d_part.SetTitle("t Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
    t_had_d_part.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")

    W_lep_d_un = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    W_had_d_un = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Not Reconstructed", 50, 0, 5)
    W_lep_d_un.SetTitle("W Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (radians);A.U.")
    W_had_d_un.SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (radians);A.U.")

    b_lep_d_un = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    b_had_d_un = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    b_lep_d_un.SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")
    b_had_d_un.SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

    t_lep_d_un = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    t_had_d_un = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 5)
    t_lep_d_un.SetTitle("t Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")
    t_had_d_un.SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

    h_W_had_true = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 5)
    h_pT_W_had_true = TH1F("h_pT_W_had_true","W Hadronic p_T, True vs Observed", 50, -500, 500)
    h_W_had_true.SetTitle("W Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    h_W_lep_true = TH1F("h_W_lep_true","W Leptonic Distances, True vs Observed", 50, 0, 5)
    h_W_lep_true.SetTitle("W Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")

    h_b_lep_true = TH1F("h_b_lep_true","b Leptonic Distances, True vs Observed", 50, 0, 5)
    h_b_had_true = TH1F("h_b_had_true","b Hadronic Distances, True vs Observed", 50, 0, 5)
    h_b_lep_true.SetTitle("b Leptonic #eta-#phi distances, True vs Observed;true leptonic (radians);A.U.")
    h_b_had_true.SetTitle("b Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    h_t_lep_true = TH1F("h_t_lep_true","t Leptonic Distances, True vs Observed", 50, 0, 5)
    h_t_had_true = TH1F("h_t_had_true","t Hadronic Distances, True vs Observed", 50, 0, 5)
    h_t_lep_true.SetTitle("t Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")
    h_t_had_true.SetTitle("t Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")

    jets = []

    b_lep_dist_t_lim = 0.39
    b_had_dist_t_lim = 0.39
    t_lep_dist_t_lim = 0.80
    t_had_dist_t_lim = 0.80
    W_lep_dist_t_lim = 0.82
    W_had_dist_t_lim = 1.28

    full_recon_dist_true = 0.
    part_recon_dist_true = 0.
    un_recon_dist_true = 0.
    full_recon_phi_true = 0.
    part_recon_phi_true = 0.
    un_recon_phi_true = 0.
    full_recon_eta_true = 0.
    part_recon_eta_true = 0.
    un_recon_eta_true = 0.

    p_full_recon_t_full_dist = 0.
    p_part_recon_t_full_dist = 0.
    p_un_recon_t_full_dist = 0.
    p_full_recon_t_part_dist = 0.
    p_part_recon_t_part_dist = 0.
    p_un_recon_t_part_dist = 0.
    p_full_recon_t_un_dist = 0.
    p_part_recon_t_un_dist = 0.
    p_un_recon_t_un_dist = 0.

    b_lep_dist_p_corr_t_full = 0.
    b_had_dist_p_corr_t_full = 0.
    t_lep_dist_p_corr_t_full = 0.
    t_had_dist_p_corr_t_full = 0.
    W_lep_dist_p_corr_t_full = 0.
    W_had_dist_p_corr_t_full = 0.

    b_lep_dist_p_corr_t_part = 0.
    b_had_dist_p_corr_t_part = 0.
    t_lep_dist_p_corr_t_part = 0.
    t_had_dist_p_corr_t_part = 0.
    W_lep_dist_p_corr_t_part = 0.
    W_had_dist_p_corr_t_part = 0.

    good_b_had = 0.
    good_b_lep = 0.
    good_W_had = 0.
    good_W_lep = 0.

    bad_b_had = 0.
    bad_b_lep = 0.
    bad_W_had = 0.
    bad_W_lep = 0.

    jump = 0

    h_b_lep_arr = []
    h_b_had_arr = []
    h_t_lep_arr = []
    h_t_had_arr = []
    h_W_lep_arr = []
    h_W_had_arr = []
    h_pT_W_had_arr = []
    h_W_lep_met_arr = []
    h_W_lep_met_arr_t = []
    h_W_lep_met_arr_p = []

    for i in range(n_events):
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
        
        jets.append([])
        jets[i].append(jet_1_vect)
        jets[i].append(jet_2_vect)
        jets[i].append(jet_3_vect)
        jets[i].append(jet_4_vect)
        jets[i].append(jet_5_vect)

        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(jet_mu[i][5])))
        met_true = np.sqrt(2*W_lep_true.Pt()*W_lep_true.Et()*(1 - np.cos(W_lep_true.Phi())))
        met_pred = np.sqrt(2*W_lep_fitted.Pt()*W_lep_fitted.Et()*(1 - np.cos(W_lep_fitted.Phi())))

        b_lep_dphi = min(np.abs(b_lep_true.Phi()-b_lep_fitted.Phi()), 2*np.pi-np.abs(b_lep_true.Phi()-b_lep_fitted.Phi()))
        b_lep_deta = b_lep_true.Eta()-b_lep_fitted.Eta()
        b_lep_R = np.sqrt(b_lep_dphi**2 + b_lep_deta**2)

        b_had_dphi = min(np.abs(b_had_true.Phi()-b_had_fitted.Phi()), 2*np.pi-np.abs(b_had_true.Phi()-b_had_fitted.Phi()))
        b_had_deta = b_had_true.Eta()-b_had_fitted.Eta()
        b_had_R = np.sqrt(b_had_dphi**2 + b_had_deta**2)

        t_lep_dphi = min(np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()), 2*np.pi-np.abs(t_lep_true.Phi()-t_lep_fitted.Phi()))
        t_lep_R = np.sqrt(t_lep_dphi**2)

        t_had_dphi = min(np.abs(t_had_true.Phi()-t_had_fitted.Phi()), 2*np.pi-np.abs(t_had_true.Phi()-t_had_fitted.Phi()))
        t_had_deta = t_had_true.Eta()-t_had_fitted.Eta()
        t_had_R = np.sqrt(t_had_dphi**2 + t_had_deta**2)

        W_lep_dphi = min(np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()), 2*np.pi-np.abs(W_lep_true.Phi()-W_lep_fitted.Phi()))
        W_lep_R = np.abs(W_lep_dphi**2)
        #W_lep_met = np.sqrt(2*W_lep_true.Et()*W_lep_fitted.Et()*(1 - np.cos(W_lep_R)))

        W_had_dphi = min(np.abs(W_had_true.Phi()-W_had_fitted.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-W_had_fitted.Phi()))
        W_had_deta = W_had_true.Eta()-W_had_fitted.Eta()
        W_had_R = np.sqrt(W_had_dphi**2 + W_had_deta**2)

        b_lep_phi_recon = False
        b_lep_eta_recon = False
        b_lep_R_recon = False
        b_had_phi_recon = False
        b_had_eta_recon = False
        b_had_R_recon = False
        W_lep_phi_recon = False
        W_lep_eta_recon = False
        W_lep_R_recon = False
        W_had_phi_recon = False
        W_had_eta_recon = False
        W_had_R_recon = False

        if ((b_lep_true.Phi() - 37.0/50.0) <= b_lep_fitted.Phi()) and (b_lep_fitted.Phi() <= (b_lep_true.Phi() + 37.0/50.0)):
            b_lep_phi_recon = True
        else:
            b_lep_phi_recon = False
        if ((b_lep_true.Eta() - 4.0/5.0) <= b_lep_fitted.Eta()) and (b_lep_fitted.Eta() <= (b_lep_true.Eta()*15.0/14.0 + 13.0/14.0)):
            if (np.abs(b_lep_true.Eta()) <= 1.8) and (b_lep_fitted.Eta() <= 2.0) and (-1.8 <= b_lep_fitted.Eta()):
                b_lep_eta_recon = True
        else:
            b_lep_eta_recon = False
        if (b_lep_phi_recon == True) and (b_lep_eta_recon == True):
            b_lep_R_recon = True
        elif (b_lep_phi_recon == False) and (b_lep_eta_recon == False):
            b_lep_R_recon = False
        else:
            if b_lep_R <= (b_lep_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                b_lep_R_recon = True
            else:
                b_lep_R_recon = False

        if ((b_had_true.Phi() - 37.0/50.0) <= b_had_fitted.Phi()) and (b_had_fitted.Phi() <= (b_had_true.Phi() + 37.0/50.0)):
            b_had_phi_recon = True
        else:
            b_had_phi_recon = False
        if ((b_had_true.Eta()*5.0/4.0 - 19.0/20.0) <= b_had_fitted.Eta()) and (b_had_fitted.Eta() <= (b_had_true.Eta()*5.0/4.0 + 19.0/20.0)):
            if (np.abs(b_had_true.Eta()) <= 2.2) and (np.abs(b_had_fitted.Eta()) <= 2.4):
                b_had_eta_recon = True
        else:
            b_had_eta_recon = False
        if (b_had_phi_recon == True) and (b_had_eta_recon == True):
            b_had_R_recon = True
        elif (b_had_phi_recon == False) and (b_had_eta_recon == False):
            b_had_R_recon = False
        else:
            if b_had_R <= (b_had_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                b_had_R_recon = True
            else:
                b_had_R_recon = False

        if ((W_lep_true.Phi() - 37.0/50.0) <= W_lep_fitted.Phi()) and (W_lep_fitted.Phi() <= (W_lep_true.Phi() + 37.0/50.0)):
            W_lep_R_recon = True
        elif W_lep_R <= (W_lep_dist_t_lim + 0.2):
            W_lep_R_recon = True
        else:
            W_lep_R_recon = False

        if ((W_had_true.Phi() - 57.0/50.0) <= W_had_fitted.Phi()) and (W_had_fitted.Phi() <= (W_had_true.Phi() + 57.0/50.0)):
            W_had_phi_recon = True
        else:
            W_had_phi_recon = False
        if ((W_had_true.Eta() - 4.0/5.0) <= W_had_fitted.Eta()) and (W_had_fitted.Eta() <= (W_had_true.Eta() + 4.0/5.0)):
            if (np.abs(W_had_true.Eta()) <= 1.8) and (np.abs(W_had_fitted.Eta()) <= 1.8):
                W_had_eta_recon = True
        else:
            W_had_eta_recon = False
        if (W_had_phi_recon == True) and (W_had_eta_recon == True):
            W_had_R_recon = True
        elif (W_had_phi_recon == False) and (W_had_eta_recon == False):
            W_had_R_recon = False
        else:
            if W_had_R <= (W_had_dist_t_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                W_had_R_recon = True
            else:
                W_had_R_recon = False

        b_had_dist_true = 1000
        b_lep_dist_true = 1000
        t_had_dist_true = 1000
        t_lep_dist_true = 1000
        W_had_true_pT = 0
        W_had_dist_true = 10000000
        #W_lep_true_pT = 0
        #W_lep_dist_true = 10000000
        for k in range(len(jets[i])):
            b_had_dphi_true = min(np.abs(b_had_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(b_had_true.Phi()-jets[i][k].Phi()))
            b_had_deta_true = b_had_true.Eta()-jets[i][k].Eta()
            b_had_d_true = np.sqrt(b_had_dphi_true**2+b_had_deta_true**2)
            b_lep_dphi_true = min(np.abs(b_lep_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(b_lep_true.Phi()-jets[i][k].Phi()))
            b_lep_deta_true = b_lep_true.Eta()-jets[i][k].Eta()
            b_lep_d_true = np.sqrt(b_lep_dphi_true**2+b_lep_deta_true**2)
            t_had_dphi_true = min(np.abs(t_had_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(t_had_true.Phi()-jets[i][k].Phi()))
            t_had_deta_true = t_had_true.Eta()-jets[i][k].Eta()#CHECK A RUN TO SEE IF NOT CHECKING ETA FOR T WILL IMPROVE RESULTS
            t_had_d_true = np.sqrt(t_had_dphi_true**2+t_had_deta_true**2)
            t_lep_dphi_true = min(np.abs(t_lep_true.Phi()-jets[i][k].Phi()), 2*np.pi-np.abs(t_lep_true.Phi()-jets[i][k].Phi()))
            t_lep_d_true = np.sqrt(t_lep_dphi_true**2)
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
            if t_had_d_true < t_had_dist_true:
                t_had_dist_true = t_had_d_true
            if t_lep_d_true < t_lep_dist_true:
                t_lep_dist_true = t_lep_d_true
            for j in range(k + 1, len(jets[i])):
                sum_vect = jets[i][k] + jets[i][j] #W_lep_eta values commented out, if that doesn't work try to look into sum_vect since I think this adds a missing E_T jet
                W_had_dphi_true = min(np.abs(W_had_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-sum_vect.Phi()))
                W_had_deta_true = W_had_true.Eta()-sum_vect.Eta()
                W_had_d_true = np.sqrt(W_had_dphi_true**2+W_had_deta_true**2)
                #W_lep_dphi_true = min(np.abs(W_lep_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_lep_true.Phi()-sum_vect.Phi()))
                #W_lep_d_true = np.abs(W_lep_dphi_true)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
                #if W_lep_d_true < W_lep_dist_true:
                    #W_lep_dist_true = W_lep_d_true
                    #W_lep_true_pT = W_lep_true.Pt() - sum_vect.Pt()
        
        W_lep_dist_true = 10000000
        if i == 0:
            for k in range(9):
                W_lep_d_true = np.abs(min(np.abs(W_lep_true.Phi()-jet_mu[k][5]), np.pi-np.abs(W_lep_true.Phi()-jet_mu[k][5])))
                if W_lep_d_true < W_lep_dist_true:
                    W_lep_dist_true = W_lep_d_true
            c = k
        else:
            for k in range(10):
                try:
                    W_lep_d_true = np.abs(min(np.abs(W_lep_true.Phi()-jet_mu[c+k+i][5]), np.pi-np.abs(W_lep_true.Phi()-jet_mu[c+k+i][5])))
                    if W_lep_d_true < W_lep_dist_true:
                        W_lep_dist_true = W_lep_d_true
                except IndexError:
                    pass
            c = c + k

        corr_jets_dist = 0.
        corr_p_jets_dist = 0.

        if (b_lep_dist_true <= b_lep_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_b_lep = good_b_lep + 1
        else:
            bad_b_lep = bad_b_lep + 1
        if (b_had_dist_true <= b_had_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_b_had = good_b_had + 1
        else:
            bad_b_had = bad_b_had + 1
        if (W_lep_dist_true <= W_lep_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_W_lep = good_W_lep + 1
        else:
            bad_W_lep = bad_W_lep + 1
        if (W_had_dist_true <= W_had_dist_t_lim):
            corr_jets_dist = corr_jets_dist + 1
            good_W_had = good_W_had + 1
        else:
            bad_W_had = bad_W_had + 1

        if corr_jets_dist == 4:
            full_recon_dist_true = full_recon_dist_true + 1
            b_lep_d.Fill(b_lep_R)
            b_had_d.Fill(b_had_R)
            t_lep_d.Fill(t_lep_R)
            t_had_d.Fill(t_had_R)
            W_lep_d.Fill(W_lep_R)
            W_had_d.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_full_dist = p_full_recon_t_full_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_full_dist = p_part_recon_t_full_dist + 1
            else:
                p_un_recon_t_full_dist = p_un_recon_t_full_dist + 1
        elif corr_jets_dist == 3:
            part_recon_dist_true = part_recon_dist_true + 1
            b_lep_d_part.Fill(b_lep_R)
            b_had_d_part.Fill(b_had_R)
            t_lep_d_part.Fill(t_lep_R)
            t_had_d_part.Fill(t_had_R)
            W_lep_d_part.Fill(W_lep_R)
            W_had_d_part.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_part_dist = p_full_recon_t_part_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_part_dist = p_part_recon_t_part_dist + 1
            else:
                p_un_recon_t_part_dist = p_un_recon_t_part_dist + 1
        else:
            un_recon_dist_true = un_recon_dist_true + 1
            b_lep_d_un.Fill(b_lep_R)
            b_had_d_un.Fill(b_had_R)
            t_lep_d_un.Fill(t_lep_R)
            t_had_d_un.Fill(t_had_R)
            W_lep_d_un.Fill(W_lep_R)
            W_had_d_un.Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (b_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1
            if (W_had_R_recon == True):
                corr_p_jets_dist = corr_p_jets_dist + 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_un_dist = p_full_recon_t_un_dist + 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_un_dist = p_part_recon_t_un_dist + 1
            else:
                p_un_recon_t_un_dist = p_un_recon_t_un_dist + 1
        h_W_lep_met_arr.append(np.float(met_obs))
        h_W_lep_met_arr_t.append(np.float(met_true))
        h_W_lep_met_arr_p.append(np.float(met_pred))
        h_b_lep_arr.append(np.float(b_lep_dist_true))
        h_b_had_arr.append(np.float(b_had_dist_true))
        h_t_lep_arr.append(np.float(t_lep_dist_true))
        h_t_had_arr.append(np.float(t_had_dist_true))
        h_W_lep_arr.append(np.float(W_lep_dist_true))
        h_W_had_arr.append(np.float(W_had_dist_true))
        h_pT_W_had_arr.append(np.float(W_had_true_pT))

    print('good_W_had', good_W_had, 'bad_W_had', bad_W_had)
    print('good_W_lep', good_W_lep, 'bad_W_lep', bad_W_lep)
    print('good_b_had', good_b_had, 'bad_b_had', bad_b_had)
    print('good_b_lep', good_b_lep, 'bad_b_lep', bad_b_lep)
    good_b_had = 0.
    good_b_lep = 0.
    good_W_had = 0.
    good_W_lep = 0.
    bad_b_had = 0.
    bad_b_lep = 0.
    bad_W_had = 0.
    bad_W_lep = 0.
    for i in range(n_events):
        if (h_b_lep_arr[i] <= b_lep_dist_t_lim):
            good_b_lep = good_b_lep + 1
        else:
            bad_b_lep = bad_b_lep + 1
        if (h_b_had_arr[i] <= b_had_dist_t_lim):
            good_b_had = good_b_had + 1
        else:
            bad_b_had = bad_b_had + 1
        if (h_W_lep_arr[i] <= W_lep_dist_t_lim):
            good_W_lep = good_W_lep + 1
        else:
            bad_W_lep = bad_W_lep + 1
        if (h_W_had_arr[i] <= W_had_dist_t_lim):
            good_W_had = good_W_had + 1
        else:
            bad_W_had = bad_W_had + 1
        h_b_lep_true.Fill(h_b_lep_arr[i])
        h_b_had_true.Fill(h_b_had_arr[i])
        h_t_had_true.Fill(h_t_had_arr[i])
        h_t_lep_true.Fill(h_t_lep_arr[i])
        h_W_had_true.Fill(h_W_had_arr[i])
        h_pT_W_had_true.Fill(h_pT_W_had_arr[i])
        h_W_lep_true.Fill(h_W_lep_arr[i])
        W_lep_met_d.Fill(h_W_lep_met_arr[i])
        W_lep_met_d_t.Fill(h_W_lep_met_arr_t[i])
        W_lep_met_d_p.Fill(h_W_lep_met_arr_p[i])
  
    print('Total number of events: ', n_events, '\n')
    print('Percentage of jets analysis for Distances, True vs Observed:')
    print('Number, Percentage of Truth events with fully reconstructable Jets:                       ', full_recon_dist_true, 'events, ', 100*full_recon_dist_true/n_events, '%')
    print('Number, Percentage of Truth events with partially reconstructable Jets:                   ', part_recon_dist_true, 'events, ', 100*part_recon_dist_true/n_events, '%')
    print('Number, Percentage of Truth events with unreconstructable Jets:                           ', un_recon_dist_true, ' events, ', 100*un_recon_dist_true/n_events, '%')
    print('SUM CHECK:                                                                                ', full_recon_dist_true + part_recon_dist_true + un_recon_dist_true, ' events, ', 100*(full_recon_dist_true + part_recon_dist_true + un_recon_dist_true)/n_events, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_full_dist, ' events, ', 100*p_full_recon_t_full_dist/full_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_full_dist, ' events, ', 100*p_part_recon_t_full_dist/full_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_full_dist, ' events, ', 100*p_un_recon_t_full_dist/full_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_full_dist + p_part_recon_t_full_dist + p_un_recon_t_full_dist, ' events, ', 100*(p_full_recon_t_full_dist + p_part_recon_t_full_dist + p_un_recon_t_full_dist)/full_recon_dist_true, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth Partially Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_part_dist, ' events, ', 100*p_full_recon_t_part_dist/part_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_part_dist, ' events, ', 100*p_part_recon_t_part_dist/part_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_part_dist, ' events, ', 100*p_un_recon_t_part_dist/part_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_part_dist + p_part_recon_t_part_dist + p_un_recon_t_part_dist, ' events, ', 100*(p_full_recon_t_part_dist + p_part_recon_t_part_dist + p_un_recon_t_part_dist)/part_recon_dist_true, '%')
    print('=================================================================')
    print('Percentage of jets analysis for Distances, Predicted vs Truth not Reconstructed:')
    print('Number, Percentage of Predicted events with fully reconstructed Jets:                     ', p_full_recon_t_un_dist, '  events, ', 100*p_full_recon_t_un_dist/un_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with partially reconstructed Jets:                 ', p_part_recon_t_un_dist, ' events, ', 100*p_part_recon_t_un_dist/un_recon_dist_true, '%')
    print('Number, Percentage of Predicted events with not reconstructed Jets:                       ', p_un_recon_t_un_dist, ' events, ', 100*p_un_recon_t_un_dist/un_recon_dist_true, '%')
    print('SUM CHECK:                                                                                ', p_full_recon_t_un_dist + p_part_recon_t_un_dist + p_un_recon_t_un_dist, ' events, ', 100*(p_full_recon_t_un_dist + p_part_recon_t_un_dist + p_un_recon_t_un_dist)/un_recon_dist_true, '%')
    print('=================================================================')
    print('=================================================================')
    print(100*good_W_had/(bad_W_had + good_W_had), ' good_W_had')
    print(100*good_W_lep/(bad_W_lep + good_W_lep), ' good_W_lep')
    print(100*good_b_had/(bad_b_had + good_b_had), ' good_b_had')
    print(100*good_b_lep/(bad_b_lep + good_b_lep), ' good_b_lep')

    c1 = TCanvas()
    W_lep_d.Draw()
    c1.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_recon.png')
    c1.Close()

    c2 = TCanvas()
    W_had_d.Draw()
    c2.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_recon.png')
    c2.Close()

    c3 = TCanvas()
    b_lep_d.Draw()
    c3.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_recon.png')
    c3.Close()

    c4 = TCanvas()
    b_had_d.Draw()
    c4.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_recon.png')
    c4.Close()

    c5 = TCanvas()
    t_lep_d.Draw()
    c5.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_recon.png')
    c5.Close()

    c6 = TCanvas()
    t_had_d.Draw()
    c6.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_recon.png')
    c6.Close()

    c7 = TCanvas()
    W_lep_d_part.Draw()
    c7.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_part_recon.png')
    c7.Close()

    c8 = TCanvas()
    W_had_d_part.Draw()
    c8.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_part_recon.png')
    c8.Close()

    c9 = TCanvas()
    b_lep_d_part.Draw()
    c9.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_part_recon.png')
    c9.Close()

    c10 = TCanvas()
    b_had_d_part.Draw()
    c10.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_part_recon.png')
    c10.Close()

    c11 = TCanvas()
    t_lep_d_part.Draw()
    c11.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_part_recon.png')
    c11.Close()

    c12 = TCanvas()
    t_had_d_part.Draw()
    c12.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_part_recon.png')
    c12.Close()

    c13 = TCanvas()
    h_b_lep_true.Draw()
    c13.SaveAs(outputdir + subdir + 'leptonic_b_true_dist.png')
    c13.Close()

    c14 = TCanvas()
    h_b_had_true.Draw()
    c14.SaveAs(outputdir + subdir + 'hadronic_b_true_dist.png')
    c14.Close()

    c15 = TCanvas()
    h_t_lep_true.Draw()
    c15.SaveAs(outputdir + subdir + 'leptonic_t_true_dist.png')
    c15.Close()

    c16 = TCanvas()
    h_t_had_true.Draw()
    c16.SaveAs(outputdir + subdir + 'hadronic_t_true_dist.png')
    c16.Close()

    c17 = TCanvas()
    h_W_had_true.Draw()
    c17.SaveAs(outputdir + subdir + 'hadronic_W_true_dist.png')
    c17.Close()

    c18 = TCanvas()
    h_pT_W_had_true.Draw()
    c18.SaveAs(outputdir + subdir + 'hadronic_W_true_pT_dist.png')
    c18.Close()

    c19 = TCanvas()
    h_W_lep_true.Draw()
    c19.SaveAs(outputdir + subdir + 'leptonic_W_true_dist.png')
    c19.Close()

    #c20 = TCanvas()
    #h_pT_W_lep_true.Draw()
    #c20.SaveAs(outputdir + subdir + 'leptonic_W_true_pT_dist.png')
    #c20.Close()

    c21 = TCanvas()
    W_lep_d_un.Draw()
    c21.SaveAs(outputdir + subdir + 'leptonic_W_dist_pred_v_true_un_recon.png')
    c21.Close()

    c22 = TCanvas()
    W_had_d_un.Draw()
    c22.SaveAs(outputdir + subdir + 'hadronic_W_dist_pred_v_true_un_recon.png')
    c22.Close()

    c23 = TCanvas()
    b_lep_d_un.Draw()
    c23.SaveAs(outputdir + subdir + 'leptonic_b_dist_pred_v_true_un_recon.png')
    c23.Close()

    c24 = TCanvas()
    b_had_d_un.Draw()
    c24.SaveAs(outputdir + subdir + 'hadronic_b_dist_pred_v_true_un_recon.png')
    c24.Close()

    c25 = TCanvas()
    t_lep_d_un.Draw()
    c25.SaveAs(outputdir + subdir + 'leptonic_t_dist_pred_v_true_un_recon.png')
    c25.Close()

    c26 = TCanvas()
    t_had_d_un.Draw()
    c26.SaveAs(outputdir + subdir + 'hadronic_t_dist_pred_v_true_un_recon.png')
    c26.Close()

    c27 = TCanvas()
    W_lep_met_d.Draw()
    c27.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_observed.png')
    c27.Close()

    c28 = TCanvas()
    W_lep_met_d_t.Draw()
    c28.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_true.png')
    c28.Close()

    c29 = TCanvas()
    W_lep_met_d_p.Draw()
    c29.SaveAs(outputdir + subdir + 'leptonic_W_transverse_mass_predicted.png')
    c29.Close()

def make_correlations():
    c_W_had_true = TH2F( "c_W_had_true", "c_W_had_true", 50, 0., 1000., 50, 0., 1000. )
    c_W_had_fitted = TH2F( "c_W_had_fitted", "c_W_had_fitted", 50, 0., 1000., 50, 0., 1000. )
    
    jets = []

    for i in range(n_events):
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W )
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W)
    
        jet_mu_vect = MakeP4(jet_mu[i], jet_mu[i][4])
        jet_1_vect = MakeP4(jet_1[i], jet_1[i][4])
        jet_2_vect = MakeP4(jet_2[i], jet_2[i][4])
        jet_3_vect = MakeP4(jet_3[i], jet_3[i][4])
        jet_4_vect = MakeP4(jet_4[i], jet_4[i][4])
        jet_5_vect = MakeP4(jet_5[i], jet_5[i][4])
        
        jets.append([])
        jets[i].append(jet_1_vect)
        jets[i].append(jet_2_vect)
        jets[i].append(jet_3_vect)
        jets[i].append(jet_4_vect)
        jets[i].append(jet_5_vect)
        
        W_had_dist_true = 10000000
        p_had_true_total = 0
        for k in range(len(jets[i])):
            for j in range(i, len(jets[i])):
                #px_total = jets[i][k].Px() + jets[i][j].Px()
                #py_total = jets[i][k].Py() + jets[i][j].Py()
                #pz_total = jets[i][k].Pz() + jets[i][j].Pz()
                #d = np.sqrt((W_had_true.Px()-px_total)**2+(W_had_true.Py()-py_total)**2+(W_had_true.Pz()-pz_total)**2)
                sum_vect = jets[i][k] + jets[i][j]
                dphi = min(np.abs(W_had_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-sum_vect.Phi()))
                d = np.sqrt(dphi**2+(W_had_true.Eta()-sum_vect.Eta())**2)
                if d < W_had_dist_true:
                    W_had_dist_true = d
                    p_had_true_total = sum_vect.Px()**2 + sum_vect.Py()**2 + sum_vect.Pz()**2
        c_W_had_true.Fill(W_had_true.Px()**2+W_had_true.Py()**2+W_had_true.Pz()**2, p_had_true_total)
    
        W_had_dist_fitted = 10000000
        p_had_fitted_total = 0
        for k in range(len(jets[i])):
            for j in range(i, len(jets[i])):
                #px_total = jets[i][k].Px() + jets[i][j].Px()
                #py_total = jets[i][k].Py() + jets[i][j].Py()
                #pz_total = jets[i][k].Pz() + jets[i][j].Pz()
                #d = np.sqrt((W_had_fitted.Px()-px_total)**2+(W_had_fitted.Py()-py_total)**2+(W_had_fitted.Pz()-pz_total)**2)
                sum_vect = jets[i][k] + jets[i][j]
                dphi = min(np.abs(W_had_fitted.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_fitted.Phi()-sum_vect.Phi()))
                d = np.sqrt(dphi**2+(W_had_fitted.Eta()-sum_vect.Eta())**2)
                if d < W_had_dist_fitted:
                    W_had_dist_fitted = d
                    p_had_fitted_total = sum_vect.Px()**2 + sum_vect.Py()**2 + sum_vect.Pz()**2
        c_W_had_fitted.Fill(W_had_fitted.Px()**2+W_had_fitted.Py()**2+W_had_fitted.Pz()**2, p_had_fitted_total)
        
    c11 = TCanvas()
    c_W_had_true.Draw()
    corr_had_true = c_W_had_true.GetCorrelationFactor()
    l3 = TLatex()
    l3.SetNDC()
    l3.SetTextFont(42)
    l3.SetTextColor(kBlack)
    l3.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr_had_true)
    c11.SaveAs(outputdir + subdir + 'hadronic_W_true_corr.png')
    c11.Close()
    
    c12 = TCanvas()
    c_W_had_fitted.Draw()
    corr_had_fitted = c_W_had_fitted.GetCorrelationFactor()
    l4 = TLatex()
    l4.SetNDC()
    l4.SetTextFont(42)
    l4.SetTextColor(kBlack)
    l4.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr_had_fitted)
    c11.SaveAs(outputdir + subdir + '/hadronic_W_fitted_corr.png')
    c11.Close()
    
if __name__ == "__main__":
    try:
        os.mkdir('{}/closejets_img'.format(outputdir))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()
    #make_correlations()
>>>>>>> 1bf561fb4cd46f312deecb7db9d058ad6bfa0f45
