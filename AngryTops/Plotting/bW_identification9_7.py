import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
#import ROOT
import pickle

representation = sys.argv[2]
outputdir = sys.argv[1]
subdir = '/closejets_img_test/'
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95

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

def compare_true_pred(parton, truth, fitted, dist_lim, phi_lim, R):
    '''
    (str, TLorentzVector(), TLorentzVector(), float, float) -> Bool
    str should be one of b_had, b_lep, W_had, W_lep
    dist_lim and phi_lim are the R and phi tolerance limits
    '''
    phi_recon = False
    eta_recon = False
    R_recon = False

    # only compare Phi distances for Leptonic W
    if parton == 'W_lep':
        if ( np.abs( truth.Phi() - fitted.Phi() ) <= 37.0/50.0 ):
            R_recon = True
        elif R <= (dist_lim + 0.2):
            R_recon = True

    elif parton in ['b_lep', 'b_had', 'W_had']:
        # compare phi distances
        if ( np.abs( truth.Phi() - fitted.Phi() ) <= phi_lim ):
            phi_recon = True

        # compare eta distances
        if parton == 'b_lep':
            if ( (truth.Eta() - 4.0/5.0) <= fitted.Eta() ) and ( fitted.Eta() <= (truth.Eta()*15.0/14.0 + 13.0/14.0) ):
                if ( np.abs(truth.Eta()) <= 1.8 ) and ( fitted.Eta() <= 2.0 ) and ( -1.8 <= fitted.Eta() ):
                    eta_recon = True
        elif parton == 'b_had':
            if ( np.abs( truth.Eta()*5.0/4.0 - fitted.Eta() ) <= 19.0/20.0 ):
                if ( np.abs(truth.Eta()) <= 2.2 ) and ( np.abs(fitted.Eta()) <= 2.4 ):
                    eta_recon = True  
        elif parton == 'W_had':
            if ( np.abs(truth.Eta() - fitted.Eta()) <= 4.0/5.0):
                if ( np.abs(truth.Eta()) <= 1.8 ) and ( np.abs(fitted.Eta()) <= 1.8 ):
                    eta_recon = True             

        # use eta and phi to determine overall reconstructability
        # R_recon is False if phi and eta are both False, or if it is greater than the limit + 0.2
        if (phi_recon == True) and (eta_recon == True):
            R_recon = True
        elif R <= (dist_lim + 0.2): #checking condition is to calculate R, and to see if it fits within a constant, hard-coded R value, as determined from the TvO distributions created
                R_recon = True

    return R_recon

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

n_events = true.shape[0]

# make histograms to be fillled
hists = {}

hists['leptonic_W_transverse_mass_observed'] = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Observed", 50, 0, 250)#120)
hists['leptonic_W_transverse_mass_observed'].SetTitle("W Leptonic Transverse Mass, Observed;Leptonic (GeV);A.U.")
hists['leptonic_W_transverse_mass_true'] = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Truth", 50, 0, 250) #0, 120)
hists['leptonic_W_transverse_mass_true'].SetTitle("W Leptonic Transverse Mass, Truth;Leptonic (GeV);A.U.")
hists['leptonic_W_transverse_mass_predicted'] = TH1F("W_lep_met_d","W Leptonic Transverse Mass, Predicted", 50, 0, 250) #0, 120)
hists['leptonic_W_transverse_mass_predicted'].SetTitle("W Leptonic Transverse Mass, Predicted;Leptonic (GeV);A.U.")

hists['leptonic_W_true_dist'] = TH1F("h_W_lep_true","W Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_W_true_dist'].SetTitle("W Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")
hists['leptonic_W_dist_pred_v_true_recon'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 3)
hists['leptonic_W_dist_pred_v_true_recon'].SetTitle("W Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (radians);A.U.")
hists['leptonic_W_dist_pred_v_true_part_recon'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['leptonic_W_dist_pred_v_true_part_recon'].SetTitle("W Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (radians);A.U.")
hists['leptonic_W_dist_pred_v_true_un_recon'] = TH1F("W_lep_d","W Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['leptonic_W_dist_pred_v_true_un_recon'].SetTitle("W Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (radians);A.U.")

hists['hadronic_W_true_pT_dist'] = TH1F("h_pT_W_had_true","W Hadronic p_T, True vs Observed", 50, -500, 500)
hists['hadronic_W_true_dist'] = TH1F("h_W_had_true","W Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_W_true_dist'].SetTitle("W Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")
hists['hadronic_W_dist_pred_v_true_recon'] = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Reconstructed", 50, 0, 3)
hists['hadronic_W_dist_pred_v_true_recon'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (radians);A.U.")
hists['hadronic_W_dist_pred_v_true_part_recon'] = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['hadronic_W_dist_pred_v_true_part_recon'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (radians);A.U.")
hists['hadronic_W_dist_pred_v_true_un_recon'] = TH1F("W_had_d","W Hadronic Distances Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['hadronic_W_dist_pred_v_true_un_recon'].SetTitle("W Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (radians);A.U.")

hists['leptonic_b_true_dist'] = TH1F("h_b_lep_true","b Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_b_true_dist'].SetTitle("b Leptonic #eta-#phi distances, True vs Observed;true leptonic (radians);A.U.")
hists['leptonic_b_dist_pred_v_true_recon'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 3)
hists['leptonic_b_dist_pred_v_true_recon'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
hists['leptonic_b_dist_pred_v_true_part_recon'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['leptonic_b_dist_pred_v_true_part_recon'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
hists['leptonic_b_dist_pred_v_true_un_recon'] = TH1F("b_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['leptonic_b_dist_pred_v_true_un_recon'].SetTitle("b Leptonic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")

hists['hadronic_b_true_dist'] = TH1F("h_b_had_true","b Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_b_true_dist'].SetTitle("b Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")
hists['hadronic_b_dist_pred_v_true_recon'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 3)
hists['hadronic_b_dist_pred_v_true_recon'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")
hists['hadronic_b_dist_pred_v_true_part_recon'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['hadronic_b_dist_pred_v_true_part_recon'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")
hists['hadronic_b_dist_pred_v_true_un_recon'] = TH1F("b_had_d","b Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['hadronic_b_dist_pred_v_true_un_recon'].SetTitle("b Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

hists['leptonic_t_true_dist'] = TH1F("h_t_lep_true","t Leptonic Distances, True vs Observed", 50, 0, 3)
hists['leptonic_t_true_dist'].SetTitle("t Leptonic #phi distances, True vs Observed;true leptonic (radians);A.U.")
hists['leptonic_t_dist_pred_v_true_recon'] = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Reconstructed", 50, 0, 3)
hists['leptonic_t_dist_pred_v_true_recon'].SetTitle("t Leptonic #phi distances, Predicted vs Truth Reconstructed;Leptonic (R);A.U.")
hists['leptonic_t_dist_pred_v_true_part_recon'] = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['leptonic_t_dist_pred_v_true_part_recon'].SetTitle("t Leptonic #phi distances, Predicted vs Truth Partially Reconstructed;Leptonic (R);A.U.")
hists['leptonic_t_dist_pred_v_true_un_recon'] = TH1F("t_lep_d","b Leptonic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['leptonic_t_dist_pred_v_true_un_recon'].SetTitle("t Leptonic #phi distances, Predicted vs Truth Not Reconstructed;Leptonic (R);A.U.")

hists['hadronic_t_true_dist'] = TH1F("h_t_had_true","t Hadronic Distances, True vs Observed", 50, 0, 3)
hists['hadronic_t_true_dist'].SetTitle("t Hadronic #eta-#phi distances, True vs Observed;true hadronic (radians);A.U.")
hists['hadronic_t_dist_pred_v_true_recon'] = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Reconstructed", 50, 0, 3)
hists['hadronic_t_dist_pred_v_true_recon'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Reconstructed;Hadronic (R);A.U.")
hists['hadronic_t_dist_pred_v_true_part_recon'] = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Partially Reconstructed", 50, 0, 3)
hists['hadronic_t_dist_pred_v_true_part_recon'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Partially Reconstructed;Hadronic (R);A.U.")
hists['hadronic_t_dist_pred_v_true_un_recon'] = TH1F("t_had_d","t Hadronic Distances, Predicted vs Truth Not Reconstructed", 50, 0, 3)
hists['hadronic_t_dist_pred_v_true_un_recon'].SetTitle("t Hadronic #eta-#phi distances, Predicted vs Truth Not Reconstructed;Hadronic (R);A.U.")

def make_histograms():

    jets = []

    # define tolerance limits
    b_lep_dist_t_lim = b_had_dist_t_lim = 0.39
    t_lep_dist_t_lim = t_had_dist_t_lim = 0.80
    W_lep_dist_t_lim = 0.82
    W_had_dist_t_lim = 1.82

    b_lep_dist_t_lim = 0.39
    b_had_dist_t_lim = 0.39
    t_lep_dist_t_lim = 0.80
    t_had_dist_t_lim = 0.80
    W_lep_dist_t_lim = 1.28
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

    for i in range(n_events): # loop through every event
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
        
        jets.append([]) # add list containing jets of correspoonding event
        jets[i].append(jet_1_vect)
        jets[i].append(jet_2_vect)
        jets[i].append(jet_3_vect)
        jets[i].append(jet_4_vect)
        jets[i].append(jet_5_vect)

        # met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(jet_mu[i][5])))
        # met_true = np.sqrt(2*W_lep_true.Pt()*W_lep_true.Et()*(1 - np.cos(W_lep_true.Phi())))
        # met_pred = np.sqrt(2*W_lep_fitted.Pt()*W_lep_fitted.Et()*(1 - np.cos(W_lep_fitted.Phi())))
        
        # Observed transverse mass distribution is square root of 2* Etmiss 
        #  * Transverse angle between daughter particles, assuming that they are massless.
        
        # First, find the transverse angle between the daughter particles, which is the angle 
        #  between the muon momentum vector and the missing Et vector. For the muon, we have 
        #  px and py, but for the missing Et, we have the vector in polar representation.
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]] # Observed transverse momentum of muon
        # Convert missing transverse energy to a momentum
        missing_px = jet_mu[i][4]*np.cos(jet_mu[i][5]) # x-component of missing momentum
        missing_py = jet_mu[i][4]*np.sin(jet_mu[i][5]) # y-component of missing momentum
        nu_pT_obs = [missing_px, missing_py] # Observed neutrino transverse momentum from missing energy.
        # Now, calculate the angle.
        obs_daughter_angle = np.arccos(np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs))
        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 
        # Pt^2 = Px^2 + Py^2
        met_true = W_lep_true.Mt() # np.sqrt(2*W_lep_true.Pt()*W_lep_true.Et()*(1 - np.cos(W_lep_true.Phi())))
        met_pred = W_lep_fitted.Mt() # np.sqrt(2*W_lep_fitted.Pt()*W_lep_fitted.Et()*(1 - np.cos(W_lep_fitted.Phi())))

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
        for k in range(len(jets[i])): # loop through each of the jets to find the minimum distance for each particle
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

                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    W_had_true_pT = W_had_true.Pt() - sum_vect.Pt()
        
        # Add muon transverse momentum components to missing momentum components
        lep_x = jet_mu[i][0] + missing_px
        lep_y = jet_mu[i][1] + missing_py
        # Calculate phi using definition in Kuunal's report 
        lep_phi = np.arctan2( lep_y, lep_x )
        # Calculate the distance between jets, if it is less than the the current minimum, update it.
        W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )

        corr_jets_dist = 0.
        corr_p_jets_dist = 0.

        if (b_lep_dist_true <= b_lep_dist_t_lim): # if minimum distance is less than the tolearance limits, everything is ok
            corr_jets_dist += 1
            good_b_lep += 1
        else:
            bad_b_lep += 1
        if (b_had_dist_true <= b_had_dist_t_lim):
            corr_jets_dist += 1
            good_b_had += 1
        else:
            bad_b_had += 1
        if (W_lep_dist_true <= W_lep_dist_t_lim): # mismatch between W_lep_dist_true and good_W_lep
            corr_jets_dist += 1
            good_W_lep += 1
        else:
            bad_W_lep += 1
        if (W_had_dist_true <= W_had_dist_t_lim):
            corr_jets_dist += 1
            good_W_had += 1
        else:
            bad_W_had += 1

        # fully reconstructable
        if corr_jets_dist == 4: 
            full_recon_dist_true = full_recon_dist_true + 1
            hists['leptonic_b_dist_pred_v_true_recon'].Fill(b_lep_R)
            hists['hadronic_b_dist_pred_v_true_recon'].Fill(b_had_R)
            hists['leptonic_t_dist_pred_v_true_recon'].Fill(t_lep_R)
            hists['hadronic_t_dist_pred_v_true_recon'].Fill(t_had_R)
            hists['leptonic_W_dist_pred_v_true_recon'].Fill(W_lep_R)
            hists['hadronic_W_dist_pred_v_true_recon'].Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (b_had_R_recon == True):
                corr_p_jets_dist += 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (W_had_R_recon == True):
                corr_p_jets_dist += 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_full_dist += 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_full_dist += 1
            else:
                p_un_recon_t_full_dist += 1
        # partially reconstructable
        elif corr_jets_dist == 3: 
            part_recon_dist_true = part_recon_dist_true + 1
            hists['leptonic_b_dist_pred_v_true_part_recon'].Fill(b_lep_R)
            hists['hadronic_b_dist_pred_v_true_part_recon'].Fill(b_had_R)
            hists['leptonic_t_dist_pred_v_true_part_recon'].Fill(t_lep_R)
            hists['hadronic_t_dist_pred_v_true_part_recon'].Fill(t_had_R)
            hists['leptonic_W_dist_pred_v_true_part_recon'].Fill(W_lep_R)
            hists['hadronic_W_dist_pred_v_true_part_recon'].Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (b_had_R_recon == True):
                corr_p_jets_dist += 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (W_had_R_recon == True):
                corr_p_jets_dist += 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_part_dist += 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_part_dist += 1
            else:
                p_un_recon_t_part_dist += 1
        # un-reconstructable
        else: 
            un_recon_dist_true += 1
            hists['leptonic_b_dist_pred_v_true_un_recon'].Fill(b_lep_R)
            hists['hadronic_b_dist_pred_v_true_un_recon'].Fill(b_had_R)
            hists['leptonic_t_dist_pred_v_true_un_recon'].Fill(t_lep_R)
            hists['hadronic_t_dist_pred_v_true_un_recon'].Fill(t_had_R)
            hists['leptonic_W_dist_pred_v_true_un_recon'].Fill(W_lep_R)
            hists['hadronic_W_dist_pred_v_true_un_recon'].Fill(W_had_R)
            if (b_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (b_had_R_recon == True):
                corr_p_jets_dist += 1
            if (W_lep_R_recon == True):
                corr_p_jets_dist += 1
            if (W_had_R_recon == True):
                corr_p_jets_dist += 1

            if corr_p_jets_dist == 4:
                p_full_recon_t_un_dist += 1
            elif corr_p_jets_dist == 3:
                p_part_recon_t_un_dist += 1
            else:
                p_un_recon_t_un_dist += 1

        hists['leptonic_b_true_dist'].Fill(np.float(b_lep_dist_true))
        hists['hadronic_b_true_dist'].Fill(np.float(b_had_dist_true))
        hists['leptonic_t_true_dist'].Fill(np.float(t_lep_dist_true))
        hists['hadronic_t_true_dist'].Fill(np.float(t_had_dist_true))
        hists['leptonic_W_true_dist'].Fill(np.float(W_lep_dist_true))
        hists['leptonic_W_transverse_mass_observed'].Fill(np.float(met_obs))
        hists['leptonic_W_transverse_mass_true'].Fill(np.float(met_true))
        hists['leptonic_W_transverse_mass_predicted'].Fill(np.float(met_pred))
        hists['hadronic_W_true_dist'].Fill(np.float(W_had_dist_true))
        hists['hadronic_W_true_pT_dist'].Fill(np.float(W_had_true_pT))

    print('good_W_had', good_W_had, 'bad_W_had', bad_W_had)
    print('good_W_lep', good_W_lep, 'bad_W_lep', bad_W_lep)
    print('good_b_had', good_b_had, 'bad_b_had', bad_b_had)
    print('good_b_lep', good_b_lep, 'bad_b_lep', bad_b_lep)

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
    
def plot_jets(key):
    c1 = TCanvas()
    hists[key].Draw()
    c1.SaveAs(outputdir + subdir + key +'.png')
    c1.Close()
    
if __name__ == "__main__":
    try:
        os.mkdir('{}/closejets_img_test'.format(outputdir))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()
    # for key in hists:
        # plot_jets(key)
