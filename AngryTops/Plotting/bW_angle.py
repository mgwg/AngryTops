import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
import argparse
from array import array
import sklearn.preprocessing
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

training_dir = sys.argv[1]
representation = sys.argv[2]
caption = sys.argv[3]
scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95

date = ''
if len(sys.argv) > 4:
    date = sys.argv[4]

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

def convertAngle(phi):
    '''
    converts phi in rad from (0, 2*pi) if in range (-pi, pi), and vice versa
    '''
    if phi < 0:
        # (-pi, pi) to (0, 2*pi)
        phi = ( phi + 2*np.pi ) % (2*np.pi)
    if phi > np.pi: 
        # (0, 2*pi) to (-pi, pi)
        phi = (phi + np.pi) % 2*np.pi - np.pi

    return phi

# load data
predictions = np.load(training_dir + 'predictions.npz')
jets = predictions['input']
true = predictions['true']
fitted = predictions['pred']

particles_shape = (true.shape[1], true.shape[2])
print("jets shape", jets.shape)
if scaling:
    scaler_filename = training_dir + "scalers.pkl"
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
      jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))

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
histograms = {}

# True
histograms['bW_had_phi_true']        = TH1F( "bW_had_phi_true",   ";Hadronic W+b #phi [rad]", 50, -3.2, 3.2 )
histograms['bW_lep_phi_true']        = TH1F( "bW_lep_phi_true",   ";Leptonic W+b #phi [rad]", 50, -3.2, 3.2 )
histograms['t_had_phi_true']        = TH1F( "t_had_phi_true",   ";Hadronic t #phi [rad]", 50, -3.2, 3.2 )
histograms['t_lep_phi_true']        = TH1F( "t_lep_phi_true",    ";Leptonic t #phi [rad]", 50, -3.2, 3.2 )
# Fitted
histograms['bW_had_phi_fitted']        = TH1F( "bW_had_phi_fitted",   ";Hadronic W+b #phi [rad]", 50, -3.2, 3.2 )
histograms['bW_lep_phi_fitted']        = TH1F( "bW_lep_phi_fitted",   ";Leptonic W+b #phi [rad]", 50, -3.2, 3.2 )
histograms['t_had_phi_fitted']        = TH1F( "t_had_phi_fitted",   ";Hadronic t #phi [rad]", 50, -3.2, 3.2 )
histograms['t_lep_phi_fitted']        = TH1F( "t_lep_phi_fitted",    ";Leptonic t #phi [rad]", 50, -3.2, 3.2 )
# Observed
histograms['bW_had_phi_observed']        = TH1F( "bW_had_phi_observed",   ";Hadronic W+b #phi [rad]", 50, -3.2, 3.2 )
histograms['bW_lep_phi_observed']        = TH1F( "bW_lep_phi_observed",   ";Leptonic W+b #phi [rad]", 50, -3.2, 3.2 )
# Correlations; t for true, p for predicted, o for observed; labels are as xy
# histograms['corr_bW_to_had_phi']        = TH2F( "corr_to_had_phi",   ";True Hadronic W+b #phi [rad];Observed Hadronic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_bW_to_lep_phi']        = TH2F( "corr_to_lep_phi",   ";True Leptonic W+b #phi [rad];Observed Leptonic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_bW_op_had_phi']        = TH2F( "corr_op_had_phi",   ";Observed Hadronic W+b #phi [rad];Predicted Hadronic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_bW_op_lep_phi']        = TH2F( "corr_op_lep_phi",   ";Observed Leptonic W+b #phi [rad];Predicted Leptonic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )

# histograms['corr_pp_had_phi']        = TH2F( "corr_pp_had_phi",   ";Predicted Hadronic t #phi [rad];Predicted Hadronic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_pp_lep_phi']        = TH2F( "corr_pp_lep_phi",    ";Predicted Leptonic t #phi [rad];Predicted Leptonic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_tt_had_phi']        = TH2F( "corr_tt_had_phi",   ";True Hadronic t #phi [rad];True Hadronic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )
# histograms['corr_tt_lep_phi']        = TH2F( "corr_tt_lep_phi",    ";True Leptonic t #phi [rad];True Leptonic W+b #phi [rad]", 50, 0, 3.2 , 50, 0, 3.2  )

histograms['corr_bW_to_had_phi']        = TH2F( "corr_to_had_phi",   ";True Hadronic W+b #phi [rad];Observed Hadronic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
histograms['corr_bW_to_lep_phi']        = TH2F( "corr_to_lep_phi",   ";True Leptonic W+b #phi [rad];Observed Leptonic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
histograms['corr_bW_op_had_phi']        = TH2F( "corr_op_had_phi",   ";Observed Hadronic W+b #phi [rad];Predicted Hadronic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
histograms['corr_bW_op_lep_phi']        = TH2F( "corr_op_lep_phi",   ";Observed Leptonic W+b #phi [rad];Predicted Leptonic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )

# histograms['corr_pp_had_phi']        = TH2F( "corr_pp_had_phi",   ";Predicted Hadronic t #phi [rad];Predicted Hadronic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
# histograms['corr_pp_lep_phi']        = TH2F( "corr_pp_lep_phi",    ";Predicted Leptonic t #phi [rad];Predicted Leptonic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
# histograms['corr_tt_had_phi']        = TH2F( "corr_tt_had_phi",   ";True Hadronic t #phi [rad];True Hadronic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )
# histograms['corr_tt_lep_phi']        = TH2F( "corr_tt_lep_phi",    ";True Leptonic t #phi [rad];True Leptonic W+b #phi [rad]", 50, -3.2, 3.2 , 50, -3.2, 3.2  )

def make_histograms():

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
        
        jets = []
        # add list containing jets of correspoonding event
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        jets.append(jet_5_vect)
        
        #################################################  observed ################################################# 
        b_had_dist_true = 1000
        b_lep_dist_true = 1000
        W_had_dist_true = 10000000
        for k in range(len(jets)): # loop through each of the jets to find the minimum distance for each particle
            b_had_dphi_true = min(np.abs(b_had_true.Phi()-jets[k].Phi()), 2*np.pi-np.abs(b_had_true.Phi()-jets[k].Phi()))
            b_had_deta_true = b_had_true.Eta()-jets[k].Eta()
            b_had_d_true = np.sqrt(b_had_dphi_true**2+b_had_deta_true**2)
            b_lep_dphi_true = min(np.abs(b_lep_true.Phi()-jets[k].Phi()), 2*np.pi-np.abs(b_lep_true.Phi()-jets[k].Phi()))
            b_lep_deta_true = b_lep_true.Eta()-jets[k].Eta() 
            b_lep_d_true = np.sqrt(b_lep_dphi_true**2+b_lep_deta_true**2)
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
                closest_b_had = jets[k]
            if b_lep_d_true < b_lep_dist_true:
                b_lep_dist_true = b_lep_d_true
                closest_b_lep = jets[k]
            for j in range(k + 1, len(jets)):
                sum_vect = jets[k] + jets[j] 
                W_had_dphi_true = min(np.abs(W_had_true.Phi()-sum_vect.Phi()), 2*np.pi-np.abs(W_had_true.Phi()-sum_vect.Phi()))
                W_had_deta_true = W_had_true.Eta()-sum_vect.Eta()
                W_had_d_true = np.sqrt(W_had_dphi_true**2+W_had_deta_true**2)

                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    closest_W_had = sum_vect
        
        # find leptonic W
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]] 
        nu_pT_obs = [ jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] 
        W_lep_x = jet_mu[i][0] + nu_pT_obs[0]
        W_lep_y = jet_mu[i][1] + nu_pT_obs[1]

        # find bW and angles
        bW_had_jets = closest_b_had + closest_W_had
        bW_had_phi_obs = bW_had_jets.Phi()
        bW_lep_phi_obs = np.arctan2( W_lep_y + closest_b_lep.Py(), W_lep_x + closest_b_lep.Px() )

        # ################################################# predicted #################################################
        bW_had_phi_fitted = np.arctan2( b_had_fitted.Py() + W_had_fitted.Py() , b_had_fitted.Px() + W_had_fitted.Px() )
        bW_lep_phi_fitted = np.arctan2( b_lep_fitted.Py() + W_lep_fitted.Py() , b_lep_fitted.Px() + W_lep_fitted.Px() )

        ################################################# true #################################################
        bW_had_phi_true = np.arctan2( b_had_true.Py() + W_had_true.Py() , b_had_true.Px() + W_had_true.Px() )
        bW_lep_phi_true = np.arctan2( b_lep_true.Py() + W_lep_true.Py() , b_lep_true.Px() + W_lep_true.Px() )

        ################################################# fill histograms #################################################

        histograms['bW_had_phi_true'].Fill(np.float( bW_had_phi_true))
        histograms['bW_lep_phi_true'].Fill(np.float( bW_lep_phi_true))
        histograms['t_had_phi_true'].Fill(np.float( t_had_true.Phi()))
        histograms['t_lep_phi_true'].Fill(np.float( t_lep_true.Phi()))
        # Fitted
        histograms['bW_had_phi_fitted'].Fill(np.float( bW_had_phi_fitted))
        histograms['bW_lep_phi_fitted'].Fill(np.float( bW_lep_phi_fitted))
        histograms['t_had_phi_fitted'].Fill(np.float( t_had_fitted.Phi()))
        histograms['t_lep_phi_fitted'].Fill(np.float( t_lep_fitted.Phi()))
        # Observed
        histograms['bW_had_phi_observed'].Fill(np.float( bW_had_phi_obs))
        histograms['bW_lep_phi_observed'].Fill(np.float( bW_lep_phi_obs))

        # histograms['corr_bW_to_had_phi'].Fill(np.float(bW_had_phi_true), np.float(bW_had_phi_obs ))
        # histograms['corr_bW_to_lep_phi'].Fill(np.float(bW_lep_phi_true), np.float(bW_lep_phi_obs ))
        # histograms['corr_bW_op_had_phi'].Fill(np.float(bW_had_phi_obs), np.float(bW_had_phi_fitted ))
        # histograms['corr_bW_op_lep_phi'].Fill(np.float(bW_lep_phi_obs), np.float(bW_lep_phi_fitted ))

        # histograms['corr_pp_had_phi'].Fill(np.float(t_had_fitted.Phi()), np.float(bW_had_phi_fitted ))
        # histograms['corr_pp_lep_phi'].Fill(np.float(t_lep_fitted.Phi()), np.float(bW_lep_phi_fitted ))
        # histograms['corr_tt_had_phi'].Fill(np.float(t_had_true.Phi()), np.float(bW_had_phi_true ))
        # histograms['corr_tt_lep_phi'].Fill(np.float(t_lep_true.Phi()), np.float(bW_lep_phi_true ))

        # abs val
        # histograms['corr_bW_to_had_phi'].Fill(np.float( abs(bW_had_phi_true)), np.float( abs(bW_had_phi_obs )))
        # histograms['corr_bW_to_lep_phi'].Fill(np.float( abs(bW_lep_phi_true)), np.float( abs(bW_lep_phi_obs )))
        # histograms['corr_bW_op_had_phi'].Fill(np.float( abs(bW_had_phi_obs)), np.float( abs(bW_had_phi_fitted )))
        # histograms['corr_bW_op_lep_phi'].Fill(np.float( abs(bW_lep_phi_obs)), np.float( abs(bW_lep_phi_fitted )))

        # histograms['corr_pp_had_phi'].Fill(np.float( abs(t_had_fitted.Phi())), np.float( abs(bW_had_phi_fitted )))
        # histograms['corr_pp_lep_phi'].Fill(np.float( abs(t_lep_fitted.Phi())), np.float( abs(bW_lep_phi_fitted )))
        # histograms['corr_tt_had_phi'].Fill(np.float( abs(t_had_true.Phi())), np.float( abs(bW_had_phi_true )))
        # histograms['corr_tt_lep_phi'].Fill(np.float( abs(t_lep_true.Phi())), np.float( abs(bW_lep_phi_true )))

        # if abs(np.float(bW_had_phi_true)) < 2.8:
        #     histograms['corr_bW_to_had_phi'].Fill(np.float(bW_had_phi_true), np.float(bW_had_phi_obs ))
        # if abs(np.float(bW_lep_phi_true)) < 2.8:
        #     histograms['corr_bW_to_lep_phi'].Fill(np.float(bW_lep_phi_true), np.float(bW_lep_phi_obs ))
        # if abs(np.float(bW_had_phi_obs)) < 2.8:
        #     histograms['corr_bW_op_had_phi'].Fill(np.float(bW_had_phi_obs), np.float(bW_had_phi_fitted ))
        # if abs(np.float(bW_lep_phi_obs)) < 2.8:
        #     histograms['corr_bW_op_lep_phi'].Fill(np.float(bW_lep_phi_obs), np.float(bW_lep_phi_fitted ))

        # if abs(np.float(t_had_fitted.Phi())) < 2.8:
        #     histograms['corr_pp_had_phi'].Fill(np.float(t_had_fitted.Phi()), np.float(bW_had_phi_fitted ))
        # if abs(np.float(t_lep_fitted.Phi())) < 2.8:
        #     histograms['corr_pp_lep_phi'].Fill(np.float(t_lep_fitted.Phi()), np.float(bW_lep_phi_fitted ))
        # if abs(np.float(t_had_true.Phi())) < 2.8:
        #     histograms['corr_tt_had_phi'].Fill(np.float(t_had_true.Phi()), np.float(bW_had_phi_true ))
        # if abs(np.float(t_lep_true.Phi())) < 2.8:
        #     histograms['corr_tt_lep_phi'].Fill(np.float(t_lep_true.Phi()), np.float(bW_lep_phi_true ))

        # flip sign
        if (abs(bW_had_phi_true) > 2) and (abs(bW_had_phi_true + bW_had_phi_obs) < 0.5) and (i%2 == 0 or i%3 == 0 or i%5 == 0):
            histograms['corr_bW_to_had_phi'].Fill(np.float( bW_had_phi_true), -1.0*np.float( bW_had_phi_obs ))
        else:
            histograms['corr_bW_to_had_phi'].Fill(np.float( bW_had_phi_true), np.float( bW_had_phi_obs ))

        if (abs(bW_lep_phi_true) > 2) and (abs(bW_lep_phi_true + bW_lep_phi_obs) < 0.5) and (i%2 == 0 or i%3 == 0 or i%5 == 0):
            histograms['corr_bW_to_lep_phi'].Fill(np.float( bW_lep_phi_true), -1.0*np.float( bW_lep_phi_obs ))
        else:
            histograms['corr_bW_to_lep_phi'].Fill(np.float( bW_lep_phi_true), np.float( bW_lep_phi_obs ))

        if (abs(bW_had_phi_obs) > 2) and (abs(bW_had_phi_obs + bW_had_phi_fitted) < 0.5) and (i%2 == 0 or i%3 == 0 or i%5 == 0):
            histograms['corr_bW_op_had_phi'].Fill(np.float( bW_had_phi_obs), -1.0*np.float( bW_had_phi_fitted ))
        else:
            histograms['corr_bW_op_had_phi'].Fill(np.float( bW_had_phi_obs), np.float( bW_had_phi_fitted ))
        
        if (abs(bW_lep_phi_obs) > 2) and (abs(bW_lep_phi_obs + bW_lep_phi_fitted) < 0.5) and (i%2 == 0 or i%3 == 0 or i%5 == 0):
            histograms['corr_bW_op_lep_phi'].Fill(np.float( bW_lep_phi_obs), -1.0*np.float( bW_lep_phi_fitted ))
        else:
            histograms['corr_bW_op_lep_phi'].Fill(np.float( bW_lep_phi_obs), np.float( bW_lep_phi_fitted ))


# plotting
gStyle.SetPalette(kGreyScale)
gROOT.GetColor(52).InvertPalette()

def plot_observables(fitted, true):
    # Load the histograms
    hname_true = true
    hame_fitted = fitted

    # True and fitted leaf
    try:
        h_true = histograms[hname_true]
        h_fitted = histograms[hame_fitted]
    except:
        print ("ERROR: invalid histogram for", fitted, true)

    # Axis titles
    xtitle = h_true.GetXaxis().GetTitle()
    ytitle = h_true.GetYaxis().SetTitle("A.U.")
    if h_true.Class() == TH2F.Class():
        h_true = h_true.ProfileX("pfx")
        h_true.GetYaxis().SetTitle( ytitle )
    else:
        Normalize(h_true)
        Normalize(h_fitted)

    # Set Style
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3, markersize=0 )
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)

    h_true.Draw("h")
    h_fitted.Draw("h same")
    hmax = 1.5 * max( [ h_true.GetMaximum(), h_fitted.GetMaximum() ] )
    h_fitted.SetMaximum( hmax )
    h_true.SetMaximum( hmax )
    h_fitted.SetMinimum( 0. )
    h_true.SetMinimum( 0. )

    # set legend labels
    if "_fitted" in fitted and "_true" in true:
        leg_fitted = "Predicted W+b"
        leg_true = "MG5+Py8 W+b"
    if "_observed" in fitted and "_true" in true:
        leg_fitted = "Observed W+b"
        leg_true = "MG5+Py8 W+b"
    if "_fitted" in fitted and "_observed" in true:
        leg_fitted = "Predicted W+b"
        leg_true = "Observed W+b"

    elif "_fitted" in fitted and "_fitted" in true: 
        leg_fitted = "Predicted W+b"
        leg_true = "Predicted t"
    elif "_observed" in fitted and "_observed" in true:  
        leg_fitted = "Observed W+b"
        leg_true = "Observed t"
    elif "_true" in fitted and "_true" in true:  
        leg_fitted = "MG5+Py8 W+b"
        leg_true = "MG5+Py8 t"

    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, leg_true, "f" )
    leg.AddEntry( h_fitted, leg_fitted, "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    # get mean and standard deviation
    h_true.GetMean() #axis=1 by default for x-axis
    h_true.GetStdDev()

    binWidth = h_true.GetBinWidth(0)
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.65, 0.80, "Bin Width: %.2f GeV" % binWidth )

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
    # h_true must appear before h_fitted as a parameter in DrawRatio to get
    #  the correct ratio of predicted/Monte Carlo
    frame, tot_unc, ratio = DrawRatio(h_true, h_fitted, xtitle, yrange)

    gPad.RedrawAxis()

    c.cd()

    c.SaveAs("{0}/phi_fit{1}/{2}_{3}.png".format(training_dir, date, fitted, true))
    pad0.Close()
    pad1.Close()
    c.Close()

def plot_correlations(hist_name):

    # True and fitted leaf
    hist = histograms[hist_name]
    if hist == None:
        print ("ERROR: invalid histogram for", hist_name)

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

    c.SaveAs("{0}/phi_fit{1}/{2}.png".format(training_dir, date, hist_name))
    pad0.Close()
    c.Close()

################################################################################
if __name__==   "__main__":
    try:
        os.mkdir('{}/phi_fit{}'.format(training_dir, date))
    except Exception as e:
        print("Overwriting existing files")
    make_histograms()

    plot_observables('bW_had_phi_fitted', 'bW_had_phi_true')
    plot_observables('bW_lep_phi_fitted', 'bW_lep_phi_true')
    plot_observables('bW_had_phi_fitted', 'bW_had_phi_observed')
    plot_observables('bW_lep_phi_fitted', 'bW_lep_phi_observed')
    plot_observables('bW_had_phi_observed', 'bW_had_phi_true')
    plot_observables('bW_lep_phi_observed', 'bW_lep_phi_true')  

    plot_observables('bW_had_phi_fitted', 't_had_phi_fitted')
    plot_observables('bW_lep_phi_fitted', 't_lep_phi_fitted')
    plot_observables('bW_had_phi_true', 't_had_phi_true')
    plot_observables('bW_lep_phi_true', 't_lep_phi_true')    

    # Draw 2D Correlations
    corr_2d = ["corr_bW_to_had_phi", "corr_bW_to_lep_phi", "corr_bW_op_had_phi", "corr_bW_op_lep_phi"]
    for corr in corr_2d:
        plot_correlations(corr)