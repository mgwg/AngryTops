import os, sys, time
import argparse
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from array import array
import sklearn.preprocessing
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

outputdir = sys.argv[1]
representation = sys.argv[2]
date = ''
if len(sys.argv) > 3:
    date = sys.argv[3]
event_type = 0
if len(sys.argv) > 4:
    event_type = sys.argv[4]

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

# Function to make histograms
def make_histograms():

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

    bad_event = 0.
    drop = []

    for i in event_index: # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
        W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)
        b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
        b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
        t_had_true   = MakeP4( y_true_t_had[i], m_t , representation)
        t_lep_true   = MakeP4( y_true_t_lep[i], m_t , representation)

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

        ############################################## check whether each event passes cuts #################################################
        W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
        W_had_pT_cut = (W_had_true_obs_pT_diff >= W_had_pT_cutoff[0] and W_had_true_obs_pT_diff <= W_had_pT_cutoff[1])
        W_had_dist_cut = (W_had_dist_true <= W_had_dist_cutoff[1]) 
        # All W_had cuts must be satisfied simultaneously.
        good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)

        W_lep_ET_cut = (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_ET_diff <= W_lep_ET_cutoff[1])
        W_lep_dist_cut = (W_lep_dist_true <= W_lep_dist_cutoff[1]) 
        good_W_lep = (W_lep_ET_cut and W_lep_dist_cut)

        b_had_pT_cut = (b_had_true_obs_pT_diff >= b_had_pT_cutoff[0] and b_had_true_obs_pT_diff <= b_had_pT_cutoff[1])
        b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
        good_b_had = (b_had_pT_cut and b_had_dist_cut)

        b_lep_pT_cut = (b_lep_true_obs_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_obs_pT_diff <= b_lep_pT_cutoff[1])
        b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
        good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

        if not (good_b_had and good_b_lep and good_W_had and good_W_lep):
            # Good events must pass cuts on all partons.
            bad_event += 1.0
            drop.append(i)

    # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))

    print("Number of bad events")
    print('{}%, {} events'.format(100.*bad_event/n_events, bad_event))

    bad_jets = np.delete(predictions['input'], drop, 0)
    bad_true = np.delete(predictions['true'], drop, 0)
    bad_fitted = np.delete(predictions['pred'], drop, 0)
    bad_event = np.delete(predictions['events'], drop, 0)

    np.savez("{}/predictions_bad".format(outputdir), input=bad_jets,
            true=bad_true, pred=bad_fitted, events=bad_event)

# Run the two helper functions above   
if __name__ == "__main__":
    # make_histograms()
    ################################################################################
    np.set_printoptions(precision=3, suppress=True, linewidth=250)
    model_filename  = "{}/simple_model.h5".format(outputdir)
    # Load Predictions
    print("INFO: fitting ttbar decay chain...")
    predictions = np.load('{}/predictions_bad.npz'.format(outputdir))
    true = predictions['true']
    y_fitted = predictions['pred']
    event_info = predictions['events']
    # Keep track of the old shape: (# of test particles, # of features per
    # test particle, number of features for input)
    old_shape = (true.shape[1], true.shape[2])

    ################################################################################
    # UNDO NORMALIZATIONS
    # Import scalars
    if scaling:
        scaler_filename = "{}/scalers.pkl".format(outputdir)
        with open( scaler_filename, "rb" ) as file_scaler:
            jets_scalar = pickle.load(file_scaler)
            lep_scalar = pickle.load(file_scaler)
            output_scalar = pickle.load(file_scaler)
            # Rescale the truth array
            true = true.reshape(true.shape[0], true.shape[1]*true.shape[2])
            true = output_scalar.inverse_transform(true)
            true = true.reshape(true.shape[0], old_shape[0], old_shape[1])
            # Rescale the fitted array
            y_fitted = y_fitted.reshape(y_fitted.shape[0], y_fitted.shape[1]*y_fitted.shape[2])
            y_fitted = output_scalar.inverse_transform(y_fitted)
            y_fitted = y_fitted.reshape(y_fitted.shape[0], old_shape[0], old_shape[1])
    ################################################################################
    # Truth
    y_true_W_had = true[:,0,:]
    y_true_W_lep = true[:,1,:]
    y_true_b_had = true[:,2,:]
    y_true_b_lep = true[:,3,:]
    y_true_t_had = true[:,4,:]
    y_true_t_lep = true[:,5,:]

    # Fitted
    y_fitted_W_had = y_fitted[:,0,:]
    y_fitted_W_lep = y_fitted[:,1,:]
    y_fitted_b_had = y_fitted[:,2,:]
    y_fitted_b_lep = y_fitted[:,3,:]
    y_fitted_t_had = y_fitted[:,4,:]
    y_fitted_t_lep = y_fitted[:,5,:]

    # Event Info
    n_events = true.shape[0]
    w = 1
    print("Shape of tions: ", y_fitted.shape)
    print("INFO ...done")

    ################################################################################
    # CREATE OUTPUT TREE/FILE
    ofilename = "{}/fitted_bad.root".format(outputdir)
    ofile = TFile.Open( ofilename, "recreate" )
    ofile.cd()

    # Create output tree
    b_eventNumber = array( 'l', [ 0 ] )
    b_runNumber   = array( 'i', [ 0 ] )
    b_mcChannelNumber = array( 'i', [ 0 ] )
    b_weight_mc   = array( 'f', [ 0.] )

    b_W_had_px_true   = array( 'f', [ -1.] )
    b_W_had_py_true   = array( 'f', [ -1.] )
    b_W_had_pz_true   = array( 'f', [ -1.] )
    b_W_had_E_true    = array( 'f', [ -1.] )
    b_W_had_m_true    = array( 'f', [ -1.] )
    b_W_had_pt_true   = array( 'f', [ -1.] )
    b_W_had_y_true    = array( 'f', [ -1.] )
    b_W_had_phi_true  = array( 'f', [ -1.] )
    b_b_had_px_true   = array( 'f', [ -1.] )
    b_b_had_py_true   = array( 'f', [ -1.] )
    b_b_had_pz_true   = array( 'f', [ -1.] )
    b_b_had_E_true    = array( 'f', [ -1.] )
    b_b_had_m_true    = array( 'f', [ -1.] )
    b_b_had_pt_true   = array( 'f', [ -1.] )
    b_b_had_y_true    = array( 'f', [ -1.] )
    b_b_had_phi_true  = array( 'f', [ -1.] )
    b_t_had_px_true   = array( 'f', [ -1.] )
    b_t_had_py_true   = array( 'f', [ -1.] )
    b_t_had_pz_true   = array( 'f', [ -1.] )
    b_t_had_E_true    = array( 'f', [ -1.] )
    b_t_had_m_true    = array( 'f', [ -1.] )
    b_t_had_pt_true   = array( 'f', [ -1.] )
    b_t_had_y_true    = array( 'f', [ -1.] )
    b_t_had_phi_true  = array( 'f', [ -1.] )
    b_W_lep_px_true   = array( 'f', [ -1.] )
    b_W_lep_py_true   = array( 'f', [ -1.] )
    b_W_lep_pz_true   = array( 'f', [ -1.] )
    b_W_lep_E_true    = array( 'f', [ -1.] )
    b_W_lep_m_true    = array( 'f', [ -1.] )
    b_W_lep_pt_true   = array( 'f', [ -1.] )
    b_W_lep_y_true    = array( 'f', [ -1.] )
    b_W_lep_phi_true  = array( 'f', [ -1.] )
    b_b_lep_px_true   = array( 'f', [ -1.] )
    b_b_lep_py_true   = array( 'f', [ -1.] )
    b_b_lep_pz_true   = array( 'f', [ -1.] )
    b_b_lep_E_true    = array( 'f', [ -1.] )
    b_b_lep_m_true    = array( 'f', [ -1.] )
    b_b_lep_pt_true   = array( 'f', [ -1.] )
    b_b_lep_y_true    = array( 'f', [ -1.] )
    b_b_lep_phi_true  = array( 'f', [ -1.] )
    b_t_lep_px_true   = array( 'f', [ -1.] )
    b_t_lep_py_true   = array( 'f', [ -1.] )
    b_t_lep_pz_true   = array( 'f', [ -1.] )
    b_t_lep_E_true    = array( 'f', [ -1.] )
    b_t_lep_m_true    = array( 'f', [ -1.] )
    b_t_lep_pt_true   = array( 'f', [ -1.] )
    b_t_lep_y_true    = array( 'f', [ -1.] )
    b_t_lep_phi_true  = array( 'f', [ -1.] )

    b_W_had_px_fitted   = array( 'f', [ -1.] )
    b_W_had_py_fitted   = array( 'f', [ -1.] )
    b_W_had_pz_fitted   = array( 'f', [ -1.] )
    b_W_had_E_fitted    = array( 'f', [ -1.] )
    b_W_had_m_fitted    = array( 'f', [ -1.] )
    b_W_had_pt_fitted   = array( 'f', [ -1.] )
    b_W_had_y_fitted    = array( 'f', [ -1.] )
    b_W_had_phi_fitted  = array( 'f', [ -1.] )
    b_b_had_px_fitted   = array( 'f', [ -1.] )
    b_b_had_py_fitted   = array( 'f', [ -1.] )
    b_b_had_pz_fitted   = array( 'f', [ -1.] )
    b_b_had_E_fitted    = array( 'f', [ -1.] )
    b_b_had_m_fitted    = array( 'f', [ -1.] )
    b_b_had_pt_fitted   = array( 'f', [ -1.] )
    b_b_had_y_fitted    = array( 'f', [ -1.] )
    b_b_had_phi_fitted  = array( 'f', [ -1.] )
    b_t_had_px_fitted   = array( 'f', [ -1.] )
    b_t_had_py_fitted   = array( 'f', [ -1.] )
    b_t_had_pz_fitted   = array( 'f', [ -1.] )
    b_t_had_E_fitted    = array( 'f', [ -1.] )
    b_t_had_m_fitted    = array( 'f', [ -1.] )
    b_t_had_pt_fitted   = array( 'f', [ -1.] )
    b_t_had_y_fitted    = array( 'f', [ -1.] )
    b_t_had_phi_fitted  = array( 'f', [ -1.] )
    b_W_lep_px_fitted   = array( 'f', [ -1.] )
    b_W_lep_py_fitted   = array( 'f', [ -1.] )
    b_W_lep_pz_fitted   = array( 'f', [ -1.] )
    b_W_lep_E_fitted    = array( 'f', [ -1.] )
    b_W_lep_m_fitted    = array( 'f', [ -1.] )
    b_W_lep_pt_fitted   = array( 'f', [ -1.] )
    b_W_lep_y_fitted    = array( 'f', [ -1.] )
    b_W_lep_phi_fitted  = array( 'f', [ -1.] )
    b_b_lep_px_fitted   = array( 'f', [ -1.] )
    b_b_lep_py_fitted   = array( 'f', [ -1.] )
    b_b_lep_pz_fitted   = array( 'f', [ -1.] )
    b_b_lep_E_fitted    = array( 'f', [ -1.] )
    b_b_lep_m_fitted    = array( 'f', [ -1.] )
    b_b_lep_pt_fitted   = array( 'f', [ -1.] )
    b_b_lep_y_fitted    = array( 'f', [ -1.] )
    b_b_lep_phi_fitted  = array( 'f', [ -1.] )
    b_t_lep_px_fitted   = array( 'f', [ -1.] )
    b_t_lep_py_fitted   = array( 'f', [ -1.] )
    b_t_lep_pz_fitted   = array( 'f', [ -1.] )
    b_t_lep_E_fitted    = array( 'f', [ -1.] )
    b_t_lep_m_fitted    = array( 'f', [ -1.] )
    b_t_lep_pt_fitted   = array( 'f', [ -1.] )
    b_t_lep_y_fitted    = array( 'f', [ -1.] )
    b_t_lep_phi_fitted  = array( 'f', [ -1.] )

    tree = TTree( "nominal", "nominal" )
    tree.Branch( "eventNumber",     b_eventNumber,     'eventNumber/l' )
    tree.Branch( 'runNumber',       b_runNumber,       'runNumber/i' )
    tree.Branch( 'mcChannelNumber', b_mcChannelNumber, 'mcChannelNumber/i' )
    tree.Branch( 'weight_mc',       b_weight_mc,       'weight_mc/F' )

    tree.Branch( 'W_had_px_true',   b_W_had_px_true,   'W_had_px_true/F' )
    tree.Branch( 'W_had_py_true',   b_W_had_py_true,   'W_had_py_true/F' )
    tree.Branch( 'W_had_pz_true',   b_W_had_pz_true,   'W_had_pz_true/F' )
    tree.Branch( 'W_had_E_true',    b_W_had_E_true,    'W_had_E_true/F' )
    tree.Branch( 'W_had_m_true',    b_W_had_m_true,    'W_had_m_true/F' )
    tree.Branch( 'W_had_pt_true',   b_W_had_pt_true,   'W_had_pt_true/F' )
    tree.Branch( 'W_had_y_true',    b_W_had_y_true,    'W_had_y_true/F' )
    tree.Branch( 'W_had_phi_true',  b_W_had_phi_true,  'W_had_phi_true/F' )
    tree.Branch( 'b_had_px_true',   b_b_had_px_true,   'b_had_px_true/F' )
    tree.Branch( 'b_had_py_true',   b_b_had_py_true,   'b_had_py_true/F' )
    tree.Branch( 'b_had_pz_true',   b_b_had_pz_true,   'b_had_pz_true/F' )
    tree.Branch( 'b_had_E_true',    b_b_had_E_true,    'b_had_E_true/F' )
    tree.Branch( 'b_had_m_true',    b_b_had_m_true,    'b_had_m_true/F' )
    tree.Branch( 'b_had_pt_true',   b_b_had_pt_true,   'b_had_pt_true/F' )
    tree.Branch( 'b_had_y_true',    b_b_had_y_true,    'b_had_y_true/F' )
    tree.Branch( 'b_had_phi_true',  b_b_had_phi_true,  'b_had_phi_true/F' )
    tree.Branch( 't_had_px_true',   b_t_had_px_true,   't_had_px_true/F' )
    tree.Branch( 't_had_py_true',   b_t_had_py_true,   't_had_py_true/F' )
    tree.Branch( 't_had_pz_true',   b_t_had_pz_true,   't_had_pz_true/F' )
    tree.Branch( 't_had_E_true',    b_t_had_E_true,    't_had_E_true/F' )
    tree.Branch( 't_had_m_true',    b_t_had_m_true,    't_had_m_true/F' )
    tree.Branch( 't_had_pt_true',   b_t_had_pt_true,   't_had_pt_true/F' )
    tree.Branch( 't_had_y_true',    b_t_had_y_true,    't_had_y_true/F' )
    tree.Branch( 't_had_phi_true',  b_t_had_phi_true,  't_had_phi_true/F' )
    tree.Branch( 'W_lep_px_true',   b_W_lep_px_true,   'W_lep_px_true/F' )
    tree.Branch( 'W_lep_py_true',   b_W_lep_py_true,   'W_lep_py_true/F' )
    tree.Branch( 'W_lep_pz_true',   b_W_lep_pz_true,   'W_lep_pz_true/F' )
    tree.Branch( 'W_lep_E_true',    b_W_lep_E_true,    'W_lep_E_true/F' )
    tree.Branch( 'W_lep_m_true',    b_W_lep_m_true,    'W_lep_m_true/F' )
    tree.Branch( 'W_lep_pt_true',   b_W_lep_pt_true,   'W_lep_pt_true/F' )
    tree.Branch( 'W_lep_y_true',    b_W_lep_y_true,    'W_lep_y_true/F' )
    tree.Branch( 'W_lep_phi_true',  b_W_lep_phi_true,  'W_lep_phi_true/F' )
    tree.Branch( 'b_lep_px_true',   b_b_lep_px_true,   'b_lep_px_true/F' )
    tree.Branch( 'b_lep_py_true',   b_b_lep_py_true,   'b_lep_py_true/F' )
    tree.Branch( 'b_lep_pz_true',   b_b_lep_pz_true,   'b_lep_pz_true/F' )
    tree.Branch( 'b_lep_E_true',    b_b_lep_E_true,    'b_lep_E_true/F' )
    tree.Branch( 'b_lep_m_true',    b_b_lep_m_true,    'b_lep_m_true/F' )
    tree.Branch( 'b_lep_pt_true',   b_b_lep_pt_true,   'b_lep_pt_true/F' )
    tree.Branch( 'b_lep_y_true',    b_b_lep_y_true,    'b_lep_y_true/F' )
    tree.Branch( 'b_lep_phi_true',  b_b_lep_phi_true,  'b_lep_phi_true/F' )
    tree.Branch( 't_lep_px_true',   b_t_lep_px_true,   't_lep_px_true/F' )
    tree.Branch( 't_lep_py_true',   b_t_lep_py_true,   't_lep_py_true/F' )
    tree.Branch( 't_lep_pz_true',   b_t_lep_pz_true,   't_lep_pz_true/F' )
    tree.Branch( 't_lep_E_true',    b_t_lep_E_true,    't_lep_E_true/F' )
    tree.Branch( 't_lep_m_true',    b_t_lep_m_true,    't_lep_m_true/F' )
    tree.Branch( 't_lep_pt_true',   b_t_lep_pt_true,   't_lep_pt_true/F' )
    tree.Branch( 't_lep_y_true',    b_t_lep_y_true,    't_lep_y_true/F' )
    tree.Branch( 't_lep_phi_true',  b_t_lep_phi_true,  't_lep_phi_true/F' )

    tree.Branch( 'W_had_px_fitted',   b_W_had_px_fitted,   'W_had_px_fitted/F' )
    tree.Branch( 'W_had_py_fitted',   b_W_had_py_fitted,   'W_had_py_fitted/F' )
    tree.Branch( 'W_had_pz_fitted',   b_W_had_pz_fitted,   'W_had_pz_fitted/F' )
    tree.Branch( 'W_had_E_fitted',    b_W_had_E_fitted,    'W_had_E_fitted/F' )
    tree.Branch( 'W_had_m_fitted',    b_W_had_m_fitted,    'W_had_m_fitted/F' )
    tree.Branch( 'W_had_pt_fitted',   b_W_had_pt_fitted,   'W_had_pt_fitted/F' )
    tree.Branch( 'W_had_y_fitted',    b_W_had_y_fitted,    'W_had_y_fitted/F' )
    tree.Branch( 'W_had_phi_fitted',  b_W_had_phi_fitted,  'W_had_phi_fitted/F' )
    tree.Branch( 'b_had_px_fitted',   b_b_had_px_fitted,   'b_had_px_fitted/F' )
    tree.Branch( 'b_had_py_fitted',   b_b_had_py_fitted,   'b_had_py_fitted/F' )
    tree.Branch( 'b_had_pz_fitted',   b_b_had_pz_fitted,   'b_had_pz_fitted/F' )
    tree.Branch( 'b_had_E_fitted',    b_b_had_E_fitted,    'b_had_E_fitted/F' )
    tree.Branch( 'b_had_m_fitted',    b_b_had_m_fitted,    'b_had_m_fitted/F' )
    tree.Branch( 'b_had_pt_fitted',   b_b_had_pt_fitted,   'b_had_pt_fitted/F' )
    tree.Branch( 'b_had_y_fitted',    b_b_had_y_fitted,    'b_had_y_fitted/F' )
    tree.Branch( 'b_had_phi_fitted',  b_b_had_phi_fitted,  'b_had_phi_fitted/F' )
    tree.Branch( 't_had_px_fitted',   b_t_had_px_fitted,   't_had_px_fitted/F' )
    tree.Branch( 't_had_py_fitted',   b_t_had_py_fitted,   't_had_py_fitted/F' )
    tree.Branch( 't_had_pz_fitted',   b_t_had_pz_fitted,   't_had_pz_fitted/F' )
    tree.Branch( 't_had_E_fitted',    b_t_had_E_fitted,    't_had_E_fitted/F' )
    tree.Branch( 't_had_m_fitted',    b_t_had_m_fitted,    't_had_m_fitted/F' )
    tree.Branch( 't_had_pt_fitted',   b_t_had_pt_fitted,   't_had_pt_fitted/F' )
    tree.Branch( 't_had_y_fitted',    b_t_had_y_fitted,    't_had_y_fitted/F' )
    tree.Branch( 't_had_phi_fitted',  b_t_had_phi_fitted,  't_had_phi_fitted/F' )
    tree.Branch( 'W_lep_px_fitted',   b_W_lep_px_fitted,   'W_lep_px_fitted/F' )
    tree.Branch( 'W_lep_py_fitted',   b_W_lep_py_fitted,   'W_lep_py_fitted/F' )
    tree.Branch( 'W_lep_pz_fitted',   b_W_lep_pz_fitted,   'W_lep_pz_fitted/F' )
    tree.Branch( 'W_lep_E_fitted',    b_W_lep_E_fitted,    'W_lep_E_fitted/F' )
    tree.Branch( 'W_lep_m_fitted',    b_W_lep_m_fitted,    'W_lep_m_fitted/F' )
    tree.Branch( 'W_lep_pt_fitted',   b_W_lep_pt_fitted,   'W_lep_pt_fitted/F' )
    tree.Branch( 'W_lep_y_fitted',    b_W_lep_y_fitted,    'W_lep_y_fitted/F' )
    tree.Branch( 'W_lep_phi_fitted',  b_W_lep_phi_fitted,  'W_lep_phi_fitted/F' )
    tree.Branch( 'b_lep_px_fitted',   b_b_lep_px_fitted,   'b_lep_px_fitted/F' )
    tree.Branch( 'b_lep_py_fitted',   b_b_lep_py_fitted,   'b_lep_py_fitted/F' )
    tree.Branch( 'b_lep_pz_fitted',   b_b_lep_pz_fitted,   'b_lep_pz_fitted/F' )
    tree.Branch( 'b_lep_E_fitted',    b_b_lep_E_fitted,    'b_lep_E_fitted/F' )
    tree.Branch( 'b_lep_m_fitted',    b_b_lep_m_fitted,    'b_lep_m_fitted/F' )
    tree.Branch( 'b_lep_pt_fitted',   b_b_lep_pt_fitted,   'b_lep_pt_fitted/F' )
    tree.Branch( 'b_lep_y_fitted',    b_b_lep_y_fitted,    'b_lep_y_fitted/F' )
    tree.Branch( 'b_lep_phi_fitted',  b_b_lep_phi_fitted,  'b_lep_phi_fitted/F' )
    tree.Branch( 't_lep_px_fitted',   b_t_lep_px_fitted,   't_lep_px_fitted/F' )
    tree.Branch( 't_lep_py_fitted',   b_t_lep_py_fitted,   't_lep_py_fitted/F' )
    tree.Branch( 't_lep_pz_fitted',   b_t_lep_pz_fitted,   't_lep_pz_fitted/F' )
    tree.Branch( 't_lep_E_fitted',    b_t_lep_E_fitted,    't_lep_E_fitted/F' )
    tree.Branch( 't_lep_m_fitted',    b_t_lep_m_fitted,    't_lep_m_fitted/F' )
    tree.Branch( 't_lep_pt_fitted',   b_t_lep_pt_fitted,   't_lep_pt_fitted/F' )
    tree.Branch( 't_lep_y_fitted',    b_t_lep_y_fitted,    't_lep_y_fitted/F' )
    tree.Branch( 't_lep_phi_fitted',  b_t_lep_phi_fitted,  't_lep_phi_fitted/F' )

    ################################################################################
    # POPULATE TREE
    print("INFO: starting event loop. Found %i events" % n_events)
    n_good = 0
    # Print out example
    for i in range(n_events):
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))

        w = 1
        jets_n  = event_info[i][3]
        bjets_n = event_info[i][4]

        W_had_true   = MakeP4( y_true_W_had[i], m_W , representation)
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W , representation)

        W_lep_true   = MakeP4( y_true_W_lep[i], m_W , representation)
        W_lep_fitted = MakeP4( y_fitted_W_lep[i],  m_W , representation)

        b_had_true   = MakeP4( y_true_b_had[i], m_b , representation)
        b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b , representation)

        b_lep_true   = MakeP4( y_true_b_lep[i], m_b , representation)
        b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b , representation)

        t_had_true   = MakeP4( y_true_t_had[i], m_t , representation)
        t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t , representation)

        t_lep_true   = MakeP4( y_true_t_lep[i], m_t , representation)
        t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t , representation)

        # Fill branches
        b_eventNumber[0] = int(event_info[i][0])
        b_runNumber[0]   = int(event_info[i][1])
        b_weight_mc[0]   = float(event_info[i][2])

        b_W_had_px_true[0]  = W_had_true.Px()
        b_W_had_py_true[0]  = W_had_true.Py()
        b_W_had_pz_true[0]  = W_had_true.Pz()
        b_W_had_E_true[0]   = W_had_true.E()
        b_W_had_m_true[0]   = W_had_true.M()
        b_W_had_pt_true[0]  = W_had_true.Pt()
        b_W_had_y_true[0]   = W_had_true.Rapidity()
        b_W_had_phi_true[0] = W_had_true.Phi()

        b_b_had_px_true[0]  = b_had_true.Px()
        b_b_had_py_true[0]  = b_had_true.Py()
        b_b_had_pz_true[0]  = b_had_true.Pz()
        b_b_had_E_true[0]   = b_had_true.E()
        b_b_had_m_true[0]   = b_had_true.M()
        b_b_had_pt_true[0]  = b_had_true.Pt()
        b_b_had_y_true[0]   = b_had_true.Rapidity()
        b_b_had_phi_true[0] = b_had_true.Phi()

        b_t_had_px_true[0]  = t_had_true.Px()
        b_t_had_py_true[0]  = t_had_true.Py()
        b_t_had_pz_true[0]  = t_had_true.Pz()
        b_t_had_E_true[0]   = t_had_true.E()
        b_t_had_m_true[0]   = t_had_true.M()
        b_t_had_pt_true[0]  = t_had_true.Pt()
        b_t_had_y_true[0]   = t_had_true.Rapidity()
        b_t_had_phi_true[0] = t_had_true.Phi()

        b_W_lep_px_true[0]  = W_lep_true.Px()
        b_W_lep_py_true[0]  = W_lep_true.Py()
        b_W_lep_pz_true[0]  = W_lep_true.Pz()
        b_W_lep_E_true[0]   = W_lep_true.E()
        b_W_lep_m_true[0]   = W_lep_true.M()
        b_W_lep_pt_true[0]  = W_lep_true.Pt()
        b_W_lep_y_true[0]   = W_lep_true.Rapidity()
        b_W_lep_phi_true[0] = W_lep_true.Phi()

        b_b_lep_px_true[0]  = b_lep_true.Px()
        b_b_lep_py_true[0]  = b_lep_true.Py()
        b_b_lep_pz_true[0]  = b_lep_true.Pz()
        b_b_lep_E_true[0]   = b_lep_true.E()
        b_b_lep_m_true[0]   = b_lep_true.M()
        b_b_lep_pt_true[0]  = b_lep_true.Pt()
        b_b_lep_y_true[0]   = b_lep_true.Rapidity()
        b_b_lep_phi_true[0] = b_lep_true.Phi()

        b_t_lep_px_true[0]  = t_lep_true.Px()
        b_t_lep_py_true[0]  = t_lep_true.Py()
        b_t_lep_pz_true[0]  = t_lep_true.Pz()
        b_t_lep_E_true[0]   = t_lep_true.E()
        b_t_lep_m_true[0]   = t_lep_true.M()
        b_t_lep_pt_true[0]  = t_lep_true.Pt()
        b_t_lep_y_true[0]   = t_lep_true.Rapidity()
        b_t_lep_phi_true[0] = t_lep_true.Phi()

        b_W_had_px_fitted[0]  = W_had_fitted.Px()
        b_W_had_py_fitted[0]  = W_had_fitted.Py()
        b_W_had_pz_fitted[0]  = W_had_fitted.Pz()
        b_W_had_E_fitted[0]   = W_had_fitted.E()
        b_W_had_m_fitted[0]   = W_had_fitted.M()
        b_W_had_pt_fitted[0]  = W_had_fitted.Pt()
        b_W_had_y_fitted[0]   = W_had_fitted.Rapidity()
        b_W_had_phi_fitted[0] = W_had_fitted.Phi()

        b_b_had_px_fitted[0]  = b_had_fitted.Px()
        b_b_had_py_fitted[0]  = b_had_fitted.Py()
        b_b_had_pz_fitted[0]  = b_had_fitted.Pz()
        b_b_had_E_fitted[0]   = b_had_fitted.E()
        b_b_had_m_fitted[0]   = b_had_fitted.M()
        b_b_had_pt_fitted[0]  = b_had_fitted.Pt()
        b_b_had_y_fitted[0]   = b_had_fitted.Rapidity()
        b_b_had_phi_fitted[0] = b_had_fitted.Phi()

        b_t_had_px_fitted[0]  = t_had_fitted.Px()
        b_t_had_py_fitted[0]  = t_had_fitted.Py()
        b_t_had_pz_fitted[0]  = t_had_fitted.Pz()
        b_t_had_E_fitted[0]   = t_had_fitted.E()
        b_t_had_m_fitted[0]   = t_had_fitted.M()
        b_t_had_pt_fitted[0]  = t_had_fitted.Pt()
        b_t_had_y_fitted[0]   = t_had_fitted.Rapidity()
        b_t_had_phi_fitted[0] = t_had_fitted.Phi()

        b_W_lep_px_fitted[0]  = W_lep_fitted.Px()
        b_W_lep_py_fitted[0]  = W_lep_fitted.Py()
        b_W_lep_pz_fitted[0]  = W_lep_fitted.Pz()
        b_W_lep_E_fitted[0]   = W_lep_fitted.E()
        b_W_lep_m_fitted[0]   = W_lep_fitted.M()
        b_W_lep_pt_fitted[0]  = W_lep_fitted.Pt()
        b_W_lep_y_fitted[0]   = W_lep_fitted.Rapidity()
        b_W_lep_phi_fitted[0] = W_lep_fitted.Phi()

        b_b_lep_px_fitted[0]  = b_lep_fitted.Px()
        b_b_lep_py_fitted[0]  = b_lep_fitted.Py()
        b_b_lep_pz_fitted[0]  = b_lep_fitted.Pz()
        b_b_lep_E_fitted[0]   = b_lep_fitted.E()
        b_b_lep_m_fitted[0]   = b_lep_fitted.M()
        b_b_lep_pt_fitted[0]  = b_lep_fitted.Pt()
        b_b_lep_y_fitted[0]   = b_lep_fitted.Rapidity()
        b_b_lep_phi_fitted[0] = b_lep_fitted.Phi()

        b_t_lep_px_fitted[0]  = t_lep_fitted.Px()
        b_t_lep_py_fitted[0]  = t_lep_fitted.Py()
        b_t_lep_pz_fitted[0]  = t_lep_fitted.Pz()
        b_t_lep_E_fitted[0]   = t_lep_fitted.E()
        b_t_lep_m_fitted[0]   = t_lep_fitted.M()
        b_t_lep_pt_fitted[0]  = t_lep_fitted.Pt()
        b_t_lep_y_fitted[0]   = t_lep_fitted.Rapidity()
        b_t_lep_phi_fitted[0] = t_lep_fitted.Phi()

        tree.Fill()

        n_good += 1

    ################################################################################
    # CLOSE PROGRAM
    ofile.Write()
    ofile.Close()