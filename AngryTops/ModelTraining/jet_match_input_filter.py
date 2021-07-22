# Filters events to be fed into network by whether their combinations of jets pass the cuts. 
import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
import pandas as pd
from AngryTops.ModelTraining.FormatInputOutput import *


scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = "None" # "None": Do not consider any b-tagged jets.
                   # "All": Consider all jets, both b-tagged and not b-tagged
                   # "Only": Consider only b-tagged jets

W_had_m_cutoff = (30, 130)
W_had_pT_cutoff = (-100, 100)
W_had_dist_cutoff = (0, 0.8)

W_lep_ET_cutoff = (-100, 120)
W_lep_dist_cutoff = (0, 1.0)

b_had_pT_cutoff = (-80, 100)
b_had_dist_cutoff = (0, 0.8)

b_lep_pT_cutoff = (-80, 100)
b_lep_dist_cutoff = (0, 0.8)

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

def filter_events(csv_file, **kwargs):
    ### Format input csv file.
    # Training observed, training truth, testing observed, testing truth
    #  Training to be split with validation.
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
                        = get_input_output(input_filename=csv_file, **kwargs)

    training_input_cuts = []
    training_output_cuts = []

    jets_lep = training_input[:,:6]
     # remove muon columns
    jets_jets = training_input[:,6:]
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) 
    # Remove the b-tagging states and put them into a new array to be re-appended later.
    b_tags = jets_jets[:,:,5]
    # delete the b-tagging states
    jets_jets = np.delete(jets_jets, 5, 2) 
    jets_jets = jets_jets.reshape((jets_jets.shape[0], 25)) 
    # need to invert the normalization due to get_input_output
    jets_lep = lep_scalar.inverse_transform(jets_lep)
    jets_jets = jets_scalar.inverse_transform(jets_jets)
    jets_jets = jets_jets.reshape((jets_jets.shape[0],5,5))
    # Re-append the b-tagging states as a column at the end of jets_jets 
    jets_jets = np.append(jets_jets, np.expand_dims(b_tags, 2), 2)

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
    y_true_W_had = training_output[:,0,:]
    y_true_W_lep = training_output[:,1,:]
    y_true_b_had = training_output[:,2,:]
    y_true_b_lep = training_output[:,3,:]
    y_true_t_had = training_output[:,4,:]
    y_true_t_lep = training_output[:,5,:]

    n_events = training_output.shape[0]
    event_index = range(n_events)

    w_had_jets = [0., 0., 0.] # List of number of events best matched by 1,2,3 jets respectively.
    W_had_total_cuts = [0., 0., 0.]

    W_lep_total_cuts = 0.
    b_had_total_cuts = 0.
    b_lep_total_cuts = 0.

    good_event = 0.

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
        # add list containing jets of corresponding event
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
        if np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)

        # Perform jet matching for the bs and Ws
        b_had_dist_true = b_lep_dist_true = W_had_dist_true = 1000
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
            # Same code for matching whether or not we are to include b-tagged jets
            if (b_tagging == "All") \
                or (b_tagging == "None" and not jet_list[k,i,5]) \
                    or (b_tagging == "Only" and jet_list[k,i,5]): # k ranges from 0 to 4 or 5 depending on event type
                # Go through each 1,2,3 combination of jets that are not b-tagged and check their sum
                sum_vect = jets[k]    
                # Single jets
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist_true:
                    W_had_dist_true = W_had_d_true
                    closest_W_had = sum_vect
                    w_jets = 0
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
                            closest_W_had = sum_vect
                            w_jets = 1
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
                                    closest_W_had = sum_vect
                                    w_jets = 2

        # Calculations for Leptonic W
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]] 
        nu_pT_obs = [ jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])]
        W_lep_Px = muon_pT_obs[0] + nu_pT_obs[0]
        W_lep_Py = muon_pT_obs[1] + nu_pT_obs[1]
        lep_phi = np.arctan2( W_lep_Py, W_lep_Px )
        # Distance
        W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
        # Transverse Energy
        W_lep_ET_observed = np.sqrt( W_lep_Px**2 + W_lep_Py**2)
        W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed

        b_had_true_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
        b_lep_true_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

        good_W_had = (W_had_dist_true <= W_had_dist_cutoff[1]) and \
                    (W_had_true_pT_diff >= W_had_pT_cutoff[0] and W_had_true_pT_diff <= W_had_pT_cutoff[1]) and\
                    (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])

        good_W_lep = (W_lep_dist_true <= W_lep_dist_cutoff[1]) and \
                    (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_true_ET_diff <= W_lep_ET_cutoff[1])

        good_b_had =  (b_had_dist_true <= b_had_dist_cutoff[1]) and \
                    (b_had_true_pT_diff >= b_had_pT_cutoff[0] and b_had_true_pT_diff <= b_had_pT_cutoff[1]) and\
                    (closest_b_had.M() >= b_had_m_cutoff[0] and closest_b_had.M() <= b_had_m_cutoff[1])

        good_b_lep = (b_lep_dist_true <= b_lep_dist_cutoff[1]) and \
                    (b_lep_true_pT_diff >= b_lep_pT_cutoff[0] and b_lep_true_pT_diff <= b_lep_pT_cutoff[1]) and\
                    (closest_b_lep.M() >= b_lep_m_cutoff[0] and closest_b_lep.M() <= b_lep_m_cutoff[1])

        if good_W_had and good_W_lep and good_b_had and good_b_lep:
            training_input_cuts.append(i)
            training_output_cuts.append(i)
            good_event += 1

        # calculate stats
        W_had_jets[jet_combo_index] += 1.
        W_had_total_cuts[jet_combo_index] += good_W_had

        W_lep_total_cuts += good_W_lep
        b_had_total_cuts += good_b_had
        b_lep_total_cuts += good_b_lep

    training_input = np.delete(training_input, training_input_cuts, axis = 0)
    training_output = np.delete(training_output, training_output_cuts, axis = 0)

    scaling = kwargs['scaling']
    training_input = normalize(training_input, scaling)
    training_output = normalize(training_output, scaling)

    # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))
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
    print('\n==================================================================\n')
    print("Number of events satisfying all leptonic W cut criteria")
    print('{}% , {} events\n'.format(100.*W_lep_total_cuts/n_events, int(W_lep_total_cuts)))
    print("Number of events satisfying all hadronic b cut criteria")
    print('{}% , {} events\n'.format(100.*b_had_total_cuts/n_events, int(b_had_total_cuts)))
    print("Number of events satisfying all leptonic b cut criteria")
    print('{}% , {} events\n'.format(100.*b_lep_total_cuts/n_events, int(b_lep_total_cuts)))
    print('\n==================================================================\n')
    print("Events satisfying cut all cut criteria for all partons")
    print('{}%, {} events'.format(100.*good_event/n_events, int(good_event)))



    return (training_input, training_output), (testing_input, testing_output), \
           (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)

