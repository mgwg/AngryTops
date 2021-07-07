# Filters events to be fed into network by whether their combinations of jets pass the cuts. 
import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
import pandas as pd
from AngryTops.ModelTraining.FormatInputOutput import get_input_output


scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
b_tagging = "None" # "None": Do not consider any b-tagged jets.
                   # "All": Consider all jets, both b-tagged and not b-tagged
                   # "Only": Consider only b-tagged jets

dist_true_v_obs_max = 0.5 # Maximum true vs. observed eta-phi distance for cuts
pT_diff_true_v_obs_max = 50
pT_diff_true_v_obs_min = -50
mass_obs_max = 100
mass_obs_min = 25

event_type = 0
if len(sys.argv) > 3:
    event_type = sys.argv[3]

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
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
                        = get_input_output(input_filename=csv_file, **kwargs)

    print(training_input)
    print(training_output)
    print(jets_scalar)

filter_events('Feb9.csv', scaling='minmax', rep='pxpypzEM', EPOCHES=25, sort_jets=False)

#     # define tolerance limits
#     b_lep_dist_t_lim = 0.39
#     b_had_dist_t_lim = 0.39
#     t_lep_dist_t_lim = 0.80
#     t_had_dist_t_lim = 0.80
#     W_lep_dist_t_lim = 1.28
#     W_had_dist_t_lim = 1.28

#     good_b_lep = good_b_had = 0.
#     good_W_lep = good_W_had = 0.
#     bad_b_lep = bad_b_had = 0.
#     bad_W_lep = bad_W_had = 0.

#     w_had_jets = [0., 0., 0.] # List of number of events best matched by 1,2,3 jets respectively.

#     for i in event_index: # loop through every event
#         if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
#             perc = 100. * i / float(n_events)
#             print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
#         W_had_true   = MakeP4( y_true_W_had[i], m_W )
#         W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W)

#         W_lep_true   = MakeP4( y_true_W_lep[i], m_W )
#         W_lep_fitted = MakeP4( y_fitted_W_lep[i],  m_W)

#         b_had_true   = MakeP4( y_true_b_had[i], m_b )
#         b_had_fitted = MakeP4( y_fitted_b_had[i],  m_b )

#         b_lep_true   = MakeP4( y_true_b_lep[i], m_b )
#         b_lep_fitted = MakeP4( y_fitted_b_lep[i],  m_b)

#         t_had_true   = MakeP4( y_true_t_had[i], m_t )
#         t_had_fitted = MakeP4( y_fitted_t_had[i],  m_t )

#         t_lep_true   = MakeP4( y_true_t_lep[i], m_t )
#         t_lep_fitted = MakeP4( y_fitted_t_lep[i],  m_t)

#         jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4])

#         jet_1_vect = MakeP4(jet_1[i], jet_1[i][4])
#         jet_2_vect = MakeP4(jet_2[i], jet_2[i][4])
#         jet_3_vect = MakeP4(jet_3[i], jet_3[i][4])
#         jet_4_vect = MakeP4(jet_4[i], jet_4[i][4])
#         jet_5_vect = MakeP4(jet_5[i], jet_5[i][4])
        
#         jets = []
#         # add list containing jets of corresponding event
#         jets.append(jet_1_vect)
#         jets.append(jet_2_vect)
#         jets.append(jet_3_vect)
#         jets.append(jet_4_vect)
#         # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
#         if np.all(jet_5[i] == 0.):
#             jets.append(jet_5_vect)

#         ################################################# true vs observed ################################################# 
#         b_had_dist_true = 1000
#         b_lep_dist_true = 1000
#         t_had_dist_true = 1000
#         t_lep_dist_true = 1000
#         W_had_true_pT_diff = 0
#         W_had_dist_true_start = 10000000
#         W_had_dist_true = W_had_dist_true_start
        
#         # Perform jet matching for the bs and Ws
#         for k in range(len(jets)): # loop through each jet to find the minimum distance for each particle
#             # For bs:
#             b_had_d_true = find_dist(b_had_true, jets[k])
#             b_lep_d_true = find_dist(b_lep_true, jets[k])
#             if b_had_d_true < b_had_dist_true:
#                 b_had_dist_true = b_had_d_true
#                 closest_b_had = jets[k]
#             if b_lep_d_true < b_lep_dist_true:
#                 b_lep_dist_true = b_lep_d_true
#                 closest_b_lep = jets[k]

#             # For hadronic Ws
#             # Same code for matching whether or not we are to include b-tagged jets
#             if (b_tagging == "All") \
#                 or (b_tagging == "None" and not jet_list[k,i,5]) \
#                     or (b_tagging == "Only" and jet_list[k,i,5]): # k ranges from 0 to 4 or 5 depending on event type
#                 # Go through each 1,2,3 combination of jets that are not b-tagged and check their sum
#                 sum_vect = jets[k]    
#                 # Single jets
#                 W_had_d_true = find_dist(W_had_true, sum_vect)
#                 if W_had_d_true < W_had_dist_true:
#                     W_had_dist_true = W_had_d_true
#                     W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
#                     closest_W_had = sum_vect
#                     w_jets = 0
#                 # Dijets
#                 for j in range(k + 1, len(jets)):
#                     # j ranges from k+1 to 4 or 5 depending on event type     
#                     if (b_tagging == "All") \
#                         or (b_tagging == "None" and not jet_list[j,i,5]) \
#                             or (b_tagging == "Only" and jet_list[j,i,5]):
#                         sum_vect = jets[k] + jets[j] 
#                         W_had_d_true = find_dist(W_had_true, sum_vect)
#                         if W_had_d_true < W_had_dist_true:
#                             W_had_dist_true = W_had_d_true
#                             W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
#                             closest_W_had = sum_vect
#                             w_jets = 1
#                 # Trijets
#                         for l in range(j+1, len(jets)):
#                             # l ranges from j+k+1 to 4 or 5 depending on event type
#                             if (b_tagging == "All") \
#                                 or (b_tagging == "None" and not jet_list[l,i,5]) \
#                                     or (b_tagging == "Only" and jet_list[l,i,5]):
#                                 sum_vect = jets[k] + jets[j] + jets[l]
#                                 W_had_d_true = find_dist(W_had_true, sum_vect)
#                                 if W_had_d_true < W_had_dist_true:
#                                     W_had_dist_true = W_had_d_true
#                                     W_had_true_pT_diff = W_had_true.Pt() - sum_vect.Pt()
#                                     closest_W_had = sum_vect
#                                     w_jets = 2

#         # If b-tagged jets are not to be considered and all jets are b-tagged 
#         #  or only b-tagged jets are to be considered and no jet is b-tagged, 
#         #  skip the event.
#         if W_had_dist_true == W_had_dist_true_start: # If there are no jets to be matched for this event,
#             continue                                 #  then the W_had_dist_true will remain unchanged. 

#         w_had_jets[w_jets] += 1 
                
#         # Calculate leptonic W distances

#         # Observed transverse momentum of muon
#         muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]] 
#         # Convert missing transverse energy to a momentum
#         nu_pT_obs = [ jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] # Observed neutrino transverse momentum from missing energy as [x, y].
#         # Add muon transverse momentum components to missing momentum components
#         lep_x = muon_pT_obs[0] + nu_pT_obs[0]
#         lep_y = muon_pT_obs[1] + nu_pT_obs[1]
#         # Calculate phi using definition in Kuunal's report 
#         lep_phi = np.arctan2( lep_y, lep_x )
#         # Calculate the distance between true and observed phi.
#         W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
 
#         # Compare hadronic t distances
#         t_had_jets = closest_b_had + closest_W_had
#         t_had_dist_true = find_dist(t_had_true, t_had_jets)
        
#         # Compare leptonic t distances
#         t_lep_x = lep_x + closest_b_lep.Px()
#         t_lep_y = lep_y + closest_b_lep.Py()
#         obs_t_phi = np.arctan2(t_lep_y, t_lep_x)
#         t_lep_dist_true = np.abs( min( np.abs(t_lep_true.Phi()-obs_t_phi), 2*np.pi-np.abs(t_lep_true.Phi() - obs_t_phi) ) )

#         # Add the number of jets that are within the tolerance limit, or reconstructable
#         corr_p_jets_dist = b_lep_R_recon + b_had_R_recon + W_lep_R_recon + W_had_R_recon
#         corr_jets_dist = 0.
#         if (b_lep_dist_true <= b_lep_dist_t_lim): # if minimum distance is less than the tolearance limits, everything is ok
#             corr_jets_dist += 1
#             good_b_lep += 1
#         else:
#             bad_b_lep += 1
#         if (b_had_dist_true <= b_had_dist_t_lim):
#             corr_jets_dist += 1
#             good_b_had += 1
#         else:
#             bad_b_had += 1
#         if (W_lep_dist_true <= W_lep_dist_t_lim): # mismatch between W_lep_dist_true and good_W_lep
#             corr_jets_dist += 1
#             good_W_lep += 1
#         else:
#             bad_W_lep += 1
#         if (W_had_dist_true <= W_had_dist_t_lim):
#             corr_jets_dist += 1
#             good_W_had += 1
#         else:
#             bad_W_had += 1

#         # Populate cut histograms
#         # Use only the observed jet(s) that pass all three cuts simultaneously: 
#         # if (W_had_dist_true <= dist_true_v_obs_max) and \
#         #     (W_had_true_pT_diff >= pT_diff_true_v_obs_min) and \
#         #         (W_had_true_pT_diff <= pT_diff_true_v_obs_max) and \
#         #             (closest_W_had.M() >= mass_obs_min) and \
#         #                 (closest_W_had.M() <= mass_obs_max)


# def normalize(arr, scaling):
#     """Normalize the arr with StandardScalar and return the normalized array
#     and the scalar"""
#     if scaling == "standard":
#         scalar = sklearn.preprocessing.StandardScaler()
#     elif scaling == 'minmax':
#         scalar = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
#     else:
#         return arr.copy(), None
#     new_arr = scalar.fit_transform(arr)
#     return new_arr, scalar