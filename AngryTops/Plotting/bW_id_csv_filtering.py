# Filters events to be fed into network by whether their combinations of jets pass the cuts. 
import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
import pandas as pd
from AngryTops.ModelTraining.FormatInputOutput import *
from AngryTops.features import *
from sklearn.utils import shuffle
import sklearn.preprocessing
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

representation = sys.argv[1]
date = ''
if len(sys.argv) > 2:
    date = sys.argv[2]   
        
input_csv = 'Feb9.csv'
good_output_csv = '{}_good.csv'.format(date)
bad_output_csv = '{}_bad.csv'.format(date)
subdir = "/../../csv/training_closejets_cuts{}/".format(date)
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
W_had_m_cutoff = (30, 130)
W_had_pT_cutoff = (-100, 100)
W_had_dist_cutoff = (0, 0.8)

W_lep_ET_cutoff = (-100, 120)
W_lep_dist_cutoff = (0, 1.0)

b_had_pT_cutoff = (-80, 100)
b_had_dist_cutoff = (0, 0.8)

b_lep_pT_cutoff = (-80, 100)
b_lep_dist_cutoff = (0, 0.8)

jets, true  = get_input_output_all(input_csv, rep="pxpypzEM")

# Inputs
scaling = 'minmax'
rep = 'pxpypzEM'

particles_shape = (true.shape[1], true.shape[2])
        
# Rescale the jets array
jets_lep = jets[:,:6]
jets_jets = jets[:,6:] # remove muon column
jets_jets = jets_jets.reshape((jets_jets.shape[0],5,6)) # reshape to 5 x 6 array

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

# store number of events as a separate variable for clarity
n_events = true.shape[0]
event_index = range(n_events)

# Function to make histograms
def make_histograms():

    # Counters to make tally number of events that pass cuts
    W_had_jets = [0., 0., 0.] # List of number of events best matched to 1,2,3 jets respectively.
    W_had_total_cuts = [0., 0., 0.]
    W_had_m_cuts = [0., 0., 0.]
    W_had_pT_cuts = [0., 0., 0.]
    W_had_dist_cuts = [0., 0., 0.]

    W_lep_total_cuts = 0.
    W_lep_ET_cuts = 0.
    W_lep_dist_cuts = 0.

    b_had_pT_cuts = 0.
    b_had_dist_cuts = 0.
    b_had_total_cuts = 0.

    b_lep_pT_cuts = 0.
    b_lep_dist_cuts = 0.
    b_lep_total_cuts = 0.

    good_event = 0.
    
    # Lists of indices corresponding to good and bad events to remove from the original csv.
    good_indices = []
    bad_indices = []

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
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
        if not np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)

        ################################################# true vs observed ################################################# 

        # Set initial distances to be large since we don't know what the minimum distance is yet 
        b_had_dist_true = b_lep_dist = W_had_dist = 1000000

        # Perform jet matching for the bs, all jets, b-tagged and not b-tagged should be considered.
        for k in range(len(jets)):
            b_had_d_true = find_dist(b_had_true, jets[k])
            if b_had_d_true < b_had_dist_true:
                b_had_dist_true = b_had_d_true
                closest_b_had = jets[k]
            b_lep_d_true = find_dist(b_lep_true, jets[k])
            if b_lep_d_true < b_lep_dist:
                b_lep_dist = b_lep_d_true
                closest_b_lep = jets[k]

        good_jets = jets[:]
        if (b_tagging > 0):
            for m in range(len(jets)):
                # if don't include any b tagged jets and jet is b tagged OR
                # if only considering b tagged jets and jet is not b tagged
                if (b_tagging == 1 and jet_list[m, i, 5]) or (b_tagging == 2 and not jet_list[m,i,5]):
                    good_jets.remove(jets[m])
        if not good_jets:
            continue
        
        if (len(good_jets) >= 2):
            for k in range(len(good_jets)):
                # if good_jets only contains one element, loop is skipped since range would be (1,1)
                for j in range(k + 1, len(good_jets)):                  
                    sum_vect = good_jets[k] + good_jets[j] 
                    W_had_d_true = find_dist(W_had_true, sum_vect)
                    if W_had_d_true < W_had_dist:
                        W_had_dist = W_had_d_true
                        closest_W_had = sum_vect
            W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 1
        
        # If the best double jet doesn't pass cuts, then consider three jets.
        if (len(good_jets) >= 3) and (closest_W_had.M() <= W_had_m_cutoff[0] or closest_W_had.M() >= W_had_m_cutoff[1] \
            or W_had_pT_diff <= W_had_pT_cutoff[0] or W_had_pT_diff >= W_had_pT_cutoff[1] \
            or W_had_dist >= W_had_dist_cutoff[1]):
            # Reset maximum eta-phi distance.
            W_had_dist = 1000000
            for k in range(len(good_jets)):
                for j in range(k + 1, len(good_jets)):     
                    for l in range(j+1, len(good_jets)):
                        sum_vect = good_jets[k] + good_jets[j] + good_jets[l]
                        W_had_d_true = find_dist(W_had_true, sum_vect)
                        if W_had_d_true < W_had_dist:
                            W_had_dist = W_had_d_true
                            closest_W_had = sum_vect
            W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 2

        # if there is only one jet in the list or previous matches don't pass cutoff conditions, find a single jet match
        if (len(good_jets) == 1) or ((closest_W_had.M() <= W_had_m_cutoff[0] or closest_W_had.M() >= W_had_m_cutoff[1]) \
            or (W_had_pT_diff <= W_had_pT_cutoff[0] or W_had_pT_diff >= W_had_pT_cutoff[1])\
            or W_had_dist >= W_had_dist_cutoff[1]):
            W_had_dist = 1000000
            for k in range(len(good_jets)):
                sum_vect = good_jets[k]    
                W_had_d_true = find_dist(W_had_true, sum_vect)
                if W_had_d_true < W_had_dist:
                    W_had_dist = W_had_d_true
                    closest_W_had = sum_vect
            W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 0

        # Special calculations for the observed leptonic W assuming massless daughters.  
        muon_pT_obs = [jet_mu[i][0], jet_mu[i][1]]
        # Observed neutrino transverse momentum from missing energy as [x, y].
        nu_pT_obs = [jet_mu[i][4]*np.cos(jet_mu[i][5]), jet_mu[i][4]*np.sin(jet_mu[i][5])] 
        W_lep_Px_observed = muon_pT_obs[0] + nu_pT_obs[0]
        W_lep_Py_observed = muon_pT_obs[1] + nu_pT_obs[1]
        
        # Calculate the distance between true and observed phi.
        lep_phi = np.arctan2(W_lep_Py_observed, W_lep_Px_observed)
        W_lep_dist = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
        # Calculate transverse energy assuming daughter particles are massless
        W_lep_ET_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
        W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed
        # Calculate the transverse mass
        obs_daughter_angle = np.arccos( np.dot(muon_pT_obs, nu_pT_obs) / norm(muon_pT_obs) / norm(nu_pT_obs) )
        met_obs = np.sqrt(2*jet_mu[i][4]*jet_mu_vect.Pt()*(1 - np.cos(obs_daughter_angle))) 

        # b quark calculations
        b_had_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
        b_lep_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

        ############################################## check whether each event passes cuts #################################################
        # counter for hadronic W
        # Update tally for which jet combination is the closest
        W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
        W_had_pT_cut = (W_had_pT_diff >= W_had_pT_cutoff[0] and W_had_pT_diff <= W_had_pT_cutoff[1])
        W_had_dist_cut = (W_had_dist <= W_had_dist_cutoff[1]) 
        good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)
        W_had_jets[jet_combo_index] += 1.
        W_had_total_cuts[jet_combo_index] += good_W_had
        W_had_m_cuts[jet_combo_index] += W_had_m_cut
        W_had_pT_cuts[jet_combo_index] += W_had_pT_cut
        W_had_dist_cuts[jet_combo_index] += W_had_dist_cut

        # counter for lep W
        W_lep_ET_cut = (W_lep_ET_diff >= W_lep_ET_cutoff[0] and W_lep_ET_diff <= W_lep_ET_cutoff[1])
        W_lep_dist_cut = (W_lep_dist <= W_lep_dist_cutoff[1]) 
        good_W_lep = (W_lep_ET_cut and W_lep_dist_cut)
        W_lep_total_cuts += good_W_lep
        W_lep_ET_cuts += W_lep_ET_cut
        W_lep_dist_cuts += W_lep_dist_cut
        # counter for hadronic b
        b_had_pT_cut = (b_had_pT_diff >= b_had_pT_cutoff[0] and b_had_pT_diff <= b_had_pT_cutoff[1])
        b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
        good_b_had = (b_had_pT_cut and b_had_dist_cut)
        b_had_total_cuts += good_b_had
        b_had_pT_cuts += b_had_pT_cut
        b_had_dist_cuts += b_had_dist_cut

        # counter for leptonic b
        b_lep_pT_cut = (b_lep_pT_diff >= b_lep_pT_cutoff[0] and b_lep_pT_diff <= b_lep_pT_cutoff[1])
        b_lep_dist_cut = (b_lep_dist <= b_lep_dist_cutoff[1]) 
        good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)
        b_lep_total_cuts += good_b_lep
        b_lep_pT_cuts += b_lep_pT_cut
        b_lep_dist_cuts += b_lep_dist_cut

        # Good events must pass cuts on all partons.
        good_event += (good_b_had and good_b_lep and good_W_had and good_W_lep)

        # Append indices to appropriate list.
        if good_W_had and good_W_lep and good_b_had and good_b_lep:
            good_indices.append(i)
        else:
            bad_indices.append(i)

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

    return (good_indices, bad_indices)


# Run the two helper functions above   
if __name__ == "__main__":
    
    # Read in again and slice original pandas dataframe to remove the event indices 
    #  not corresponding to the appropriate filter.
    
    # Load jets, leptons and output columns of the correct representation
    input_csv = "../csv/{}".format(input_csv)
    df = pd.read_csv(input_csv, names=column_names)

    # Create csv of reconstructable events
    good_df = df.drop(labels = bad_indices, axis =0)
    print(good_df.shape)
    print(bad_indices[0:100])
    print(good_df.index.tolist()[0:100])
    good_df.to_csv("../csv/{}".format(good_output_csv), header = False, index = False)

    # Create csv of unreconstructable events
    bad_df = df.drop(labels = good_indices, axis =0)
    print(bad_df.shape)
    print(good_indices[0:100])
    print(bad_df.index.tolist()[0:100])
    bad_df.to_csv("../csv/{}".format(bad_output_csv), header = False, index = False)