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


scaling = True
m_t = 172.5
m_W = 80.4
m_b = 4.95
ALL = 0
NONE = 1
ONLY = 2
b_tagging = NONE   # 0/All: Consider all jets, both b-tagged and not b-tagged
                # 1/None: Do not consider any b-tagged jets.
                # 2/Only: Consider only b-tagged jets

W_had_m_cutoff = (30, 130)
W_had_pT_cutoff = (-100, 100)
W_had_dist_cutoff = (0, 0.8)

W_lep_ET_cutoff = (-100, 120)
W_lep_dist_cutoff = (0, 1.0)

b_had_pT_cutoff = (-80, 100)
b_had_dist_cutoff = (0, 0.8)

b_lep_pT_cutoff = (-80, 100)
b_lep_dist_cutoff = (0, 0.8)

train_dir = sys.argv[1]

# Helper function to create histograms of eta-phi distance distributions
def MakeP4(y, m, representation):
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

    print("filtering training events...")
    (training_input, training_output), (testing_input, testing_output), \
    (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) \
                        = get_input_output(input_filename=csv_file, **kwargs)

    # Inputs
    scaling = kwargs['scaling']
    rep = kwargs['rep']

    training_input_cuts = []
    training_output_cuts = []

    training_output = training_output.reshape(training_output.shape[0], training_output.shape[1]*training_output.shape[2])
    training_output = output_scalar.inverse_transform(training_output)
    training_output = training_output.reshape(training_output.shape[0], -1, 3)

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

    for i in event_index: # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
        
        W_had_true   = MakeP4( y_true_W_had[i], m_W , rep)
        W_lep_true   = MakeP4( y_true_W_lep[i], m_W , rep)
        b_had_true   = MakeP4( y_true_b_had[i], m_b , rep)
        b_lep_true   = MakeP4( y_true_b_lep[i], m_b , rep)

        jet_mu_vect = MakeP4(jet_mu[i],jet_mu[i][4], rep)

        jet_1_vect = MakeP4(jet_1[i], jet_1[i][4], rep)
        jet_2_vect = MakeP4(jet_2[i], jet_2[i][4], rep)
        jet_3_vect = MakeP4(jet_3[i], jet_3[i][4], rep)
        jet_4_vect = MakeP4(jet_4[i], jet_4[i][4], rep)
        jet_5_vect = MakeP4(jet_5[i], jet_5[i][4], rep)
    
        jets = []
        # add list containing jets of corresponding event
        jets.append(jet_1_vect)
        jets.append(jet_2_vect)
        jets.append(jet_3_vect)
        jets.append(jet_4_vect)
        # If there is no fifth jet, do not append it to list of jets to avoid considering it in the pairs of jets.
        if np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)

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
            W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 1
        
        # If the best double jet doesn't pass cuts, then consider three jets.
        if (len(good_jets) >= 3) and (closest_W_had.M() <= W_had_m_cutoff[0] \
            or closest_W_had.M() >= W_had_m_cutoff[1] or W_had_pT_diff <= W_had_pT_cutoff[0] \
            or W_had_pT_diff >= W_had_pT_cutoff[1] \
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
            W_had_pT_diff = W_had_true.Pt() - closest_W_had.Pt()
            jet_combo_index = 2

        # if there is only one jet in the list or previous matches don't pass cutoff conditions, find a single jet match
        if (len(good_jets) == 1) or ((closest_W_had.M() <= W_had_m_cutoff[0] or closest_W_had.M() >= W_had_m_cutoff[1]) \
            or (W_had_pT_diff <= W_had_pT_cutoff[0] or W_had_pT_diff >= W_had_pT_cutoff[1])\
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
        W_lep_dist_true = np.abs( min( np.abs(W_lep_true.Phi()-lep_phi), 2*np.pi-np.abs(W_lep_true.Phi()-lep_phi) ) )
        # Calculate transverse energy assuming daughter particles are massless
        W_lep_ET_observed = np.sqrt( W_lep_Px_observed**2 + W_lep_Py_observed**2)
        W_lep_ET_diff = W_lep_true.Et() - W_lep_ET_observed

        # b quark calculations
        b_had_pT_diff = b_had_true.Pt() - closest_b_had.Pt()
        b_lep_pT_diff = b_lep_true.Pt() - closest_b_lep.Pt()

        ############################################## check whether each event passes cuts #################################################
        
        # counter for hadronic W
        # Update tally for which jet combination is the closest
        W_had_m_cut = (closest_W_had.M() >= W_had_m_cutoff[0] and closest_W_had.M() <= W_had_m_cutoff[1])
        W_had_pT_cut = (W_had_pT_diff >= W_had_pT_cutoff[0] and W_had_pT_diff <= W_had_pT_cutoff[1])
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
        b_had_pT_cut = (b_had_pT_diff >= b_had_pT_cutoff[0] and b_had_pT_diff <= b_had_pT_cutoff[1])
        b_had_dist_cut = (b_had_dist_true <= b_had_dist_cutoff[1]) 
        good_b_had = (b_had_pT_cut and b_had_dist_cut)

        b_had_total_cuts += good_b_had
        b_had_pT_cuts += b_had_pT_cut
        b_had_dist_cuts += b_had_dist_cut

        # counter for leptonic b
        b_lep_pT_cut = (b_lep_pT_diff >= b_lep_pT_cutoff[0] and b_lep_pT_diff <= b_lep_pT_cutoff[1])
        b_lep_dist_cut = (b_lep_dist_true <= b_lep_dist_cutoff[1]) 
        good_b_lep = (b_lep_pT_cut and b_lep_dist_cut)

        b_lep_total_cuts += good_b_lep
        b_lep_pT_cuts += b_lep_pT_cut
        b_lep_dist_cuts += b_lep_dist_cut

        # Good events must pass cuts on all partons.
        good_event += (good_b_had and good_b_lep and good_W_had and good_W_lep)
        ################################################# populate histograms #################################################

        if not (good_W_had and good_W_lep and good_b_had and good_b_lep):
            training_input_cuts.append(i)
            training_output_cuts.append(i)

    training_input = np.delete(training_input, training_input_cuts, axis = 0)
    training_output = np.delete(training_output, training_output_cuts, axis = 0)

    # re-normalize training_output because it was un-normalized earlier
    training_output = training_output.reshape((training_output.shape[0], 18)) 
    training_output, _ = normalize(training_output, scaling)
    training_output = training_output.reshape((training_output.shape[0], -1, 3))

    np.savez("{}/cut_events".format(train_dir), training_input=testing_input, training_output=training_output,\
            testing_input=testing_input, testing_output=testing_output,\
            event_training=event_training, event_testing=event_testing)

    scaler_filename = "{}/scalers.pkl".format(train_dir)
    with open( scaler_filename, "wb" ) as file_scaler:
      pickle.dump(jets_scalar, file_scaler, protocol=2)
      pickle.dump(lep_scalar, file_scaler, protocol=2)
      pickle.dump(output_scalar, file_scaler, protocol=2)
    print("INFO: scalers saved to file:", scaler_filename)

    # Print data regarding percentage of each class of event
    print("jets shape", training_input.shape)
    print("b tagging option", b_tagging)
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
    # return (training_input, training_output), (testing_input, testing_output), \
    #        (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing)
    return True

if __name__=='__main__':
    # (training_input, training_output), (testing_input, testing_output), \
    #        (jets_scalar, lep_scalar, output_scalar), (event_training, event_testing) = \
    # filter_events("Feb9.csv", scaling='minmax', rep="pxpypzEM", sort_jets=False)
    # print(training_input.shape)
    # print(training_output.shape)
    try:
        os.mkdir(train_dir)
    except Exception as e:
        print("Directory already created")

    filter_events("Feb9.csv", scaling='minmax', rep="pxpypzEM", sort_jets=False)

