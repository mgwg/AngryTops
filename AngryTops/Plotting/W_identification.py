import os, sys
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
from ROOT import *
import pickle
from AngryTops.Plotting.identification_helper import MakeP4, find_dist, plot_hists

outputdir = sys.argv[1]
representation = sys.argv[2]
date = ''
if len(sys.argv) > 3:
    date = sys.argv[3]
event_type = 0
if len(sys.argv) > 4:
    event_type = sys.argv[4]

subdir = '/closejets_img{}/'.format(date)
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
# For plot_cuts:
# True if you want to plot only the events that pass the cuts
# False to include events for which no combo of 1,2,3 jets pass cuts.                
plot_cuts = True
if plot_cuts:
    subdir = '/closejets_img_cuts{}/'.format(date)

# Cut ranges for the partons
mass_cut = (25, 130)
pT_cut = (-50, 50)
dist_cut = (0, 0.7)

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

# make histograms to be filled
hists = {}

# Hadronic W
hists['had_W_obs_pT'] = TH1F("h_pT_W_had", "W Hadronic p_{T}, Observed", 80, 0, 400)
hists['had_W_obs_pT'].SetTitle("W Hadronic p_{T}, Observed; Hadronic (GeV); A.U.")
hists['had_W_true_obs_pT_diff'] = TH1F("h_pT_W_had_true","W Hadronic p_T, True - Observed", 50, -300, 300)
hists['had_W_true_obs_pT_diff'].SetTitle("W Hadronic p_{T} Diff, True - Observed; Hadronic (GeV); A.U.")
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

def make_histograms():

    # Counters to make tally number of events that pass cuts
    W_had_jets = [0., 0., 0.] # List of number of events best matched to 1,2,3 jets respectively.
    W_had_total_cuts = [0., 0., 0.]
    W_had_m_cuts = [0., 0., 0.]
    W_had_pT_cuts = [0., 0., 0.]
    W_had_dist_cuts = [0., 0., 0.]

    for i in range(n_events): # loop through every event
        if ( n_events < 10 ) or ( (i+1) % int(float(n_events)/10.)  == 0 ):
            perc = 100. * i / float(n_events)
            print("INFO: Event %-9i  (%3.0f %%)" % ( i, perc ))
            
        W_had_true   = MakeP4( y_true_W_had[i], m_W, representation)
        W_had_fitted = MakeP4( y_fitted_W_had[i],  m_W, representation)

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
        if np.all(jet_5[i] == 0.):
            jets.append(jet_5_vect)


        ################################################# true vs observed ################################################# 

        # Set initial distances to be large since we don't know what the minimum distance is yet 
        b_had_dist_true = b_lep_dist_true = W_had_dist_true = 1000000

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
        if (len(good_jets) >= 3) and (closest_W_had.M() <= mass_cut[0] \
            or closest_W_had.M() >= mass_cut[1] or W_had_pT_diff <= pT_cut[0] ):#\
            # or W_had_pT_diff >= pT_cut[1] \
            # or W_had_dist_true >= dist_cut[1]):
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
        if (len(good_jets) == 1) or ((closest_W_had.M() <= mass_cut[0] or closest_W_had.M() >= mass_cut[1]) ):#\
            # or (W_had_pT_diff <= pT_cut[0] or W_had_pT_diff >= pT_cut[1])\
            # or W_had_dist_true >= dist_cut[1]):
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

        ############################################## check whether each event passes cuts #################################################
        # counter for hadronic W
        # Update tally for which jet combination is the closest
        W_had_m_cut = (closest_W_had.M() >= mass_cut[0] and closest_W_had.M() <= mass_cut[1])
        W_had_pT_cut = (W_had_pT_diff >= pT_cut[0] and W_had_pT_diff <= pT_cut[1])
        W_had_dist_cut = (W_had_dist_true <= dist_cut[1]) 

        # All W_had cuts must be satisfied simultaneously.

        W_had_jets[jet_combo_index] += 1.
        W_had_m_cuts[jet_combo_index] += W_had_m_cut

        ################################################# populate histograms and check percentages #################################################

        if W_had_m_cut:
            
            W_had_pT_cuts[jet_combo_index] += W_had_pT_cut
            W_had_dist_cuts[jet_combo_index] += W_had_dist_cut
            good_W_had = (W_had_m_cut and W_had_pT_cut and W_had_dist_cut)
            W_had_total_cuts[jet_combo_index] += good_W_had

            # Hadronic W
            hists['had_W_dist_true_v_obs'].Fill(np.float(W_had_dist_true))
            # pT diff and Invariant mass:
            hists['had_W_obs_pT'].Fill(np.float(closest_W_had.Pt()))
            hists['had_W_true_obs_pT_diff'].Fill(np.float(W_had_pT_diff))
            hists['had_W_obs_mass'].Fill(np.float(closest_W_had.M()))
            # Jet matching criteria correlation plots
            hists['had_W_corr_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
            hists['had_W_corr_mass_Pt'].Fill(closest_W_had.M(), W_had_pT_diff)
            hists['had_W_corr_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_pT_diff)

            # Plots that depend on whether a 1,2, or 3-jet sum is the best match to truth:
            if jet_combo_index == 0:
                hists['had_W_obs_1_mass'].Fill(closest_W_had.M())
                hists['had_W_dist_1_true_v_obs'].Fill(np.float(W_had_dist_true))
                hists['had_W_corr_1_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_corr_1_mass_Pt'].Fill(closest_W_had.M(), W_had_pT_diff)
                hists['had_W_corr_1_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_pT_diff)
                hists['had_W_true_1_pT_diff'].Fill(np.float(W_had_pT_diff))
            if jet_combo_index == 1:
                hists['had_W_obs_2_mass'].Fill(closest_W_had.M())
                hists['had_W_dist_2_true_v_obs'].Fill(np.float(W_had_dist_true))
                hists['had_W_corr_2_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_corr_2_mass_Pt'].Fill(closest_W_had.M(), W_had_pT_diff)
                hists['had_W_corr_2_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_pT_diff)
                hists['had_W_true_2_pT_diff'].Fill(np.float(W_had_pT_diff))
            if jet_combo_index == 2:
                hists['had_W_obs_3_mass'].Fill(closest_W_had.M())
                hists['had_W_dist_3_true_v_obs'].Fill(np.float(W_had_dist_true))
                hists['had_W_corr_3_mass_dist'].Fill(closest_W_had.M(), np.float(W_had_dist_true))
                hists['had_W_corr_3_mass_Pt'].Fill(closest_W_had.M(), W_had_pT_diff)
                hists['had_W_corr_3_dist_Pt'].Fill(np.float(W_had_dist_true), W_had_pT_diff)
                hists['had_W_true_3_pT_diff'].Fill(np.float(W_had_pT_diff))

    # Print data regarding percentage of each class of event
    print('Total number of events: {} \n'.format(n_events))
    print('NOTE: some percentages do not reach 100%, as events where no Hadronic W can be matched after removing the b-tagged jets are skipped (all jets are b-tagged)')
    print('\n==================================================================\n')
    print('Cut Criteria')
    print("Number of hadronic W's satisfying mass cut criteria")
    print('Hadronic W, mass: {}, pT: {}, distance: {}'.format(mass_cut, pT_cut, dist_cut))
    print('\n==================================================================\n')

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets, before applying cuts on events matched to 1 jet:")
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_jets[0]/n_events, W_had_jets[0]))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_jets[1]/n_events, W_had_jets[1]))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_jets[2]/n_events, W_had_jets[2]))

    print("Number of events satisfying all hadronic W cut criteria, as a percentage of their respective categories before applying cuts:")
    print('{}% Total Hadronic Ws within cuts, {} events'.format(100.*sum(W_had_total_cuts)/n_events, sum(W_had_total_cuts)))
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/W_had_jets[0], W_had_total_cuts[0], W_had_jets[0]))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/W_had_jets[1], W_had_total_cuts[1], W_had_jets[1]))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/W_had_jets[2], W_had_total_cuts[2], W_had_jets[2]))

    print("Breakdown of total Hadronic Ws matched to 1, 2, and 3 jets after cuts are applied: ")
    print('{}% 1 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[0]/sum(W_had_total_cuts), W_had_total_cuts[0], sum(W_had_total_cuts)))
    print('{}% 2 jet Hadronic W, {} events, out of {}'.format(100.*W_had_total_cuts[1]/sum(W_had_total_cuts), W_had_total_cuts[1], sum(W_had_total_cuts)))
    print('{}% 3 jet Hadronic W, {} events, out of {}\n'.format(100.*W_had_total_cuts[2]/sum(W_had_total_cuts), W_had_total_cuts[2], sum(W_had_total_cuts)))

    print("Number of events satisfying hadronic W mass cut criteria")
    print('{}% total Hadronic Ws, {} events'.format(100.*sum(W_had_m_cuts)/sum(W_had_jets), sum(W_had_m_cuts)))
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[0]/W_had_jets[0], W_had_m_cuts[0]))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_m_cuts[1]/W_had_jets[1], W_had_m_cuts[1]))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_m_cuts[2]/W_had_jets[2], W_had_m_cuts[2]))
    print("Number of events satisfying hadronic W pT cut criteria")
    print('{}% total Hadronic Ws, {} events'.format(100.*sum(W_had_pT_cuts)/sum(W_had_jets), sum(W_had_pT_cuts)))
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[0]/W_had_jets[0], W_had_pT_cuts[0]))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_pT_cuts[1]/W_had_jets[1], W_had_pT_cuts[1]))
    print('{}% 3 jet Hadronic Ws, {} events\n'.format(100.*W_had_pT_cuts[2]/W_had_jets[2], W_had_pT_cuts[2]))
    print("Number of events satisfying hadronic W distance cut criteria")
    print('{}% total Hadronic Ws, {} events'.format(100.*sum(W_had_dist_cuts)/sum(W_had_jets), sum(W_had_dist_cuts)))
    print('{}% 1 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[0]/W_had_jets[0], W_had_dist_cuts[0]))
    print('{}% 2 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[1]/W_had_jets[1], W_had_dist_cuts[1]))
    print('{}% 3 jet Hadronic Ws, {} events'.format(100.*W_had_dist_cuts[2]/W_had_jets[2], W_had_dist_cuts[2]))
    
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