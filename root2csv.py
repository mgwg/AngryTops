#!/usr/bin/env python
import os, sys
import csv

from ROOT import *
from helper_functions import MakeInput, RotateEvent
import numpy as np

gROOT.SetBatch(True)

from features import *

###############################
# CONSTANTS

GeV = 1e3
TeV = 1e6
rng = TRandom3()

# Artificially increase training data size by 5 by rotating events differently 5 different ways
n_data_aug = 5

# What is this?
n_evt_max = -1
if len(sys.argv) > 2: n_evt_max = int( sys.argv[2] )

###############################
# BUILDING OUTPUTFILE

# List of filenames
filelistname = sys.argv[1]

# Output filename
outfilename = filelistname.split("/")[-1]
outfilename = "csv/topreco." + outfilename.replace(".txt", ".%s.csv" % ( syst ) )
outfile = open( outfilename, "wt" )
csvwriter = csv.writer( outfile )
print ("INFO: output file:", outfilename)

###############################
# BUILDING OUTPUTFILE

# Not entirely sure how TCHAIN works, but I am guessing the truth/nominal input determines which file it loads in?
tree = TChain("Delphes", "Delphes")
f = open( filelistname, 'r' )
for fname in f.readlines():
   fname = fname.strip()
   tree.AddFile( fname )

# Switch on only useful branches. We are doing Muon only right now.
# branches_active = [
#     "Event.Number", "Event.ProcessID", "Event.Weight", "Vertex_size",
#     "Muon.PT", "Muon.Eta", "Muon.Phi", "MissingET.MET", "MissingET.Phi"
#     "Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.T", "Jet.Mass",
#     "GenJet.PT", "GenJet.Eta", "GenJet.Phi", "GenJet.T", "GenJet.Mass",
#     "Particle.PID",
# ]
# tree.SetBranchStatus("*", 0)
# for branch in branches_active:
#     tree.SetBranchStatus(branch, 1)

n_entries = tree.GetEntries()
print("INFO: entries found:", n_entries)

###############################
# LOOPING THROUGH EVENTS

# Cap on number of reconstructed events.
if n_evt_max > 0: n_entries = min( [ n_evt_max, n_entries ] )
print("INFO: looping over %i reco-level events" % n_entries)
print("INFO: using data augmentation: rotateZ %ix" % n_data_aug)

# Number of events which are actually copied over
n_good = 0

# Looping through the reconstructed entries
for ientry in range(n_entries_reco):
    # Withdraw next event
    tree_reco.GetEntry(ientry)

    # Printing how far along in the loop we are
    if ( n_entries_reco < 10 ) or ( (ientry+1) % int(float(n_entries_reco)/10.)  == 0 ):
        perc = 100. * ientry / float(n_entries_reco)
        print("INFO: Event %-9i  (%3.0f %%)" % ( ientry, perc ))

    # Number of muons, leptons, jets and bjets (bjet_n set later)
    #el_n    = len(tree_reco.el_pt)
    mu_n    = len(tree_reco.mu_pt)
    #lep_n   = el_n + mu_n
    jets_n  = len(tree_reco.jet_pt)
    bjets_n = 0

    # If more than one lepton of less than 4 jets, cut
    if mu_n != 1: continue
    if jets_n < 4: continue

    # Muon vector. Will have to manually add electron energy as a constant, since not contained in tree information
    lep = TLorentzVector()
    lep.SetPtEtaPhiE( tree_reco.mu_pt[0]/GeV, tree_reco.mu_eta[0], tree_reco.mu_phi[0], tree_reco.mu_e[0]/GeV )

    # Missing Energy values
    met_met = tree_reco.met_met/GeV
    met_phi = tree_reco.met_phi

    # Append jets, check prob of being a bjet, and update bjet number
    # This is what will be fed into the RNN
    jets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break

        jets += [ TLorentzVector() ]
        j = jets[-1]
        j.index = i
        j.SetPtEtaPhiE( tree_reco.jet_pt[i], tree_reco.jet_eta[i], tree_reco.jet_phi[i], tree_reco.jet_e[i] )
        j.mv2c10 = tree_reco.jet_mv2c10[i]
        if j.mv2c10 > 0.83: bjets_n += 1

    # sort by b-tagging weight?
#    jets.sort( key=lambda jet: jet.mv2c10, reverse=True )

    # Build output data we are trying to predict with RNN

    t_had = TLorentzVector()
    t_lep = TLorentzVector()
    W_had = TLorentzVector()
    W_lep = TLorentzVector()
    b_had = TLorentzVector()
    b_lep = TLorentzVector()

    t_had.SetPtEtaPhiM( tree_parton.MC_thad_afterFSR_pt,
                        tree_parton.MC_thad_afterFSR_eta,
                        tree_parton.MC_thad_afterFSR_phi,
                        tree_parton.MC_thad_afterFSR_m )

    W_had.SetPtEtaPhiM( tree_parton.MC_W_from_thad_pt,
                        tree_parton.MC_W_from_thad_eta,
                        tree_parton.MC_W_from_thad_phi,
                        tree_parton.MC_W_from_thad_m )

    t_lep.SetPtEtaPhiM( tree_parton.MC_tlep_afterFSR_pt,
                        tree_parton.MC_tlep_afterFSR_eta,
                        tree_parton.MC_tlep_afterFSR_phi,
                        tree_parton.MC_tlep_afterFSR_m )

    W_lep.SetPtEtaPhiM( tree_parton.MC_W_from_tlep_pt,
                        tree_parton.MC_W_from_tlep_eta,
                        tree_parton.MC_W_from_tlep_phi,
                        tree_parton.MC_W_from_tlep_m )

    # b-quarks ignored...
    if abs( tree_parton.MC_Wdecay1_from_t_pdgId ) < 10:
        # t = t_had, tbar = t_lep
        b_had.SetPtEtaPhiM( tree_parton.MC_b_from_t_pt,
                            tree_parton.MC_b_from_t_eta,
                            tree_parton.MC_b_from_t_phi,
                            tree_parton.MC_b_from_t_m )
        b_lep.SetPtEtaPhiM( tree_parton.MC_b_from_tbar_pt,
                            tree_parton.MC_b_from_tbar_eta,
                            tree_parton.MC_b_from_tbar_phi,
                            tree_parton.MC_b_from_tbar_m )
    else:
        # t = t_lep, tbar = t_had
        b_had.SetPtEtaPhiM( tree_parton.MC_b_from_tbar_pt,
                            tree_parton.MC_b_from_tbar_eta,
                            tree_parton.MC_b_from_tbar_phi,
                            tree_parton.MC_b_from_tbar_m )
        b_lep.SetPtEtaPhiM( tree_parton.MC_b_from_t_pt,
                            tree_parton.MC_b_from_t_eta,
                            tree_parton.MC_b_from_t_phi,
                            tree_parton.MC_b_from_t_m )

    # sanity checks
    if (t_had.Pz() == 0.) or (t_had.M() != t_had.M()): continue
    if (t_lep.Pz() == 0.) or (t_lep.M() != t_lep.M()): continue

    n_good += 1

    phi = 0.
    for n in range(n_data_aug+1):
       # rotate f.s.o.
       lep.RotateZ( phi )

       met_phi = TVector2.Phi_mpi_pi( met_phi + phi )

       for j in jets: j.RotateZ(phi)

       W_had.RotateZ( phi )
       b_had.RotateZ( phi )
       t_had.RotateZ( phi )
       W_lep.RotateZ( phi )
       b_lep.RotateZ( phi )
       t_lep.RotateZ( phi )

       # make event wrapper
       sjets, target_W_had, target_b_had, target_t_had, target_W_lep, target_b_lep, target_t_lep = MakeInput( jets, W_had, b_had, t_had, W_lep, b_lep, t_lep )

       # write out
       csvwriter.writerow( (
          "%i" % tree_reco.runNumber, "%i" % tree_reco.eventNumber, "%.3f" % weight, "%i" % jets_n, "%i" % bjets_n,
          "%.3f" % lep.Px(),     "%.3f" % lep.Py(),     "%.3f" % lep.Pz(),     "%.3f" % lep.E(),      "%.3f" % met_met,      "%.3f" % met_phi,
          "%.3f" % sjets[0][0],  "%.3f" % sjets[0][1],  "%.3f" % sjets[0][2],  "%.3f" % sjets[0][3],  "%.3f" % sjets[0][4],  "%.3f" % sjets[0][5],
          "%.3f" % sjets[1][0],  "%.3f" % sjets[1][1],  "%.3f" % sjets[1][2],  "%.3f" % sjets[1][3],  "%.3f" % sjets[1][4],  "%.3f" % sjets[1][5],
          "%.3f" % sjets[2][0],  "%.3f" % sjets[2][1],  "%.3f" % sjets[2][2],  "%.3f" % sjets[2][3],  "%.3f" % sjets[2][4],  "%.3f" % sjets[2][5],
          "%.3f" % sjets[3][0],  "%.3f" % sjets[3][1],  "%.3f" % sjets[3][2],  "%.3f" % sjets[3][3],  "%.3f" % sjets[3][4],  "%.3f" % sjets[3][5],
          "%.3f" % sjets[4][0],  "%.3f" % sjets[4][1],  "%.3f" % sjets[4][2],  "%.3f" % sjets[4][3],  "%.3f" % sjets[4][4],  "%.3f" % sjets[4][5],
          "%.3f" % target_W_had[0], "%.3f" % target_W_had[1], "%.3f" % target_W_had[2], "%.3f" % target_W_had[3], "%.3f" % target_W_had[4],
          "%.3f" % target_W_lep[0], "%.3f" % target_W_lep[1], "%.3f" % target_W_lep[2], "%.3f" % target_W_lep[3], "%.3f" % target_W_lep[4],
          "%.3f" % target_b_had[0], "%.3f" % target_b_had[1], "%.3f" % target_b_had[2], "%.3f" % target_b_had[3], "%.3f" % target_b_had[4],
          "%.3f" % target_b_lep[0], "%.3f" % target_b_lep[1], "%.3f" % target_b_lep[2], "%.3f" % target_b_lep[3], "%.3f" % target_b_lep[4],
          "%.3f" % target_t_had[0], "%.3f" % target_t_had[1], "%.3f" % target_t_had[2], "%.3f" % target_t_had[3], "%.3f" % target_t_had[4],
          "%.3f" % target_t_lep[0], "%.3f" % target_t_lep[1], "%.3f" % target_t_lep[2], "%.3f" % target_t_lep[3], "%.3f" % target_t_lep[4]
       ) )

       phi = rng.Uniform( -TMath.Pi(), TMath.Pi() )



outfile.close()

f_good = 100. * n_good / n_entries_reco
print "INFO: output file:", outfilename
print "INFO: %i entries written (%.2f %%)" % ( n_good, f_good)
