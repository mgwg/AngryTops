#!/usr/bin/env python
import os, sys
import csv

import ROOT
from ROOT import TLorentzVector
from helper_functions import *
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
    tree.GetEntry(ientry)

    # Printing how far along in the loop we are
    if (n_entries_reco < 10) or ((ientry+1) % int(float(n_entries)/10.) == 0):
        perc = 100. * ientry / float(n_entries)
        print("INFO: Event %-9i  (%3.0f %%)" % (ientry, perc))

    # Number of muons, leptons, jets and bjets (bjet_n set later)
    # For now, I am cutting out reactions with electrons, or more than two
    mu_n = tree.GetLeaf("Muon.PT").GetLen()
    jets_n  = tree.GetLeaf("Jet.PT").GetLen()
    bjets_n = 0

    # If more than one lepton of less than 4 jets, cut
    if mu_n != 1: continue
    if jets_n < 4: continue

    # Muon vector. Replaced E w/ T
    lep = TLorentzVector()
    lep.SetPtEtaPhiE(
    lep.tree.GetLeaf("Muon.PT").GetValue(0)/GeV,
    tree.GetLeaf("Muon.Eta").GetValue(0),
    tree.GetLeaf("Muon.Phi").GetValue(0),
    tree.GetLeaf("Muon.T").GetValue(0)/GeV)

    # Missing Energy values
    met_met = tree.GetLeaf("MissingET.MET").GetValue() / GeV
    met_phi = tree.GetLeaf("MissingET.Phi").GetValue()

    # Append jets, check prob of being a bjet, and update bjet number
    # This is what will be fed into the RNN
    # Replaced the mv2c10 value with the bjet tag value, as that is what is
    # recoreded by Delphes
    jets = []
    for i in range(jets_n):
        if i >= n_jets_per_event: break

        jets += [ TLorentzVector() ]
        j = jets[-1]
        j.index = i
        j.SetPtEtaPhiE(
        tree.GetLeaf("Jet.PT").GetValue(i),
        tree.GetLeaf("Jet.Eta").GetValue(i),
        tree.GetLeaf("Jet.Phi").GetValue(i),
        tree.GetLeaf("Jet.Mass").GetValue(i))
        j.btag = tree.GetLeaf("Jet.BTag").GetValue(i)
        if tree.GetLeaf("Jet.BTag").GetValue(i) > 0.0:
            bjets_n += 1
    # sort by b-tagging weight?
#    jets.sort( key=lambda jet: jet.mv2c10, reverse=True )

    # Build output data we are trying to predict with RNN
    try:
        indices = GetIndices(tree, ienty)
    catch Exception as e:
        continue

    t_had = TLorentzVector()
    t_lep = TLorentzVector()
    W_had = TLorentzVector()
    W_lep = TLorentzVector()
    b_had = TLorentzVector()
    b_lep = TLorentzVector()

    t_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_had'])
                        )

    W_had.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_had']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_had'])
                        )

    t_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['t_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['t_lep']))

    W_lep.SetPtEtaPhiM( tree.GetLeaf("Particle.PT").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Eta").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Phi").GetValue(indices['W_lep']),
                        tree.GetLeaf("Particle.Mass").GetValue(indices['W_lep']))

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
