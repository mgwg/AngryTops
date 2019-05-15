# USE THIS SCRIPT TO FIND LOCATIONS OF PARTICLES IN THE ROOT tree
import ROOT
from ROOT import TLorentzVector
from array import array
import numpy as np

GeV = 1e3
TeV = 1e6

def TraverseSelfDecay(tree, entry, index):
    """Returns the index for the last self decay of the particle
    @ Parameters
    tree: TChain Object containing tree
    entry: Event number
    index: Index for the particle that we are to self decay
    @==========================================================
    @ Return
    The leaf index referring to the final stage of self
    """
    tree.GetEntry(entry)
    pids = tree.GetLeaf("Particle.PID")
    d1 = tree.GetLeaf("Particle.D1")
    pid = pids.GetValue(index)
    new_index = int(index)
    # Loop through children
    flag = True
    while flag:
        id = pids.GetValue(int(d1.GetValue(new_index)))
        if id == pid:
            new_index = int(d1.GetValue(new_index))
        else:
            flag = False
    return new_index


def GetParticleIndex(tree, entry, pid):
    """For a specified tree and event entry, return the index referring to
    particle with PDGID pid. NOTE: This is meant to be used for t, tbar, b,
    bbar, W and Wbar particles in the semileptonic ttbar decay, where there is
    only one of each particle for each reaction. Need to modify if particles are
    not unique.
    @==========================================================
    @ Parameters
    tree: TChain Object containing tree
    entry: Event number
    pid: PDGID for the particle we are looking for
    @==========================================================
    @ Return
    The leaf index referring to the desired particle, or -1 if the PID is not
    in the event
    """
    tree.GetEntry(entry)
    pids = tree.GetLeaf("Particle.PID")
    d1 = tree.GetLeaf("Particle.D1")
    index = -1
    # Loop through pids, update index
    for i in range(pids.GetLen()):
        value = pids.GetValue(i)
        if value == pid:
            if index == -1:
                index = i
            else:
                assert d1.GetValue(index) == i, "Particle not unique"
                index = i
    return index

def ClassifyTopQuark(tree, entry, t_indices):
    """
    Make sorted array containing indices for t, tbar, W, and b quarks
    @==========================================================
    @ Parameters
    tree: TChain Object containing tree
    entry: Event number
    t_indices: The indices for the t quarks
    @==========================================================
    @ Return
    Dict of indices with the following keys:
    [t_hadronic, W_hadronic, b_hadronic, t_leptonic, W_leptonic, b_leptonic]
    """
    indices = {}
    tree.GetEntry(entry)
    pids = tree.GetLeaf("Particle.PID")
    d1 = tree.GetLeaf("Particle.D1")
    d2 = tree.GetLeaf("Particle.D2")
    # Check just the first index to save time
    for index in t_indices:
        # One of these will be the W quark, the other will be the b quark.
        d_index1 = TraverseSelfDecay(tree, entry, int(d1.GetValue(index)))
        d_index2 = TraverseSelfDecay(tree, entry, int(d2.GegtValue(index)))
        assert np.abs(pids.GetValue(d_index1)) == 24 \
                        or np.abs(pids.GetValue(d_index2)) == 24, "No W quark found in decay of top quark"
        # Find the children for the W quark
        if np.abs(pids[d_index1]) == 24:
            W_index, b_index = d_index1, d_index2
        else:
            W_index, b_index = d_index2, d_index1
        # Decay constituents of the W particle
        child1 = pids.GetValue(int(d1.GetValue(W_index)))
        child2 = pids.GetValue(int(d2.GetValue(W_index)))
        # Check for leptonic and hadronic decay of W boson
        if np.min(np.abs(child1), np.abs(child2)) > 10:
            assert "t_lep" not in t_indices.keys(), "Two leptonic top quarks"
            indices["t_lep"] = t_indices[index]
            indices["W_lep"] = W_index
            indices["b_lep"] = b_index
        elif np.max(np.abs(child1), np.abs(child2)) < 10:
            assert "t_had" not in t_indices.keys(), "Two hadronic top quarks"
            indices["t_had"] = t_indices[index]
            indices["W_had"] = W_index
            indices["b_had"] = b_index
        else:
            raise Exception("Not Valid Decay for W quark")
    return indices

def GetIndices(tree, entry):
    """
    Return a dictionary containing the leaf indices for t_had, t_lep, W_had,
    W_lep, b_had, b_quark. Wrapper class for ClassifyTopQuark, GetParticleIndex
    and TraverseSelfDecay.
    @==========================================================
    @ Parameters
    tree: TChain Object containing tree
    entry: Event number
    @==========================================================
    @ Return
    Dict of indices with the following keys:
    [t_hadronic, W_hadronic, b_hadronic, t_leptonic, W_leptonic, b_leptonic]
    """
    t_index = [GetParticleIndex(tree, entry, 6), GetParticleIndex(tree, entry, -6)]
    return ClassifyTopQuark(tree, entry, t_index)
