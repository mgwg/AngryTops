from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem, Double
from features import *

def draw_contour(attribute_name, treename='fitted.root'):
    # Create a new canvas, and customize it.
    c1 = TCanvas( 'c1', 'Dynamic Filling Example', 200, 10, 700, 500 )
    c1.SetFillColor( 42 )
    c1.GetFrame().SetFillColor( 21 )
    c1.GetFrame().SetBorderSize( 6 )
    c1.GetFrame().SetBorderMode( -1 )

    # Open tree
    ttree = TChain('nominal', 'nominal')
    ttree.AddFile(treename)

    # Draw and save contour plot
    ttree.Draw("{0}_true:{0}_fitted".format(attribute_name), "", "colz")
    cl.SaveAs("ContourPlots/{}".format(attribute_name))

if __name__=="__main__":
    draw_contour("W_had_px")
