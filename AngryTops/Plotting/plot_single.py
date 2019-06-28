import os, sys
from ROOT import *
import numpy as np
import pickle
from AngryTops.features import *
from AngryTops.Plotting.PlottingHelper import *

################################################################################
# CONSTANTS
training_dir = sys.argv[1]
caption = sys.argv[2]
nbins = int(sys.argv[3])
lowerlim = float(sys.argv[4])
upperlim = float(sys.argv[5])
m_t = 172.5
m_W = 80.4
m_b = 4.95
np.set_printoptions(precision=3, suppress=True, linewidth=250)
model_filename  = "{}/simple_model.h5".format(training_dir)

if __name__=="__main__":
    # LOAD PREDICTIONS
    print("INFO: fitting ttbar decay chain...")
    predictions = np.load('{}/predictions.npz'.format(training_dir))
    true = predictions['true']
    y_fitted = predictions['pred']

    # UNDO NORMALIZATION
    scaler_filename = "{}/scalers.pkl".format(training_dir)
    with open( scaler_filename, "rb" ) as file_scaler:
      jets_scalar = pickle.load(file_scaler)
      lep_scalar = pickle.load(file_scaler)
      output_scalar = pickle.load(file_scaler)
    true = output_scalar.inverse_transform(true)
    y_fitted = output_scalar.inverse_transform(y_fitted)

    # MAKE HISTOGRAM
    h_true = TH1F("true", ";idk ? arb [arb]", nbins, lowerlim, upperlim)
    h_fitted = TH1F("pred", ";idk ? arb [arb]", nbins, lowerlim, upperlim)
    h_corr = TH2F("true", ";True [arb];Fitted [arb]", nbins, lowerlim, upperlim, nbins, lowerlim, upperlim)
    for i in range(true.shape[0]):
        h_true.Fill(true[i], 1.0)
        h_fitted.Fill(y_fitted[i], 1.0)
        h_corr.Fill(true[i], y_fitted[i], 1.0)


    # FORMAT HISTOGRAMS
    xtitle = h_true.GetXaxis().GetTitle()
    ytitle = h_true.GetYaxis().GetTitle()
    for h in [h_true, h_fitted]:
        h.Sumw2()
        h.SetMarkerColor(kRed)
        h.SetLineColor(kRed)
        h.SetMarkerStyle(24)
    Normalize(h_true)
    Normalize(h_fitted)
    SetTH1FStyle( h_true,  color=kGray+2, fillstyle=1001, fillcolor=kGray, linewidth=3)
    SetTH1FStyle( h_fitted, color=kBlack, markersize=0, markerstyle=20, linewidth=3 )

    # DRAW HISTOGRAMS
    c, pad0, pad1 = MakeCanvas()
    pad0.cd()
    gStyle.SetOptTitle(0)
    h_true.Draw("h")
    h_fitted.Draw("h same")
    hmax = 1.5 * max([h_true.GetMaximum(), h_fitted.GetMaximum()])
    h_fitted.SetMaximum(hmax)
    h_true.SetMaximum(hmax)
    h_fitted.SetMinimum(0.)
    h_true.SetMinimum(0.)

    # Legend
    leg = TLegend( 0.20, 0.80, 0.50, 0.90 )
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.05)
    leg.AddEntry( h_true, "MG5+Py8", "f" )
    leg.AddEntry( h_fitted, "fitted", "f" )
    leg.SetY1( leg.GetY1() - 0.05 * leg.GetNRows() )
    leg.Draw()

    # Statistical tests
    KS = h_true.KolmogorovTest( h_fitted )
    X2 = h_true.Chi2Test( h_fitted, "UU NORM CHI2/NDF" ) # UU NORM
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.7, 0.80, "KS test: %.2f" % KS )
    l.DrawLatex( 0.7, 0.75, "#chi^{2}/NDF = %.2f" % X2 )

    # TITLE HISTOGRAM W/ CAPTION
    gPad.RedrawAxis()
    newpad = TPad("newpad","a caption",0.1,0,1,1)
    newpad.SetFillStyle(4000)
    newpad.Draw()
    newpad.cd()
    title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
    title.SetFillColor(16)
    title.SetTextFont(52)
    title.Draw()

    # SAVE AND CLOSE HISTOGRAM
    gPad.RedrawAxis()
    pad1.cd()
    yrange = [0.4, 1.6]
    frame, tot_unc, ratio = DrawRatio(h_fitted, h_true, xtitle, yrange)
    gPad.RedrawAxis()
    c.cd()
    c.SaveAs("{}/histogram.png".format(training_dir))
    pad0.Close()
    pad1.Close()
    c.Close()

    # MAKE CONTOUR PLOT
    c = TCanvas()
    c.cd()
    pad0 = TPad( "pad0","pad0",0, 0,1,1,0,0,0 )
    pad0.SetLeftMargin( 0.18 ) #0.16
    pad0.SetRightMargin( 0.05 )
    pad0.SetBottomMargin( 0.05 )
    pad0.SetTopMargin( 0.07 ) #0.05
    pad0.SetFillColor(0)
    pad0.SetFillStyle(4000)
    pad0.Draw()
    pad0.cd()
    h_corr.Draw("colz")
    corr = h_corr.GetCorrelationFactor()
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextColor(kBlack)
    l.DrawLatex( 0.2, 0.8, "Corr Coeff: %.2f" % corr )
    gPad.RedrawAxis()
    newpad = TPad("newpad","a caption",0.1,0,1,1)
    newpad.SetFillStyle(4000)
    newpad.Draw()
    newpad.cd()
    title = TPaveLabel(0.1,0.94,0.9,0.99,caption)
    title.SetFillColor(16)
    title.SetTextFont(52)
    title.Draw()
    c.cd()
    c.SaveAs("{0}/ContourPlot.png".format(training_dir))
    pad0.Close()
    c.Close()
