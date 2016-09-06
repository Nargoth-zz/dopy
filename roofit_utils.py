from .timing import timing

@timing
def read_roodataset_from_tree(tchain, data_tot, observables, print_status=False):
  obs_to_values = {}
  iter = observables.createIterator()
  var = iter.Next()
  while var:
    obs_to_values[var.GetName()] = [[None, -2000],
                                  [var.getMin(), var.getMax()]]
    var = iter.Next()

  step = 5
  nextstep = 0
  tchain.GetEntries()
  for i in range(tchain.GetEntriesFast()):
    tchain.GetEntry(i)
    if (i/tchain.GetEntriesFast() * 100) > nextstep and print_status:
      print(i/tchain.GetEntriesFast() * 100)
      data_tot.Print()
      nextstep += step
    iter = observables.createIterator()
    var = iter.Next()
    skip_entry = False
    while var:
      obs_to_values[var.GetName()][0][0] = tchain.__getattr__(var.GetName())
      if hasattr(obs_to_values[var.GetName()][0][0], "__len__"): #multi pvs!
        obs_to_values[var.GetName()][0][1] = obs_to_values[var.GetName()][0][0][0]
      else:
        obs_to_values[var.GetName()][0][1] = obs_to_values[var.GetName()][0][0]
      if(obs_to_values[var.GetName()][0][1] < obs_to_values[var.GetName()][1][0] or 
         obs_to_values[var.GetName()][0][1] > obs_to_values[var.GetName()][1][1]):
        skip_entry=True
        break;
      var = iter.Next()
    if not skip_entry:
      iter = observables.createIterator()
      var = iter.Next()
      while var:
        observables.find(var.GetName()).setVal(obs_to_values[var.GetName()][0][1])
        var = iter.Next()
      data_tot.add(observables)

def build_tchain_from_files(filelist, treename, cutstring):
  from ROOT import TChain
  tchain = None
  tchain_init = TChain()
  for file in filelist:
    tchain_init.Add(file + '/' + treename)
  tchain = tchain_init
  if cutstring:
    print("WARNING Cutting the TChain with" + cutstring + "will be slower than just creating it")
    tchain = tchain_init.CopyTree()
  return tchain

def get_luminosity(list_of_files):
  lumituple = build_tchain_from_files(list_of_files,
                                      "GetIntegratedLuminosity/LumiTuple",
                                      "")
  lumituple_nentries = lumituple.GetEntries()
  lumi = 0.
  for i in range(0,lumituple_nentries):
    lumituple.GetEntry(i)
    lumi += lumituple.GetLeaf("IntegratedLuminosity").GetValue()
  return lumi

import ROOT
def build_gauss(obs, postfix):
  from ROOT import RooGaussian
  from ROOT import RooRealVar
  par_gauss_m0    = RooRealVar("par_gauss_m0"    + postfix,    "par_gauss_m0" + postfix,    5279, 5000., 5400.,)
  par_gauss_sigma = RooRealVar("par_gauss_sigma" + postfix, "par_gauss_sigma" + postfix, 10, 1, 300)
  pdf_mass_gauss = RooGaussian("pdf_mass_gauss" + postfix, "pdf_mass_gauss" + postfix, obs, par_gauss_m0, par_gauss_sigma)

  ROOT.SetOwnership( par_gauss_m0, False )
  ROOT.SetOwnership( par_gauss_sigma, False )
  ROOT.SetOwnership( pdf_mass_gauss, False)
    
  return pdf_mass_gauss

def build_ipatia(obs, postfix):
  from ROOT import gSystem
  gSystem.Load("libRooFit.so")
  gSystem.Load("~/storage03/repos/bd2jpsieeks/external/CustomShapes/libKll")
  from ROOT import RooIpatia2
  from ROOT import RooRealVar
  par_ipatia_zeta  = RooRealVar("par_ipatia_zeta"  + postfix,  "par_ipatia_zeta"  + postfix, 2.3, 0.3, 5.3)
  par_ipatia_fb    = RooRealVar("par_ipatia_fb"    + postfix,    "par_ipatia_fb"  + postfix, 0., 0., 0.)
  par_ipatia_l     = RooRealVar("par_ipatia_l"     + postfix,     "par_ipatia_l"  + postfix, 1., 1., 1.)
  par_ipatia_m     = RooRealVar("par_ipatia_m"     + postfix,     "par_ipatia_m"  + postfix, 5279, 5000., 5400.,)
  par_ipatia_sigma = RooRealVar("par_ipatia_sigma" + postfix, "par_ipatia_sigma"  + postfix, 10, 2., 100.)
  par_ipatia_a1    = RooRealVar("par_ipatia_a1"    + postfix,     "par_ipatia_a1" + postfix, 1., 0.01, 4)
  par_ipatia_a2    = RooRealVar("par_ipatia_a2"    + postfix,     "par_ipatia_a2" + postfix, 1., 0.01, 4)
  par_ipatia_n1    = RooRealVar("par_ipatia_n1"    + postfix,     "par_ipatia_n1" + postfix, 1., 0.00001, 150.)
  par_ipatia_n2    = RooRealVar("par_ipatia_n2"    + postfix,     "par_ipatia_n2" + postfix, 1., 0.00001, 150.)

  # template to have uniform const parameter initialization (works when input file is given)
  # par_ipatia_zeta  = RooRealVar("par_ipatia_zeta"  + postfix,  "par_ipatia_zeta"  + postfix, 0.)
  # par_ipatia_fb    = RooRealVar("par_ipatia_fb"    + postfix,    "par_ipatia_fb"  + postfix, 0.)
  # par_ipatia_l     = RooRealVar("par_ipatia_l"     + postfix,     "par_ipatia_l"  + postfix, 0.)
  # par_ipatia_m     = RooRealVar("par_ipatia_m"     + postfix,     "par_ipatia_m"  + postfix, 0.)
  # par_ipatia_sigma = RooRealVar("par_ipatia_sigma" + postfix, "par_ipatia_sigma"  + postfix, 0.)
  # par_ipatia_a1    = RooRealVar("par_ipatia_a1"    + postfix,     "par_ipatia_a1" + postfix, 0.)
  # par_ipatia_a2    = RooRealVar("par_ipatia_a2"    + postfix,     "par_ipatia_a2" + postfix, 0.)
  # par_ipatia_n1    = RooRealVar("par_ipatia_n1"    + postfix,     "par_ipatia_n1" + postfix, 0.)
  # par_ipatia_n2    = RooRealVar("par_ipatia_n2"    + postfix,     "par_ipatia_n2" + postfix, 0.)

  pdf_mass_ipatia  = ROOT.RooIpatia2("pdf_mass_ipatia", "pdf_mass_ipatia", obs, par_ipatia_l, par_ipatia_zeta, par_ipatia_fb,
                                                 par_ipatia_sigma, par_ipatia_m, par_ipatia_a1, par_ipatia_n1,
                                                 par_ipatia_a2, par_ipatia_n2)
  ROOT.SetOwnership( par_ipatia_zeta, False )
  ROOT.SetOwnership( par_ipatia_fb, False )
  ROOT.SetOwnership( par_ipatia_l, False )
  ROOT.SetOwnership( par_ipatia_m, False )
  ROOT.SetOwnership( par_ipatia_sigma, False )
  ROOT.SetOwnership( par_ipatia_a1, False )
  ROOT.SetOwnership( par_ipatia_n1, False )
  ROOT.SetOwnership( par_ipatia_a2, False )
  ROOT.SetOwnership( par_ipatia_n2, False )
  ROOT.SetOwnership( pdf_mass_ipatia, False)

  return pdf_mass_ipatia

def build_simple_data_model(mass, postfix, brem_category):
  from ROOT import RooRealVar
  from ROOT import RooExponential
  from ROOT import RooExtendPdf
  from ROOT import RooAddPdf
  from ROOT import RooArgList
  exp_a = RooRealVar("exp_a" + postfix, "exp_a" + postfix, -0.001, -0.01, -0.00001)
  exp   = RooExponential("exp" + postfix, "exp" + postfix, mass, exp_a)
  bkg_pdf = exp

  sig_pdf = build_gauss(mass, postfix)

  ROOT.SetOwnership( exp_a, False )
  ROOT.SetOwnership( bkg_pdf, False )
  ROOT.SetOwnership( sig_pdf, False )

  sig_yield = RooRealVar("sig_yield" + postfix, "sig_yield" + postfix, 100, 0, 1000000)
  bkg_yield = RooRealVar("bkg_yield" + postfix, "bkg_yield" + postfix, 5000, 0, 1000000)
  ROOT.SetOwnership( sig_yield, False )
  ROOT.SetOwnership( bkg_yield, False)

  sig_pdf_ext = RooExtendPdf("sig_pdf_ext" + postfix, "sig_pdf_ext" + postfix, sig_pdf, sig_yield)
  bkg_pdf_ext = RooExtendPdf("bkg_pdf_ext" + postfix, "bkg_pdf_ext" + postfix, bkg_pdf, bkg_yield)
  ROOT.SetOwnership( sig_pdf_ext, False )
  ROOT.SetOwnership( bkg_pdf_ext, False)

  data_model = RooAddPdf("model" + postfix, "model" + postfix, RooArgList(sig_pdf_ext, bkg_pdf_ext))
  ROOT.SetOwnership( data_model, False )
  return data_model


def build_ipatia_data_model(mass, postfix, brem_category):
  from ROOT import RooRealVar
  from ROOT import RooExponential
  from ROOT import RooExtendPdf
  from ROOT import RooAddPdf
  from ROOT import RooArgList
  exp_a = RooRealVar("exp_a" + postfix, "exp_a" + postfix, -0.001, -0.01, -0.00001)
  exp   = RooExponential("exp" + postfix, "exp" + postfix, mass, exp_a)
  bkg_pdf = exp

  sig_pdf = build_ipatia(mass, postfix)

  ROOT.SetOwnership( exp_a, False )
  ROOT.SetOwnership( bkg_pdf, False )
  ROOT.SetOwnership( sig_pdf, False )

  sig_yield = RooRealVar("sig_yield" + postfix, "sig_yield" + postfix, 100, 0, 1000000)
  bkg_yield = RooRealVar("bkg_yield" + postfix, "bkg_yield" + postfix, 5000, 0, 1000000)
  ROOT.SetOwnership( sig_yield, False )
  ROOT.SetOwnership( bkg_yield, False)

  sig_pdf_ext = RooExtendPdf("sig_pdf_ext" + postfix, "sig_pdf_ext" + postfix, sig_pdf, sig_yield)
  bkg_pdf_ext = RooExtendPdf("bkg_pdf_ext" + postfix, "bkg_pdf_ext" + postfix, bkg_pdf, bkg_yield)
  ROOT.SetOwnership( sig_pdf_ext, False )
  ROOT.SetOwnership( bkg_pdf_ext, False)

  data_model = RooAddPdf("model" + postfix, "model" + postfix, RooArgList(sig_pdf_ext, bkg_pdf_ext))
  ROOT.SetOwnership( data_model, False )
  return data_model

def build_ipatia_kstar_data_model(mass, postfix, brem_category):
  from ROOT import RooRealVar
  from ROOT import RooExponential
  from ROOT import RooExtendPdf
  from ROOT import RooAddPdf
  from ROOT import RooArgList
  exp_a = RooRealVar("exp_a" + postfix, "exp_a" + postfix, -0.001, -0.01, -0.00001)
  exp   = RooExponential("exp" + postfix, "exp" + postfix, mass, exp_a)
  bkg_pdf = exp

  sig_pdf = build_ipatia(mass, postfix)
  kstar_pdf = build_gauss(mass, "_Kstar" + postfix)

  ROOT.SetOwnership( exp_a, False )
  ROOT.SetOwnership( bkg_pdf, False )
  ROOT.SetOwnership( sig_pdf, False )
  ROOT.SetOwnership( kstar_pdf, False )

  sig_yield = RooRealVar("sig_yield" + postfix, "sig_yield" + postfix, 100, 0, 1000000)
  bkg_yield = RooRealVar("bkg_yield" + postfix, "bkg_yield" + postfix, 5000, 0, 1000000)
  kstar_yield = RooRealVar("kstar_yield" + postfix, "kstar_yield" + postfix, 100, 0, 1000000)
  ROOT.SetOwnership( sig_yield, False )
  ROOT.SetOwnership( bkg_yield, False)
  ROOT.SetOwnership( kstar_yield, False)

  sig_pdf_ext = RooExtendPdf("sig_pdf_ext" + postfix, "sig_pdf_ext" + postfix, sig_pdf, sig_yield)
  bkg_pdf_ext = RooExtendPdf("bkg_pdf_ext" + postfix, "bkg_pdf_ext" + postfix, bkg_pdf, bkg_yield)
  kstar_pdf_ext = RooExtendPdf("kstar_pdf_ext" + postfix, "kstar_pdf_ext" + postfix, kstar_pdf, kstar_yield)
  ROOT.SetOwnership( sig_pdf_ext, False )
  ROOT.SetOwnership( bkg_pdf_ext, False)
  ROOT.SetOwnership( kstar_pdf_ext, False)

  data_model = RooAddPdf("model" + postfix, "model" + postfix, RooArgList(sig_pdf_ext, bkg_pdf_ext, kstar_pdf_ext))
  ROOT.SetOwnership( data_model, False )
  return data_model
