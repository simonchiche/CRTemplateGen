#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 02:21:35 2024

@author: chiche
"""

# Modules import
#region Modules 
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import glob
import sys
import pickle
from MainModules.ShowerClass import CreateShowerfromHDF5
from MainModules.PlotConfig import MatplotlibConfig
from scipy.interpolate import interp1d
import scipy
from scipy.interpolate import griddata
from datetime import datetime
from scipy.optimize import curve_fit
from Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE, PlotEradtotThetaScaling, GetMeanEradScalingVsE, PlotMeanEradScalingVsE, PlotEradIceEScalingvsDepth, PlotGroundParticleEVsZenith, PlotEradIcevsZenE, PlotEradIcevsEgroundPart, EradicevsZenE, GetEradvsEgroundPart, GetGroundParticleEnergy, GetEgroundPart_E
#endregion

#region Path definition
SimDir = "CRTemplateGen" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "EnergyScaling" #"DepthScaling" #"ThetaScaling"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/FullDenseDeepCr"
SimpathAll = glob.glob(simpath + "/*")

DataPath="/Users/chiche/Desktop/CRTemplateGen/1_ScalingLaws/EradScaling/Data/EradParamData/"


Path1 =DataPath + "Eradice_faerieParam/"
Path2=DataPath + "GroundPartE/"
Path3 = DataPath + "Eradice_signatures/"


zen_sorted, Erad_ice_16_5, XmaxE16_5_all = np.loadtxt(Path1 + "Eradice_z100_E16_5.txt", unpack = True).T
zen_sorted, Erad_ice_17, XmaxE17_all = np.loadtxt(Path1 + "Eradice_z100_E17.txt", unpack = True).T
zen_sorted, Erad_ice_17_5, XmaxE17_5_all = np.loadtxt(Path1 + "Eradice_z100_E17_5.txt", unpack = True).T

#
XmaxDist16_5, Epart16_5 = np.loadtxt(Path2 + "Epartice_z100_E16_5.txt", unpack = True).T
XmaxDist17, Epart17 = np.loadtxt(Path2 + "Epartice_z100_E17.txt", unpack = True).T
XmaxDist17_5, Epart17_5 = np.loadtxt(Path2 + "Epartice_z100_E17_5.txt", unpack = True).T

plt.scatter(XmaxDist16_5, Epart16_5)
plt.scatter(XmaxDist17, Epart17)
plt.scatter(XmaxDist17_5, Epart17_5)
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm part}^{\rm ground}/E_p$")
plt.show()

### We load the in-ice radiation energy for all sims
Eradice_all = np.loadtxt(Path3 + "Erad_ice_all.txt")
###
#  Primary particle energy and depth cut to build sqrt(Erad_ice) for each energy

seldepth = 3116
selE = 0.0316
def GetEfieldFromErad(Erad, seldepth, selE):
    sel = (Erad[:,4] == seldepth)  & (Erad[:,5] ==selE)
    # We sort the values by increasing zenith angle
    arg = np.argsort(Erad[sel][:,6])
    ZenE_Erad = Erad[sel][:,6][arg]
    
    Efield = np.sqrt(Erad[sel][:,3][arg])

    return Efield

seldepth = 3116
selE = 0.0316
EfieldiceE_16_5 = GetEfieldFromErad(Eradice_all, seldepth, selE)
selE = 0.1
EfieldiceE_17 = GetEfieldFromErad(Eradice_all, seldepth, selE)
selE = 0.316
EfieldiceE_17_5 = GetEfieldFromErad(Eradice_all, seldepth, selE)
#
###


################
# Efield vs Epart
################

### 1) sqrt(Erad_ice)/Ep vs Epart/Ep
from scipy.optimize import curve_fit
def ModelFunc(x, a):
    return a*x
Epartallbins = np.concatenate([Epart16_5[1:], Epart17[1:], Epart17_5[1:]])
Efieldice_all =  np.concatenate([EfieldiceE_16_5[1:]/0.0316,  EfieldiceE_17[1:]/0.1,  EfieldiceE_17_5[1:]/0.316])

popt, pcov = curve_fit(ModelFunc, Epartallbins, Efieldice_all)
Efield_vs_Eground_fit = ModelFunc(Epartallbins, popt[0])

plt.scatter(Epart16_5[1:],  EfieldiceE_16_5[1:]/0.0316)
plt.scatter(Epart17[1:],  EfieldiceE_17[1:]/0.1)
plt.scatter(Epart17_5[1:],  EfieldiceE_17_5[1:]/0.316)
arg = np.argsort(Efield_vs_Eground_fit)
plt.xscale("log")
plt.yscale("log")
plt.plot(Epartallbins[arg], Efield_vs_Eground_fit[arg], c="red")
plt.xlabel(r"$E_{\rm part}^{\rm ground}/E_p$")
plt.ylabel(r"$E_{\rm field}^{\rm ice}/E_p$")
plt.show()

### 2) sqrt(Erad_ice) vs Epart
Epartallbins = np.concatenate([Epart16_5[1:]*0.0316, Epart17[1:]*0.1, Epart17_5[1:]*0.316])
Efieldice_all =  np.concatenate([EfieldiceE_16_5[1:],  EfieldiceE_17[1:],  EfieldiceE_17_5[1:]])

popt, pcov = curve_fit(ModelFunc, Epartallbins, Efieldice_all)
Efield_vs_Eground_fit = ModelFunc(Epartallbins, popt[0])

loga = np.mean(np.log10(Efieldice_all) - np.log10(Epartallbins))
a = 10**loga
fit = a*Epartallbins

plt.scatter(Epart16_5[1:]*0.0316,  EfieldiceE_16_5[1:])
plt.scatter(Epart17[1:]*0.1,  EfieldiceE_17[1:])
plt.scatter(Epart17_5[1:]*0.316,  EfieldiceE_17_5[1:])
plt.xscale("log")
plt.yscale("log")
arg = np.argsort(Efield_vs_Eground_fit)
plt.plot(Epartallbins[arg], fit[arg], c="red")
#plt.plot(Epartallbins[arg], Efield_vs_Eground_fit[arg], c="red")
plt.show()


######## Fitting Erad  instead of Efield

seldepth = 3116
selE = 0.0316
def GetGivenErad(Erad, seldepth, selE):
    sel = (Erad[:,4] == seldepth)  & (Erad[:,5] ==selE)
    # We sort the values by increasing zenith angle
    arg = np.argsort(Erad[sel][:,6])
    ZenE_Erad = Erad[sel][:,6][arg]

    return  Erad[sel][:,3][arg]

seldepth = 3116
selE = 0.0316
EradiceE_16_5 = GetGivenErad(Eradice_all, seldepth, selE)
selE = 0.1
EradiceE_17 = GetGivenErad(Eradice_all, seldepth, selE)
selE = 0.316
EradiceE_17_5 = GetGivenErad(Eradice_all, seldepth, selE)
#

Epartallbins = np.concatenate([Epart16_5[1:]*1e3*0.0316, Epart17[1:]*1e3*0.1, Epart17_5[1:]*1e3*0.316])
Eradice_allE =  np.concatenate([EradiceE_16_5[1:],  EradiceE_17[1:], EradiceE_17_5[1:]])

from scipy.optimize import curve_fit
def ModelFunc(x, loga):
    return loga + 2*np.log10(x)

popt, pcov = curve_fit(ModelFunc, Epartallbins, np.log10(Eradice_allE))
a = 10**popt[0]
fit = a*Epartallbins**2

print(a)

plt.scatter((Epart16_5[1:]*1e3*0.0316)**2,  EradiceE_16_5[1:],  marker = "+", s =70, color="blue", label =r"$E=10^{16.5}\, \rm eV$")
plt.scatter((Epart17[1:]*1e3*0.1)**2,  EradiceE_17[1:],  marker = "+", s =70, color="orange", label =r"$E=10^{17}\, \rm eV$")
plt.scatter((Epart17_5[1:]*1e3*0.316)**2,  EradiceE_17_5[1:],  marker = "+", s =70, color="green", label =r"$E=10^{17.5}\, \rm eV$")
arg = np.argsort(fit)
plt.plot(Epartallbins[arg]**2, fit[arg], c="red", label=r"$2\times 10^{-3}\,(E_{\rm ground}^{\rm part})^2$")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.3)
plt.xlabel(r"$(E_{\rm part}^{\rm ground}/1\, {\rm GeV})^{2}$")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}$ [MeV]")
plt.legend()
plt.savefig("Eradice_vs_EgroundPart.pdf", bbox_inches="tight")
plt.show()
'''
plt.errorbar((Epart16_5[1:]*1e3*0.0316)**2,  EradiceE_16_5[1:], yerr=0.1*EradiceE_16_5[1:],  fmt = "+", markersize =70, color="blue", label =r"$E=10^{16.5}\, \rm eV$")
plt.errorbar((Epart17[1:]*1e3*0.1)**2,  EradiceE_17[1:], yerr=0.1*EradiceE_17[1:],  fmt = "+",markersizes =70, color="orange", label =r"$E=10^{17}\, \rm eV$")
plt.errorbar((Epart17_5[1:]*1e3*0.316)**2,  EradiceE_17_5[1:], yerr=0.1*EradiceE_17_5[1:],  fmt = "+", markersize =70, color="green", label =r"$E=10^{17.5}\, \rm eV$")
arg = np.argsort(fit)
plt.plot(Epartallbins[arg]**2, fit[arg], c="red")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.3)
plt.xlabel(r"$(E_{\rm part}^{\rm ground})^{2}$ [eV$^{2}$]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}$ [eV]")
plt.legend()
#plt.savefig("Eradice_vs_EgroundPart.pdf", bbox_inches="tight")
plt.show()
'''

################
# Ground Particle Fit: EGROUND/EP vs DMAX
################

from scipy.optimize import curve_fit
def ExponentialDecay(x, A, b):
    return A*np.exp(-x/b)

Epartallbins = np.concatenate([Epart16_5, Epart17, Epart17_5])
XmaxDistallbins = np.concatenate([XmaxDist16_5, XmaxDist17, XmaxDist17_5])
args = np.argsort(XmaxDistallbins)
XmaxDistallbins = XmaxDistallbins[args]
Epartallbins = Epartallbins[args]

popt, pcov = curve_fit(ExponentialDecay, XmaxDistallbins, Epartallbins, p0=[0.5, 5000])
ExpDecayfit = ExponentialDecay(XmaxDistallbins, popt[0], popt[1])
plt.scatter(XmaxDistallbins, Epartallbins)
plt.plot(XmaxDistallbins, ExpDecayfit)
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm part}^{\rm ground}/E_p$")
plt.show()



#######################
# ERAD_ICE VS DXMAX FIT
#######################

# DATA LOADING
######

DataPath="/Users/chiche/Desktop/CRTemplateGen/1_ScalingLaws/EradScaling/Data/EradParamData/"
Path1 =DataPath + "Eradice_faerieParam/"
Path2=DataPath + "GroundPartE/"
Path3 = DataPath + "Eradice_signatures/"
### Zenith Bins, In-ice radiation energies and Xmax distances
zen_sorted, Erad_ice_16_5, XmaxE16_5_all = np.loadtxt(Path1 + "Eradice_z100_E16_5.txt", unpack = True).T
zen_sorted, Erad_ice_17, XmaxE17_all = np.loadtxt(Path1 + "Eradice_z100_E17.txt", unpack = True).T
zen_sorted, Erad_ice_17_5, XmaxE17_5_all = np.loadtxt(Path1 + "Eradice_z100_E17_5.txt", unpack = True).T
####

# Distance to Xmax in meters
XmaxE16_5_all, XmaxE17_all, XmaxE17_5_all = 1e3*XmaxE16_5_all, 1e3*XmaxE17_all, 1e3*XmaxE17_5_all
# Normalized raditaion energy
Erad_ice_16_5 = Erad_ice_16_5/(0.0316)**2
Erad_ice_17 = Erad_ice_17/(0.1)**2
Erad_ice_17_5 = Erad_ice_17_5/(0.316)**2

Dxmax_all = np.concatenate([XmaxE16_5_all, XmaxE17_all, XmaxE17_5_all])
Erad_ice_comb = np.concatenate([Erad_ice_16_5, Erad_ice_17,Erad_ice_17_5 ])

### Fitting the values
arg = np.argsort(Dxmax_all)
Erad_ice_comb= Erad_ice_comb[arg]
Dxmax_all = Dxmax_all[arg]

def Erad_Model(x, A, Lambda):

    return A*np.exp(-2*x/Lambda)

#popt, pcov = curve_fit(Erad_Model, Dxmax_all, Erad_ice_comb,  p0=[0.05, 1800])
#Erad_fit = Erad_Model(Dxmax_all, popt[0], popt[1])
arg = np.argsort(Erad_ice_comb)

slope, intercept = np.polyfit(
    Dxmax_all,
    np.log10(Erad_ice_comb),
    1
)

Lambda = -2/(slope*np.log(10))
A = 10**intercept
print(Lambda)

Erad_fit = A*np.exp(-2*Dxmax_all/Lambda)

# Main plots Erad_ice vs Dmax
######

plt.scatter(Dxmax_all, Erad_ice_comb)
plt.plot(Dxmax_all, Erad_fit, color="red")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
#plt.savefig(OutputPath + "EradIceScaling_vs_dmax.pdf", bbox_inches="tight")
plt.show()

# Logscale
plt.scatter(Dxmax_all, Erad_ice_comb)
plt.plot(Dxmax_all, Erad_fit, color="red")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
plt.xscale("log")
plt.yscale("log")
#plt.savefig(OutputPath + "EradIceScaling_vs_dmax.pdf", bbox_inches="tight")
plt.show()

# Energy split
plt.scatter(XmaxE16_5_all, Erad_ice_16_5, label=r"$E_{\rm p}=10^{16.5}$ eV", marker="o", color="blue")
plt.scatter(XmaxE17_all, Erad_ice_17, label=r"$E_{\rm p}=10^{17}$ eV", marker="s", color="goldenrod")
plt.scatter(XmaxE17_5_all, Erad_ice_17_5, label=r"$E_{\rm p}=10^{17.5}$ eV", marker="^", color="firebrick")
plt.plot(Dxmax_all, Erad_fit, color="red", label =r"$\exp{(-2D_{\rm xmax}/\lambda)}$")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
plt.legend()
#plt.xscale("log")
plt.yscale("log")
plt.savefig(OutputPath + "EradIceScaling_vs_dmax_ebins.pdf", bbox_inches="tight")
plt.show()


## Quantifying the precision of the fit
y= Erad_ice_16_5 
yfit = Erad_Model(XmaxE16_5_all, popt[0], popt[1])
reldev = (y - yfit)/y
plt.scatter(XmaxE16_5_all, reldev, label=r"$E_{\rm p}=10^{16.5}$ eV", marker="o", color="#8ecae6")
y= Erad_ice_17 
yfit = Erad_Model(XmaxE17_all, popt[0], popt[1])
reldev = (y - yfit)/y
plt.scatter(XmaxE17_all, reldev, label=r"$E_{\rm p}=10^{17}$ eV", marker="s", color="#ffb703")
y= Erad_ice_17_5 
yfit = Erad_Model(XmaxE17_5_all, popt[0], popt[1])
reldev = (y - yfit)/y
plt.scatter(XmaxE17_5_all, reldev, label=r"$E_{\rm p}=10^{17.5}$ eV", marker="^", color="#90be6d")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
plt.legend()
#plt.xscale("log")
plt.savefig(OutputPath + "EradIceScaling_relerr_vs_dmax_ebins_log.pdf", bbox_inches="tight")
plt.show()



#####################
# Backup
#####################

######
#Ice Zenith Scaling
"""
plt.plot(zen_sorted, np.sqrt(Erad_ice_16_5), label =r"$E_{\rm p}=10^{16.5}$ eV")
plt.plot(zen_sorted, np.sqrt(Erad_ice_17), label =r"$E_{\rm p}=10^{17}$ eV")
plt.plot(zen_sorted, np.sqrt(Erad_ice_17_5), label =r"$E_{\rm p}=10^{17.5}$ eV")
plt.legend()
plt.xlabel(r"Zenith [Deg.]")
plt.ylabel(r"$\sqrt{E_{\rm rad}^{\rm ice}}$ [MeV$^{-1/2}$]")
plt.xscale("log")
plt.yscale("log")
plt.show()
"""


# sqrt(Erad/Ep) vs Eground/Ep
"""

popt, pcov = curve_fit(ModelFunc, Epartallbins, Efieldice_all)
print(popt)
test_func = ModelFunc(Epartallbins, popt[0])

plt.scatter(Epart16_5[1:],  EfieldiceE_16_5[1:]/0.0316)
plt.scatter(Epart17[1:],  EfieldiceE_17[1:]/0.1)
plt.scatter(Epart17_5[1:],  EfieldiceE_17_5[1:]/0.316)
arg = np.argsort(test_func)
plt.xscale("log")
plt.yscale("log")
plt.plot(Epartallbins[arg], test_func[arg], c="red")
plt.xlabel(r"$E_{\rm part}^{\rm ground}/E_p$")
plt.ylabel(r"$E_{\rm field}^{\rm ice}/E_p$")
plt.show()
"""

"""
### Scaling factors for ARA analysis

DataPath="/Users/chiche/Desktop/CRTemplateGen/1_ScalingLaws/EradScaling/Data/"
DxmaxData_p = np.loadtxt(DataPath + "Dxmax_p.txt")
DxmaxData_fe = np.loadtxt(DataPath + "Dxmax_fe.txt")

def GetScalingFactor(DxmaxData_p, DxmaxData_fe):

    fDxmax_p_E18 =  Erad_Model(DxmaxData_p[0], popt[0], popt[1])
    fDxmax_p_E18_5 =  Erad_Model(DxmaxData_p[1], popt[0], popt[1])
    fDxmax_p_E19 =  Erad_Model(DxmaxData_p[2], popt[0], popt[1])
    fDxmax_fe_E18 =  Erad_Model(DxmaxData_fe[0], popt[0], popt[1])
    fDxmax_fe_E18_5 =  Erad_Model(DxmaxData_fe[1], popt[0], popt[1])
    fDxmax_fe_E19 =  Erad_Model(DxmaxData_fe[2], popt[0], popt[1])

    eps_scale_p_E18_5= np.sqrt(10)*np.sqrt(fDxmax_p_E18_5/fDxmax_p_E18)
    eps_scale_p_E19= 10*np.sqrt(fDxmax_p_E19/fDxmax_p_E18)
    eps_scale_fe_E18_5= np.sqrt(10)*np.sqrt(fDxmax_fe_E18_5/fDxmax_fe_E18)
    eps_scale_fe_E19= 10*np.sqrt(fDxmax_fe_E19/fDxmax_fe_E18)

    scale_all = [eps_scale_p_E18_5, eps_scale_p_E19, eps_scale_fe_E18_5, eps_scale_fe_E19]

    return scale_all
scale_all = GetScalingFactor(DxmaxData_p, DxmaxData_fe)
print(scale_all)
"""

