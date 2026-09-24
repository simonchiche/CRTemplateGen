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

#plt.plot(zen_sorted, XmaxE16_5_all*1e3)
#plt.plot(XmaxDist16_5[1:])

#plt.plot(XmaxE17_all*1e3)
#plt.plot(XmaxDist17[1:])

#plt.plot(XmaxE17_5_all*1e3)
#plt.plot(XmaxDist17_5[1:])

plt.plot(zen_sorted, np.sqrt(Erad_ice_16_5), label =r"$E_{\rm p}=10^{16.5}$ eV")
plt.plot(zen_sorted, np.sqrt(Erad_ice_17), label =r"$E_{\rm p}=10^{17}$ eV")
plt.plot(zen_sorted, np.sqrt(Erad_ice_17_5), label =r"$E_{\rm p}=10^{17.5}$ eV")
plt.legend()
plt.xlabel(r"Zenith [Deg.]")
plt.ylabel(r"$\sqrt{E_{\rm rad}^{\rm ice}}$ [MeV$^{-1/2}$]")
plt.show()



XmaxDist16_5, Epart16_5 = np.loadtxt(Path2 + "Epartice_z100_E16_5.txt", unpack = True).T
XmaxDist17, Epart17 = np.loadtxt(Path2 + "Epartice_z100_E17.txt", unpack = True).T
XmaxDist17_5, Epart17_5 = np.loadtxt(Path2 + "Epartice_z100_E17_5.txt", unpack = True).T

plt.scatter(XmaxDist16_5, Epart16_5)
plt.scatter(XmaxDist17, Epart17)
plt.scatter(XmaxDist17_5, Epart17_5)
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm part}^{\rm ground}/E_p$")
plt.show()


Eradice_all = np.loadtxt(Path3 + "Erad_ice_all.txt")


sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.0316)
arg = np.argsort(Eradice_all[sel][:,6])
ZenE_Erad = Eradice_all[sel][:,6][arg]
EfieldiceE_16_5 = np.sqrt(Eradice_all[sel][:,3][arg])

sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.1)
arg = np.argsort(Eradice_all[sel][:,6])
ZenE_Erad = Eradice_all[sel][:,6][arg]
EfieldiceE_17 = np.sqrt(Eradice_all[sel][:,3][arg])

sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.316)
arg = np.argsort(Eradice_all[sel][:,6])
ZenE_Erad = Eradice_all[sel][:,6][arg]
EfieldiceE_17_5 = np.sqrt(Eradice_all[sel][:,3][arg])



################
# Efield vs Epart
################

plt.plot(Epart16_5[1:],  EfieldiceE_16_5[1:]/0.0316)
plt.plot(Epart17[1:],  EfieldiceE_17[1:]/0.1)
plt.plot(Epart17_5[1:],  EfieldiceE_17_5[1:]/0.316)
plt.show()

from scipy.optimize import curve_fit
def ModelFunc(x, a):
    return a*x

Epartallbins = np.concatenate([Epart16_5[1:], Epart17[1:], Epart17_5[1:]])
Efieldice_all =  np.concatenate([EfieldiceE_16_5[1:]/0.0316,  EfieldiceE_17[1:]/0.1,  EfieldiceE_17_5[1:]/0.316])

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

plt.scatter(Epart16_5[1:]*0.0316,  EfieldiceE_16_5[1:])
plt.scatter(Epart17[1:]*0.1,  EfieldiceE_17[1:])
plt.scatter(Epart17_5[1:]*0.316,  EfieldiceE_17_5[1:])
plt.xscale("log")
plt.yscale("log")
arg = np.argsort(test_func)
plt.plot(Epartallbins[arg], test_func[arg], c="red")
plt.show()


######## Fitting Erad  instead of Efield
sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.0316)
arg = np.argsort(Eradice_all[sel][:,6])
EradiceE_16_5 = Eradice_all[sel][:,3][arg]

sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.1)
arg = np.argsort(Eradice_all[sel][:,6])
ZenE_Erad = Eradice_all[sel][:,6][arg]
EradiceE_17 = Eradice_all[sel][:,3][arg]

sel = (Eradice_all[:,4] == 3116)  & (Eradice_all[:,5] ==0.316)
arg = np.argsort(Eradice_all[sel][:,6])
ZenE_Erad = Eradice_all[sel][:,6][arg]
EradiceE_17_5 = Eradice_all[sel][:,3][arg]

Epartallbins = np.concatenate([Epart16_5[1:]*1e3*0.0316, Epart17[1:]*1e3*0.1, Epart17_5[1:]*1e3*0.316])
Eradice_allE =  np.concatenate([EradiceE_16_5[1:],  EradiceE_17[1:], EradiceE_17_5[1:]])

from scipy.optimize import curve_fit
def ModelFunc(x, a):
    return a*x**2

popt, pcov = curve_fit(ModelFunc, Epartallbins, Eradice_allE)

print( popt)
test_func = ModelFunc(Epartallbins, popt)

plt.scatter(Epart16_5[1:]*1e3*0.0316,  EradiceE_16_5[1:])
plt.scatter(Epart17[1:]*1e3*0.1,  EradiceE_17[1:])
plt.scatter(Epart17_5[1:]*1e3*0.316,  EradiceE_17_5[1:])
arg = np.argsort(test_func)
plt.plot(Epartallbins[arg], test_func[arg], c="red")
plt.xscale("log")
plt.yscale("log")
plt.show()

#normalized
plt.plot(Epartallbins[1:],  Eradice_allE[1:])
arg = np.argsort(test_func)
plt.plot(Epartallbins[arg],  test_func[arg])
plt.show()

plt.scatter(Epartallbins,  Eradice_allE/test_func)
plt.show()
################
# Ground Particle Fit
################

from scipy.optimize import curve_fit
def ModelFunc(x, A, b):
    return A*np.exp(-x/b)

Epartallbins = np.concatenate([Epart16_5, Epart17, Epart17_5])
XmaxDistallbins = np.concatenate([XmaxDist16_5, XmaxDist17, XmaxDist17_5])
args = np.argsort(XmaxDistallbins)
XmaxDistallbins = XmaxDistallbins[args]
Epartallbins = Epartallbins[args]

popt, pcov = curve_fit(ModelFunc, XmaxDistallbins, Epartallbins, p0=[0.5, 5000])

test_func_2 = ModelFunc(XmaxDistallbins, popt[0], popt[1])
plt.scatter(XmaxDistallbins, Epartallbins)
plt.plot(XmaxDistallbins, test_func_2)
plt.show()

plt.scatter(XmaxDistallbins, Epartallbins/test_func_2)
plt.show()

##### Erad_ice fit


DataPath="/Users/chiche/Desktop/CRTemplateGen/1_ScalingLaws/EradScaling/Data/EradParamData/"


Path1 =DataPath + "Eradice_faerieParam/"
Path2=DataPath + "GroundPartE/"
Path3 = DataPath + "Eradice_signatures/"


zen_sorted, Erad_ice_16_5, XmaxE16_5_all = np.loadtxt(Path1 + "Eradice_z100_E16_5.txt", unpack = True).T
zen_sorted, Erad_ice_17, XmaxE17_all = np.loadtxt(Path1 + "Eradice_z100_E17.txt", unpack = True).T
zen_sorted, Erad_ice_17_5, XmaxE17_5_all = np.loadtxt(Path1 + "Eradice_z100_E17_5.txt", unpack = True).T


k =1
XmaxE16_5_all, XmaxE17_all, XmaxE17_5_all = 1e3*XmaxE16_5_all, 1e3*XmaxE17_all, 1e3*XmaxE17_5_all
Erad_ice_16_5 = Erad_ice_16_5/(0.0316)**2
Erad_ice_17 = Erad_ice_17/(0.1)**2
Erad_ice_17_5 = Erad_ice_17_5/(0.316)**2

Dxmax_all = np.concatenate([XmaxE16_5_all, XmaxE17_all, XmaxE17_5_all])
Erad_ice_comb = np.concatenate([Erad_ice_16_5, Erad_ice_17,Erad_ice_17_5 ])

plt.scatter(Dxmax_all, Erad_ice_comb)
plt.show()

arg = np.argsort(Dxmax_all)
Erad_ice_comb= Erad_ice_comb[arg]
Dxmax_all = Dxmax_all[arg]

def Erad_Model(x, A, Lambda):

    return A*np.exp(-2*x/Lambda)

popt, pcov = curve_fit(Erad_Model, Dxmax_all, Erad_ice_comb,  p0=[0.05, 1800])
Erad_fit = Erad_Model(Dxmax_all, popt[0], popt[1])
arg = np.argsort(Erad_ice_comb)


####
# Main plot
#### 
plt.scatter(Dxmax_all, Erad_ice_comb)
plt.plot(Dxmax_all, Erad_fit, color="red")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
#plt.savefig(OutputPath + "EradIceScaling_vs_dmax.pdf", bbox_inches="tight")
plt.show()

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
plt.plot(Dxmax_all, Erad_fit, color="red")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
plt.legend()
#plt.xscale("log")
#plt.yscale("log")
plt.savefig(OutputPath + "EradIceScaling_vs_dmax_ebins.pdf", bbox_inches="tight")
plt.show()


#plt.scatter(XmaxE16_5_all, Erad_ice_16_5, label=r"$E_{\rm p}=10^{16.5}$ eV", marker="o", color="blue")
#plt.scatter(XmaxE17_all, Erad_ice_17, label=r"$E_{\rm p}=10^{17}$ eV", marker="s", color="goldenrod")
plt.scatter(XmaxE17_5_all, Erad_ice_17_5, label=r"$E_{\rm p}=10^{17.5}$ eV", marker="^", color="firebrick")#
plt.plot(Dxmax_all, Erad_fit, color="red")
plt.xlabel(r"$D_{\rm xmax}^{\rm air}$ [m]")
plt.ylabel(r"$E_{\rm rad}^{\rm ice}/E_{\rm p}^{2}$ $[{\rm MeV}^{-1}]$")
plt.legend()
#plt.xscale("log")
#plt.yscale("log")
#plt.savefig(OutputPath + "EradIceScaling_vs_dmax_ebins.pdf", bbox_inches="tight")
plt.show()


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
plt.yscale("symlog")
plt.savefig(OutputPath + "EradIceScaling_relerr_vs_dmax_ebins_log.pdf", bbox_inches="tight")
plt.show()



rel_dev = (Erad_ice_comb - Erad_fit) / Erad_ice_comb
plt.scatter(Dxmax_all, rel_dev)
plt.show()

print(np.std(rel_dev))
rel_dev_clean = rel_dev[abs(rel_dev)<1]


plt.hist(abs(rel_dev))
plt.show()

perr = np.sqrt(np.diag(pcov))


theta = 0*np.pi/180
phi =0*np.pi/180

uv = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)])
Bgeo = np.array([7.705, 0, 54.111])
ub = Bgeo/np.linalg.norm(Bgeo)
alpha = np.arcsin(np.linalg.norm(np.cross(uv, ub)))*180/np.pi
print(alpha)

sin_alpha = np.linalg.norm(np.cross(uv, ub))
print(sin_alpha)
