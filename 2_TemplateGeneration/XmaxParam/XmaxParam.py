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
from .Modules.PlotErad import PlotEradThetaScaling, PlotEradDepthScaling, PlotEradEnergyScaling, PlotEradEScalingvsDepth,PlotAirIceEradRatiovsTheta, PlotAirIceEradRatiovsThetavsE, PlotHpoleVpoleEradRatiovsThetavsE, PlotEradtotThetaScaling, GetMeanEradScalingVsE, PlotMeanEradScalingVsE, PlotEradIceEScalingvsDepth, PlotGroundParticleEVsZenith, PlotEradIcevsZenE, PlotEradIcevsEgroundPart, EradicevsZenE, GetEradvsEgroundPart, GetGroundParticleEnergy, GetEgroundPart_E
#endregion

#region Path definition
SimDir = "CRTemplateGen" #"DeepCrLibV1"  #"InterpSim"
WorkPath = os.getcwd()
BatchID = "XmaxHadronicModel" #"DepthScaling" #"ThetaScaling"
OutputPath = MatplotlibConfig(WorkPath, SimDir, BatchID)
#endregion
Save = True
simpath = "/Users/chiche/Desktop/DeepCrAnalysis/Simulations/FullDenseDeepCr"
SimpathAll = glob.glob(simpath + "/*")


### INITIAL PARAMETERIZATION from fit of ZHS simulations

def Xmax_param(primary, energy, fluctuations=False):

    #input energy in EeV

    
    if(primary == 'Iron'):
        a =65.2
        c =270.6
        
        Xmax = a*np.log10(energy*1e6) + c
        
        if(fluctuations):
            a = 20.9
            b = 3.67
            c = 0.21
            
            sigma_xmax = a + b/energy**c
            Xmax = np.random.normal(Xmax, sigma_xmax)
        
        return Xmax
    
    elif(primary == 'Proton'):
        a = 57.4
        c = 421.9
        Xmax = a*np.log10(energy*1e6) + c
        
        if(fluctuations):
            a = 66.5
            b = 2.84
            c = 0.48
            
            sigma_xmax = a + b/energy**c
            Xmax = np.random.normal(Xmax, sigma_xmax)
        
        return Xmax
    
    else:
        print("missing primary")  

Ebins = np.logspace(17, 20, 10)/1e18 # EeV
Xmax_proton = [Xmax_param('Proton', E) for E in Ebins]
Xmax_iron = [Xmax_param('Iron', E) for E in Ebins]

Plot = False
if(Plot):
    plt.plot(Ebins, Xmax_proton, label='Proton')
    plt.plot(Ebins, Xmax_iron, label='Iron')
    plt.show()


### UPDATED PARAMETERIZATION from https://arxiv.org/pdf/2602.18118


def Xmax_param_QGSJETII(E, A, fluctuations=False):
    # E in eV
    # A is the mass number of the primary particle

    E0= 1e19 # eV 
    Efrac =np.log10(E/E0)

    a0 = 789.28
    a1 = 53.71
    b0 = -23.8
    b1 = 0.43

    ap = a0 + a1 * Efrac
    bp = b0 + b1 * Efrac
    Y = np.log(A)
    MeanXmax = ap + bp * Y

    f0 = 8.215
    f1 = -0.120 
    g0= -0.486
    g1= 6.60e-03

    fp = f0 +f1*Efrac
    gp = g0 + g1*Efrac

    var_Xmax = np.exp( fp + gp * Y ) 
    sigma_Xmax = np.sqrt(var_Xmax)

    if(not fluctuations):
        Xmax = MeanXmax
    else:
        Xmax = np.random.normal(MeanXmax, sigma_Xmax)

    return Xmax, MeanXmax, sigma_Xmax


if __name__ == "__main__":

    Ebins = np.logspace(16.5, 20, 10) # EeV

    Xmax_proton_QGSJETII, sigma_proton_QGSJETII = zip(
        *[Xmax_param_QGSJETII(E, 1)[1:] for E in Ebins]
    )

    Xmax_iron_QGSJETII, sigma_iron_QGSJETII = zip(
        *[Xmax_param_QGSJETII(E, 56)[1:] for E in Ebins]
    )

    Plot = True
    if(Plot):
        plt.plot(Ebins, Xmax_proton_QGSJETII, label='Proton QGSJETII')
        #plt.plot(Ebins, Xmax_proton, label='Proton', linestyle='dashed')
        plt.plot(Ebins, Xmax_iron_QGSJETII, label='Iron QGSJETII')  
        #plt.plot(Ebins, Xmax_iron, label='Iron', linestyle='dashed')
        plt.show()

        plt.errorbar(Ebins, Xmax_proton_QGSJETII, yerr=sigma_proton_QGSJETII, label='Proton QGSJETII', fmt='-o')
        plt.errorbar(Ebins, Xmax_iron_QGSJETII, yerr=sigma_iron_QGSJETII, label='Iron QGSJETII', fmt='-o')
        plt.show()