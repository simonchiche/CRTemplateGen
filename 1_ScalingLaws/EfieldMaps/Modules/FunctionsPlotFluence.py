#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:43:13 2024

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt
import sys

def EfieldMap(Pos, Depths, Nplanes, E, sim, save, energy, theta, path):
    
    
    Nlay = len(Depths)
    for i in range(Nlay):
        sel = (Pos[:,2] == Depths[i])
        s = 10 + 20 * (E[sel] - np.min(E)) / (np.max(E) - np.min(E))
        plt.scatter(Pos[sel,0], Pos[sel,1], \
                    c= E[sel], cmap = "jet", s=s, edgecolors='k', linewidth=0.2)
        cbar = plt.colorbar()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        #cbar.set_label("$\log_{10}(E)$ [$\mu V/m$]")
        cbar.set_label("$\log_{10}(E)$ [$\mu V ns/m$]")
        depth =Depths[0]- Depths[i]
        #plt.xlim(-200,200)
        #plt.ylim(-200,200)
        plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
        plt.title(sim + " map (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =14)
        plt.grid(True, linestyle='--', alpha=0.3)
        if(save):
            plt.savefig\
            (path + sim + "EfieldMap_E%.2f_th%.1f_depth%1.f.pdf" \
             %(energy, theta, depth), bbox_inches = "tight")
        plt.show()


def PlotSurfaceEz(Nant, glevel, Pos, Traces, thresold):
    for i in range(Nant):
        if(Pos[i,2]==glevel):
            if(max(abs(Traces[i][:,3]>thresold))):
                plt.plot(Traces[i][:,0]*1e9, Traces[i][:,3])
                plt.title("antenna %d" %i)
                plt.ylabel("E [$\mu V/m$]")
                plt.xlabel("Time [ns]")
                plt.show()

                dt = Traces[i][1,0]-  Traces[i][0,0]
                E_fft = np.fft.fft(Traces[i][:,3])
                freqs = np.fft.fftfreq(len(Traces[i][:,3]), d=dt)
                amplitude_spectrum = np.abs(E_fft) / len(Traces[i][:,3])

                plt.figure(figsize=(8, 4))
                plt.plot(freqs[:len(freqs)//2]/1e6, amplitude_spectrum[:len(freqs)//2])  # Only positive half
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Amplitude")
                plt.title("Fourier Spectrum antenna %d" %i)
                plt.grid()
                plt.xlim(0,500)
                plt.show()
def PlotLDF(Pos, Nplane, E, sim, Nlay):
    
    for i in range(Nlay):
        antmin = i*Nplane
        antmax = int((i+0.5)*Nplane)
        plt.plot(Pos[antmin:antmax,0], E[antmin:antmax])
        #plt.scatter(Pos[:int(Nlay/2),0], EtotC[int(Nlay/2):Nlay])
        plt.xlabel("x [m]")
        plt.ylabel("E [$\mu V/m$]")
        plt.title(sim + " LDF")
        plt.show()
    
    
def PlotTraces(Traces, start, stop):
    
    for i in range(start, stop, 1):
    
        plt.plot(Traces[i][:, 0]*1e9,Traces[i][:, 2])
        plt.xlabel("Time [ns]")
        plt.ylabel("E [$\mu V/m$]")
        #plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
        plt.show()


def PlotGivenTrace(Traces, arg, ax):
    
    if(ax == "x"): ax =1
    if(ax == "y"): ax =2
    if(ax == "z"): ax =3
    plt.plot(Traces[arg][:, 0]*1e9,Traces[arg][:, ax])
    plt.xlabel("Time [ns]")
    plt.ylabel("E [$\mu V/m$]")
    #plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
    plt.show()

def plot_polarisation(Pos, Etot_sp, Evxb, Evxvxb, Depths, path):
    
    # We keep the signal of suface antennas only
    sel = (Pos[:,2] == Depths[0]) 
    vxb, vxvxb = Pos[sel, 0], Pos[sel, 1]
    Evxb = Evxb[sel]
    Evxvxb = Evxvxb[sel]
    # function that plots the normalised polarisation in a given plane
    
    #r = np.sqrt(Evxb**2 + Evxvxb**2)
    plt.scatter(vxb, vxvxb, color = "white")
    #cbar = plt.colorbar()
    plt.xlabel('v x b [m]')
    plt.ylabel('v x (v x b) [m]')
    #plt.xlim(-200,200)
    #plt.ylim(-200,200)
    #plt.title(r'$\phi$ $= %.2f \degree$, $\theta$ $= %.2f \degree$, E = %.3f Eev' %(azimuth, zenith, energy), fontsize = 12)
    #cbar.set_label(r"$ E\ [\mu V/m]$")
    plt.quiver(vxb, vxvxb, -Evxb/20, Evxvxb/20)
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.tight_layout()
    plt.savefig(path + 'polarisation_sp.png', dpi = 500)
    plt.show()
    
def PlotMaxTraces(Traces, Etot, NantPlot):
    
    MaxId = np.argsort(abs(Etot))
    for i in range(NantPlot):
        arg = MaxId[-(i+1)]
        plt.plot(Traces[arg][:,0],Traces[arg][:,2] )
        plt.show()
        #PlotTraces(Traces, arg, arg+1)
    
    return

def PlotAllTraces(Nant, Traces, Threshold, Nmax):
    k =0
    
    for i in range(Nant):
        if(k<Nmax):
            if(max(abs(Traces[i][:, 2]))>Threshold):
                PlotTraces(Traces, i, i+1)
                print(i)
                k = k +1
    return

def PlotLayer(Pos, k, Nplane, path):
    plt.scatter(Pos[k*Nplane:(k+1)*Nplane,0], Pos[k*Nplane:(k+1)*Nplane,1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(path + "RectGrid.pdf")
    plt.show()
    
    return

def PlotAllChannels(Traces, ID):
   
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 1], label ="x-channel")
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 2], label ="y-channel")
    plt.plot(Traces[ID][:, 0]*1e9,Traces[ID][:, 3], label ="z-channel")
    plt.xlabel("Time [ns]")
    plt.ylabel("E [$\mu V/m$]")
    plt.legend()
    #plt.xlim(500,700)
    plt.xlim(630,650)
    #plt.title(r"Proton  E = 0.01 EeV - $\theta = 0^{\circ}$ $\phi = 0^{\circ}$")
    plt.tight_layout()
    plt.savefig("/Users/chiche/Desktop/AllChannelsInIce_%.d.pdf" %ID)
    plt.show()
    
    return
    

def getcoredistance(Pos):
    r = np.sqrt(Pos[:,0]**2 + Pos[:,1]**2)
    return r

def RemoveCoreAntennas(Pos, rlim, Ex, Ey, Ez, Etot):
    coredist =getcoredistance(Pos) 
    sel = (coredist>=rlim)
    Etot, Ex, Ey, Ez =\
        Etot[sel], Ex[sel], Ey[sel], Ez[sel] 
    Pos = Pos[sel]

    return Pos, Etot, Ex, Ey, Ez


from scipy.interpolate import Rbf
from scipy.interpolate import griddata

def interpolate_2d(x, y, z, 
                   method='linear', 
                   grid_resolution=100, 
                   bounds=None):
    """
    Interpolates scattered 2D data (x, y, z) onto a regular grid.
    
    Parameters:
    ----------
    x, y, z : array-like
        1D arrays of coordinates (x, y) and values (z) at those points.
    method : str
        Interpolation method: 'linear', 'cubic', or 'nearest'.
    grid_resolution : int or tuple of two ints
        Number of points in the x and y directions for the grid.
    bounds : tuple (xmin, xmax, ymin, ymax), optional
        Explicit bounds for the interpolation grid. If None, inferred from data.
    
    Returns:
    -------
    grid_x, grid_y : 2D arrays
        Grid coordinates.
    grid_z : 2D array
        Interpolated z values over the grid.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    if isinstance(grid_resolution, int):
        nx = ny = grid_resolution
    else:
        nx, ny = grid_resolution

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, nx),
        np.linspace(ymin, ymax, ny)
    )

    grid_z = griddata(
        points=(x, y),
        values=z,
        xi=(grid_x, grid_y),
        method=method
    )

    return grid_x, grid_y, grid_z

def interpolate_rbf(x, y, z, grid_resolution=100, bounds=None, function='linear'):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if bounds is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    if isinstance(grid_resolution, int):
        nx = ny = grid_resolution
    else:
        nx, ny = grid_resolution

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, nx),
        np.linspace(ymin, ymax, ny)
    )

    rbf = Rbf(x, y, z, function=function)  # function='linear', 'multiquadric', 'gaussian', etc.
    grid_z = rbf(grid_x, grid_y)

    return grid_x, grid_y, grid_z


def InterpolatedEfieldMap(Pos, Depths, Nplanes, E, sim, save, energy, theta, path, unit, scalelim):
    
    Nlay = len(Depths)
    for i in range(Nlay):
        sel = (Pos[:,2] == Depths[i])
        #
        # s = 10 + 20 * (E[sel] - np.min(E)) / (np.max(E) - np.min(E))
        argxmax = np.argmax(E[sel])
        xmax = abs(Pos[:,0][sel][argxmax])
        if(xmax<50): xmax = 50


        grid_x, grid_y, grid_z = \
         interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], E[sel])
        
        plt.figure(figsize=(6, 5))
        plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet')
        #plt.scatter(Pos[:729,0], Pos[:729,1], c=np.log10(EtotC_int[:729] +1), edgecolor='white', s=100)
        #plt.colorbar(label="$\log_{10}(E)$ [$\mu V/m$]")
        plt.colorbar(label=unit)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
  
        #plt.title(sim + " map (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =12)
        plt.title(sim + " map (E =$%.1f\,$eV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =12)

        depth =Depths[0]- Depths[i]
        #plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
        plt.text(0.05, 0.95, f"Depth = {depth:.0f} m", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        #plt.xlim(min(grid_x), max(grid_x))
        #plt.ylim(min(grid_x), max(grid_x))
        #plt.xlim(-250,350)
        #plt.ylim(-250,350)
        #plt.ylim(-300,300)
        plt.xlim(-scalelim*xmax, scalelim*xmax)
        plt.ylim(-scalelim*xmax, scalelim*xmax)
        if(save):
            plt.savefig\
            (path + sim + "EfieldMap_E%.2f_th%.1f_depth%1.f.pdf" \
             %(energy, theta, depth), bbox_inches = "tight")
        plt.show()
        

def InterpolatedfluencedMap(Pos, Depths, Nplanes, E, sim, save, energy, theta, path, scalelim):
    
    Nlay = len(Depths)
    for i in range(Nlay):
        sel = (Pos[:,2] == Depths[i])
        #
        # s = 10 + 20 * (E[sel] - np.min(E)) / (np.max(E) - np.min(E))
        argxmax = np.argmax(E[sel])
        xmax = abs(Pos[:,0][sel][argxmax])
        if(xmax<50): xmax = 50


        grid_x, grid_y, grid_z = \
         interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], E[sel])
        
        plt.figure(figsize=(6, 5))
        plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='jet')
        #plt.scatter(Pos[:729,0], Pos[:729,1], c=np.log10(EtotC_int[:729] +1), edgecolor='white', s=100)
        #plt.colorbar(label="$\log_{10}(E)$ [$\mu V/m$]")
        plt.colorbar(label="fluence [$\mathrm{e}V/\mathrm{m}^{2}$]")
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
  
        #plt.title(sim + " map (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =12)
        plt.title(sim + " map (E =$%.1f\,$eV, $\\theta=%.1f^{\circ}$)" %(energy, theta), size =14)

        depth =Depths[0]- Depths[i]
        #plt.legend(["Depth = %.f m" %(depth)], loc ="upper right")
        plt.text(0.05, 0.95, f"Depth = {depth:.0f} m", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        #plt.xlim(min(grid_x), max(grid_x))
        #plt.ylim(min(grid_x), max(grid_x))
        #plt.xlim(-250,350)
        #plt.ylim(-250,350)
        #plt.ylim(-300,300)
        plt.xlim(-scalelim*xmax, scalelim*xmax)
        plt.ylim(-scalelim*xmax, scalelim*xmax)
        if(save):
            plt.savefig\
            (path + sim + "fluenceMap_E%.2f_th%.1f_depth%1.f.pdf" \
             %(energy, theta, depth), bbox_inches = "tight")
        plt.show()
