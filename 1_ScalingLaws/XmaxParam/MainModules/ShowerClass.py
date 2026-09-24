#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:45:27 2022

@author: chiche
"""

import h5py
import numpy as np
import sys
from MainModules.HDF5reader import LoadHDF5file
from scipy.integrate import trapz, simps
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.interpolate import Rbf


class Shower:
    
    def __init__(self, primary, energy, zenith, azimuth, injection_height, \
                 nantennas, Bgeo, GroundAltitude, XmaxParam, Positions, \
                 Traces_C, Traces_G):
    
        self.primary = primary
        self.energy = energy
        self.zenith = zenith
        self.azimuth = azimuth
        self.injection = injection_height
        self.nant = nantennas
        self.B = Bgeo
        self.pos = Positions
        self.traces_c = Traces_C
        self.traces_g = Traces_G
        self.glevel = GroundAltitude
        self.xmax = XmaxParam[0]
        self.xmaxdist = XmaxParam[1]
        
        
        self.xmaxpos = [0,0,0] #XmaxPosition
        self.distplane = self.get_distplane()
        self.inclination = np.arctan(Bgeo[1]/Bgeo[0])

# =============================================================================
#                     GetInShowerPlane Functions
# =============================================================================
        
    def showerdirection(self):
        
        azimuth, zenith = self.azimuth, self.zenith
        zenith = zenith*np.pi/180.0
        azimuth = azimuth*np.pi/180.0
        
        uv = np.array([np.sin(zenith)*np.cos(azimuth), \
                       np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
        
        return uv
    
    def GetDepths(self):
        
        Pos = self.pos
        Nant = len(Pos)
        
        Depths = np.sort(np.unique(Pos[:,2]))[::-1]
        
        Nlay = len(Depths)
        Nplane = int(Nant/Nlay)
        
        return Nlay, Nplane, Depths
    
    def GetPeakTraces(self, Traces):
    
        Nant = self.nant
        #print(Nant)
        Etot = np.zeros(Nant)
        Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
        
        for i in range(Nant):
            
            Ex[i] = max(abs(Traces[i][:,1]))
            Ey[i] = max(abs(Traces[i][:,2]))
            Ez[i] = max(abs(Traces[i][:,3]))
            Etot[i] = max(np.sqrt((Traces[i][:,1])**2 + \
                (Traces[i][:,2])**2 + (Traces[i][:,3])**2)) 

        return Ex, Ey, Ez, Etot
    

    def GetIntTraces(self, Traces):
    
        Nant = self.nant
        Etot, peakTime = np.zeros(Nant), np.zeros(Nant)
        Ex, Ey, Ez = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
        binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10
        for i in range(Nant):
            
            #Etot_all = np.sqrt(Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2)
            Etot_all = abs(hilbert(np.sqrt(Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2)))
            extent = 10000
            peak_id = np.argmax(Etot_all)
            peakTime[i] = Traces[i][peak_id,0]
            minid = peak_id -extent
            maxid = peak_id + extent
            if(minid<0): minid = 0
            if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
            
            time = np.arange(0, len(Traces[i][minid:maxid,0]))*binT
            
            Ex[i] = simps(abs(hilbert(Traces[i][minid:maxid,1])), time)*1e9
            Ey[i] = simps(abs(hilbert(Traces[i][minid:maxid,2])), time)*1e9
            Ez[i] = simps(abs(hilbert(Traces[i][minid:maxid,3])), time)*1e9
            Etot[i] = np.sqrt(Ex[i]**2 + Ey[i]**2 + Ez[i]**2)

        return Ex, Ey, Ez, Etot, peakTime
    

    def GetFluence(self, Traces):
        eps0 = 8.85e-12 # F.m^{-1}
        c = 3e8 # m.s^{-1}

        Nant = self.nant
        ftot = np.zeros(Nant)
        fx, fy, fz = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
        binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10 # time in seconds
        

        for i in range(Nant):
            
            ftot_t = Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2
            extent = 10000
            peak_id = np.argmax(ftot_t)
            minid = peak_id -extent
            maxid = peak_id + extent
            if(minid<0): minid = 0
            if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
            
            time = np.arange(0, len(Traces[i][minid:maxid,0]))*binT
            #Traces[i][:,1:] = Traces[i][:,1:]/1e6
            
            #fx[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,1]**2)), time)
            #fy[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,2]**2)), time)
            #fz[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,3]**2)), time)
            #sys.exit()
            fx[i] = eps0*c*simps(Traces[i][minid:maxid,1]**2, time)/1e12 #conversion in J/m^2 since Efield is in µV/m and squared
            fy[i] = eps0*c*simps(Traces[i][minid:maxid,2]**2, time)/1e12
            fz[i] = eps0*c*simps(Traces[i][minid:maxid,3]**2, time)/1e12

            ftot[i] = (fx[i] + fy[i] + fz[i])
            #Traces[i][:,1:] = Traces[i][:,1:]*1e6
        return fx, fy, fz, ftot
    
    def interpolate_rbf(self, x, y, z, grid_resolution, bounds=None, function='cubic'):
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
    

    def GetEradFromSim(self, Traces):

        Nlay, Nplane, Depths = self.GetDepths()
        print("Depths", Depths)
        fx, fy, fz, ftot = self.GetFluence(Traces)
        Pos = self.pos
        Erad_all = []
        for k in range(len(Depths)):
            sel = (Pos[:,2] == Depths[k])
            sortedPos = sorted(set(Pos[sel][:,0]))
            spacing = abs(sortedPos[1] - sortedPos[0])
            #Power
            Eradx, Erady, Eradz, Eradtot = \
                np.sum(fx[sel]*spacing**2), np.sum(fy[sel]*spacing**2),\
                np.sum(fz[sel]*spacing**2), np.sum(ftot[sel]*spacing**2)
            
            Eradx, Erady, Eradz, Eradtot = (x / (1.6e-19) for x in (Eradx, Erady, Eradz, Eradtot))
            Eradx, Erady, Eradz, Eradtot = (x / (1e6) for x in (Eradx, Erady, Eradz, Eradtot))

            Erad_all.append(np.array([Eradx, Erady, Eradz, Eradtot, Depths[k], self.energy, self.zenith]))

        return np.array(Erad_all)
    
    def GetRadiationEnergyGeneric(self, Traces):
        
        Pos = self.pos
        Nlay, Nplane, Depths = self.GetDepths()
        fx, fy, fz, ftot = self.GetFluence(Traces)

        Erad = []#np.zeros(len(Depths))
        for i in range(len(Depths)):
            sel = (Pos[:,2] == Depths[i])

            grid_x, grid_y, f_int = \
            self.interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], ftot[sel], grid_resolution=100)            
            # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
            integral_x = trapz(f_int, x=grid_x[0], axis=1)  # integrate over x (axis=1)
            Eradtot = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)

            grid_x, grid_y, f_int = \
            self.interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], fx[sel], grid_resolution=100)            
            # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
            integral_x = trapz(f_int, x=grid_x[0], axis=1)  # integrate over x (axis=1)
            Eradx = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)

            grid_x, grid_y, f_int = \
            self.interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], fy[sel], grid_resolution=100)            
            # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
            integral_x = trapz(f_int, x=grid_x[0], axis=1)  # integrate over x (axis=1)
            Erady = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)

            grid_x, grid_y, f_int = \
            self.interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], fz[sel], grid_resolution=100)            
            # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
            integral_x = trapz(f_int, x=grid_x[0], axis=1)  # integrate over x (axis=1)
            Eradz = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)

            Eradx, Erady, Eradz, Eradtot = (x / (1.6e-19) for x in (Eradx, Erady, Eradz, Eradtot)) # From J to eV
            Eradx, Erady, Eradz, Eradtot = (x / (1e6) for x in (Eradx, Erady, Eradz, Eradtot)) # From eV to MeV

            Erad.append(np.array([Eradx, Erady, Eradz, Eradtot, Depths[i], self.energy, self.zenith]))

        return Erad
    

    def CombineTraces(self):
        Nant = self.nant
        Traces_C = self.traces_c
        Traces_G = self.traces_g
        TracesTot = dict()

        for i in range(Nant):
            if(i%1000==0): print(i)

            binT = round((Traces_C[i][1,0] -Traces_C[i][0,0])*1e10)/1e10
            
            Ex = [] 
            Ey = [] 
            Ez = [] 
            Twindow = []
            
            if(max(Traces_C[i][:,0])< min(Traces_G[i][:,0])):   
                
                Ex = np.concatenate((Traces_C[i][:,1], [0,0], Traces_G[i][:,1]))
                Ey = np.concatenate((Traces_C[i][:,2],[0,0], Traces_G[i][:,2]))
                Ez = np.concatenate((Traces_C[i][:,3],[0,0], Traces_G[i][:,3]))
                gap1 = Traces_C[i][-1,0] + binT
                gap2 = Traces_G[i][0,0] - binT
                Twindow = np.concatenate((Traces_C[i][:,0], [gap1, gap2], Traces_G[i][:,0]))
                
                TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
                
            if((max(Traces_C[i][:,0])> min(Traces_G[i][:,0])) &\
            (max(Traces_C[i][:,0])<max(Traces_G[i][:,0]))):
                
                argminG = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][0,0]))
                argmaxC = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][-1,0]))
                
                Twindow = np.concatenate((Traces_C[i][:argminG,0], Traces_G[i][:,0]))
                
                Ex_air, Ey_air, Ez_air = Traces_C[i][:argminG,1], Traces_C[i][:argminG,2],\
                Traces_C[i][:argminG,3]
                
                tmin = Traces_C[i][argminG,0]
                t = tmin
                k = 0
                GapTrace = False
                Ex_air_ice, Ey_air_ice, Ez_air_ice = [], [], []
        
                for j in range(argminG, len(Traces_C[i][:,0]), 1):
                    argmint = np.argmin(abs(t - Traces_G[i][:,0]))
                    if(abs(t - Traces_G[i][argmint,0])<=binT):
                        
                        Ex_air_ice.append(Traces_C[i][j,1] + Traces_G[i][k,1])
                        Ey_air_ice.append(Traces_C[i][j,2] + Traces_G[i][k,2])
                        Ez_air_ice.append(Traces_C[i][j,3] + Traces_G[i][k,3])
                        
                        k = k +1
                        t = t + binT
                        kmaxG =k
                        
                    else:
                        Ex_air_ice.append(Traces_C[i][j,1])
                        Ey_air_ice.append(Traces_C[i][j,2])
                        Ez_air_ice.append(Traces_C[i][j,3])
                        
                        k = k +1
                        t = t + binT
                        GapTrace = True
                
                if(GapTrace): Twindow = np.concatenate((Traces_C[i][:,0], Traces_G[i][kmaxG:,0]))
                    
                Ex_ice, Ey_ice, Ez_ice = Traces_G[i][kmaxG:,1], \
                Traces_G[i][kmaxG:,2], Traces_G[i][kmaxG:,3]
                
                Ex = np.concatenate((Ex_air, Ex_air_ice, Ex_ice))
                Ey = np.concatenate((Ey_air, Ey_air_ice, Ey_ice))
                Ez = np.concatenate((Ez_air, Ez_air_ice, Ez_ice))
        
                TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
                
            if((max(Traces_C[i][:,0])>max(Traces_G[i][:,0]))):
                argminG = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][0,0]))
                argmaxC = np.argmin(abs(Traces_C[i][:,0]-Traces_G[i][-1,0]))
            
                Twindow = Traces_C[i][:,0]
            
                Ex_air, Ey_air, Ez_air = Traces_C[i][:argminG,1], Traces_C[i][:argminG,2],\
                Traces_C[i][:argminG,3]
                
                tmin = Traces_C[i][argminG,0]
                t = tmin
                k = 0
                
                Ex_air_ice, Ey_air_ice, Ez_air_ice = [], [], []
                
                for j in range(argminG, argmaxC, 1):
                    
                    argmint = np.argmin(abs(t - Traces_G[i][:,0]))
                    if(abs(t - Traces_G[i][argmint,0])<=binT):
                        
                        Ex_air_ice.append(Traces_C[i][j,1] + Traces_G[i][k,1])
                        Ey_air_ice.append(Traces_C[i][j,2] + Traces_G[i][k,2])
                        Ez_air_ice.append(Traces_C[i][j,3] + Traces_G[i][k,3])
                        
                        k = k +1
                        t = t + binT
                        
                    else:
                        Ex_air_ice.append(Traces_C[i][j,1])
                        Ey_air_ice.append(Traces_C[i][j,2])
                        Ez_air_ice.append(Traces_C[i][j,3])
                        
                        k = k +1
                        t = t + binT
                    
                Ex_air2, Ey_air2, Ez_air2 = Traces_C[i][argmaxC:,1], \
                Traces_C[i][argmaxC:,2], Traces_C[i][argmaxC:,3]
                
                Ex = np.concatenate((Ex_air, Ex_air_ice, Ex_air2))
                Ey = np.concatenate((Ey_air, Ey_air_ice, Ey_air2))
                Ez = np.concatenate((Ez_air, Ez_air_ice, Ez_air2))
        
                TracesTot[i] = np.array([Twindow, Ex, Ey, Ez]).T
        return TracesTot
    

    def filter_single_trace(self, signal, fs, lowcut, highcut, order=4):
        """
        Apply a bandpass filter to a signal.
        
        Parameters:
        - signal: array-like, the input signal (E(t)).
        - fs: float, the sampling frequency of the signal in Hz.
        - lowcut: float, the lower bound of the frequency band in Hz.
        - highcut: float, the upper bound of the frequency band in Hz.
        - order: int, the order of the Butterworth filter (default is 4).
        
        Returns:
        - filtered_signal: array-like, the filtered signal.
        """
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design the Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='band')

        # Apply the filter using filtfilt for zero phase shift
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def filter_all_traces(self, Traces, fs, lowcut, highcut, order=4):

        Traces_filtered = dict()
        
        for i in range(len(Traces)):
            Exg_f = self.filter_single_trace(Traces[i][:,1], fs, lowcut, highcut, order=4)
            Eyg_f = self.filter_single_trace(Traces[i][:,2], fs, lowcut, highcut, order=4)
            Ezg_f = self.filter_single_trace(Traces[i][:,3], fs, lowcut, highcut, order=4)
            trace_filtered = np.array([Traces[i][:,0],Exg_f, Eyg_f, Ezg_f]).T
            #sys.exit()
            Traces_filtered[i] = trace_filtered
            
        return Traces_filtered

        
    
    def get_distplane(self):
 
    #function that returns "w" at each antenna, i.e. the angle between the 
    #direction that goes from Xmax to the core and the direction that 
    #goes from Xmax to a given antenna
    
        pos = self.pos
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        xmaxpos = self.xmaxpos
        x_Xmax, y_Xmax, z_Xmax = xmaxpos[0], xmaxpos[1], xmaxpos[2]
            
        x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
        y_antenna = y - y_Xmax
        z_antenna = z - z_Xmax
    
        uv = self.showerdirection()
        u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
        distplane = np.dot(np.transpose(u_antenna), uv)
        #print(distplane)
        #print(u_antenna)
        return np.mean(distplane)

    def get_center(self, distplane = 0):

        xmaxpos = self.xmaxpos
        x_Xmax, y_Xmax, z_Xmax = xmaxpos[0], xmaxpos[1], xmaxpos[2]
        GroundLevel = self.glevel
    
        uv = self.showerdirection()

        distground = np.sqrt(x_Xmax**2 + y_Xmax**2 + (z_Xmax-GroundLevel)**2)
        
        distplane = self.distplane

        dist_plane_ground = distground - distplane
        core = -uv*(dist_plane_ground)
        core[2] = core[2] + GroundLevel
            
        return core    
    
    def GetinShowerPlane(self):
        
        # function that returns the trcaes in the shower plane (v, vxb, vxvxb) 
        #from the traces in the geographic plane (x, y, z)

        inclination = self.inclination*np.pi/180.0
        
        pos =  self.pos
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        n = len(x) # number of antennas
        
        # We move the core position in (0,0,0) before changing the 
        #reference frame
        
        core = self.get_center()
        
        x = x - core[0]
        y = y - core[1] 
        z = z - core[2]
        
        Traces = self.traces
        time_sample = len(Traces[:,0])
        
        # antennas positions in the  shower reference frame (v, vxB, vxvxB)
        v = np.zeros(n)   
        vxb = np.zeros(n)
        vxvxb = np.zeros(n)
        
        #Traces in the shower reference frame
        Traces_Ev = np.zeros([time_sample,n])
        Traces_Evxb = np.zeros([time_sample,n])
        Traces_Evxvxb = np.zeros([time_sample,n])
        Time = np.zeros([time_sample, n])
        
        # unit vectors 
        uv = self.showerdirection()
        uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field 
        
        uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
        uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
        
        uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
        uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
        
        P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
        
        P_inv = np.linalg.inv(P) # matrix to go from the geographic reference frame to the shower reference frame
        
        # We calculate the positions in the shower plane
        Position_geo = np.array([x,y,z]) # position in the geographic reference frame
        Position_shower = np.dot(P_inv, Position_geo) # position in the shower reference frame
        
        # We deduce the different components
        v = Position_shower[0, :] 
        vxb = Position_shower[1, :]
        vxvxb =  Position_shower[2, :]
        
        # We calulate the traces in the shower plane
        Traces_geo = np.zeros([time_sample,3])
        Traces_shower_temp = np.zeros([3, time_sample])
        
        for i in range(n):
            
            Traces_geo = np.array([Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i + 3*n]])
            
            Traces_shower_temp = np.dot(P_inv, Traces_geo)
            
            Traces_Ev[:,i] = np.transpose(Traces_shower_temp[0,:]) # Ev component of the traces
            Traces_Evxb[:,i] = np.transpose(Traces_shower_temp[1,:]) # Evxb component of the traces
            Traces_Evxvxb[:,i] = np.transpose(Traces_shower_temp[2,:]) # Evxvxb component of the traces
            
            Time[:,i] = Traces[:,i]
        
        # We derive the traces in the shower plane
        
        Traces_sp = np.transpose(np.concatenate((np.transpose(Time), \
        np.transpose(Traces_Ev), np.transpose(Traces_Evxb), np.transpose(Traces_Evxvxb))))
    
        Positions_sp = np.zeros([n,3]) 
        Positions_sp[:,0], Positions_sp[:,1], Positions_sp[:,2] = v, vxb, vxvxb
    
        return Positions_sp, Traces_sp
    
# =============================================================================
#                        shower angles
# =============================================================================
      
    def get_alpha(self):
    
    # function that returns the angle between the direction of the shower and 
    #the direction of the magnetic field 
    
        inclination = self.inclination
        inclination = inclination*np.pi/180.0
        
        # unit vectors    
        uv = self.showerdirection()
        uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
        cos_alpha = np.dot(uv,uB)
        alpha = np.arccos(cos_alpha) # angle between the direction of the shower and the direction of the magnetic field
        
        return alpha
    
    def get_w(self):
 
        #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
        
        inclination = self.inclination
        xmaxpos = self.xmaxpos
        pos = self.pos
        x_Xmax, y_Xmax, z_Xmax =  xmaxpos[0], xmaxpos[1], xmaxpos[2]
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        
        inclination = inclination*np.pi/180.0
        
        x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
        y_antenna = y - y_Xmax
        z_antenna = z - z_Xmax
        
        uv = self.showerdirection()
        u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
        u_antenna /= np.linalg.norm(u_antenna, axis =0)
        w = np.arccos(np.dot(np.transpose(u_antenna), uv))
        w = w*180.0/np.pi # we calculte w in degrees
    
        return w  
    
# =============================================================================
#                    Get parametrized Xmax position
# =============================================================================
        
    def _getAirDensity(self, _height, model):

        '''Returns the air density at a specific height, using either an 
        isothermal model or the Linsley atmoshperic model as in ZHAireS
    
        Parameters:
        ---------
            h: float
                height in meters
    
        Returns:
        -------
            rho: float
                air density in g/cm3
        '''
    
        if model == "isothermal":
                #Using isothermal Model
                rho_0 = 1.225*0.001    #kg/m^3
                M = 0.028966    #kg/mol
                g = 9.81        #m.s^-2
                T = 288.        #
                R = 8.32        #J/K/mol , J=kg m2/s2
                rho = rho_0*np.exp(-g*M*_height/(R*T))  # kg/m3
    
        elif model == "linsley":
            #Using Linsey's Model
            bl = np.array([1222., 1144., 1305.5948, 540.1778,1])*10  # g/cm2  ==> kg/cm3
            cl = np.array([9941.8638, 8781.5355, 6361.4304, 7721.7016, 1e7])  #m
            hl = np.array([4,10,40,100,113])*1e3  #m
            if (_height>=hl[-1]):  # no more air
                rho = 0
            else:
                hlinf = np.array([0, 4,10,40,100])*1e3  #m
                ind = np.logical_and([_height>=hlinf],[_height<hl])[0]
                rho = bl[ind]/cl[ind]*np.exp(-_height/cl[ind])
                rho = rho[0]*0.001
        else:
            print("#### Error in GetDensity: model can only be isothermal or linsley.")
            return 0
    
        return rho
        
    def getSphericalXmaxHeight(self):
    
        XmaxPosition = self.xmaxpos
        Rearth = 6370949 
        XmaxHeight = np.sqrt((Rearth + XmaxPosition[2])**2 + XmaxPosition[0]**2 +\
                             XmaxPosition[1]**2) - Rearth
    
        return XmaxHeight
    
    def Xmax_param(self):

        #input energy in EeV
        
        primary = self.primary
        energy= self.energy
        fluctuations = self.fluctuations
        
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
            
    
    
    def _get_CRzenith(self):
        ''' Corrects the zenith angle for CR respecting Earth curvature, zenith seen by observer
            ---fix for CR (zenith computed @ shower core position
        
        Arguments:
        ----------
        zen: float
            GRAND zenith in deg
        injh: float
            injection height wrt to sealevel in m
        GdAlt: float
            ground altitude of array/observer in m (should be substituted)
        
        Returns:
        --------
        zen_inj: float
            GRAND zenith computed at shower core position in deg
            
        Note: To be included in other functions   
        '''
    
        #Note: To be included in other functions
        zen = self.zenith

        GdAlt = self.glevel
        injh = self.injection
                
        Re= 6370949 # m, Earth radius
    
        a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 *np.sin(np.pi-np.deg2rad(zen))**2) - (Re+GdAlt)*np.cos(np.pi-np.deg2rad(zen))
        zen_inj = np.rad2deg(np.pi-np.arccos((a**2 +(Re+injh)**2 -Re**2)/(2*a*(Re+injh))))
        
        
        return zen_inj    

            
    def _dist_decay_Xmax(self): 
        ''' Calculate the height of Xmax and the distance injection point to Xmax along the shower axis
        
        Arguments:
        ----------
        zen: float
            GRAND zenith in deg, for CR shower use _get_CRzenith()
        injh2: float
            injectionheight above sealevel in m
        Xmax_primary: float
            Xmax in g/cm2 
            
        Returns:
        --------
        h: float
            vertical Xmax_height in m
        ai: float
            Xmax_distance injection to Xmax along shower axis in m
        '''
        
        zen = self._get_CRzenith()
        injh2 = self.injection
        Xmax_primary = self.Xmax_param() 
        zen2 = np.deg2rad(zen)
        
        hD=injh2
        step=10 #m
        if hD>10000:
            step=100 #m
        Xmax_primary= Xmax_primary#* 10. # g/cm2 to kg/m2: 1g/cm2 = 10kg/m2
        gamma=np.pi-zen2 # counterpart of where it goes to
        Re= 6370949 # m, Earth radius
        X=0.
        i=0.
        h=hD
        ai=0
        while X< Xmax_primary:
            i=i+1
            ai=i*step #100. #m
            hi= -Re+np.sqrt(Re**2. + ai**2. + hD**2. + 2.*Re*hD - 2*ai*np.cos(gamma) *(Re+hD))## cos(gamma)= + to - at 90dg
            deltah= abs(h-hi) #(h_i-1 - hi)= delta h
            h=hi # new height
            rho = self._getAirDensity(hi, "linsley")
            X=X+ rho * step*100. #(deltah*100) *abs(1./np.cos(np.pi-zen2)) # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta); density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
           
        return h, ai # Xmax_height in m, Xmax_distance in m    
    
    def getGroundXmaxDistance(self):
        
        # zenith in cosmic ray convention here
        
        zenith = (180 -self.zenith)*np.pi/180
        GroundAltitude = self.glevel
        XmaxHeight, DistDecayXmax = self._dist_decay_Xmax()
    
        Rearth = 6370949 
        
        dist = np.sqrt((Rearth+ XmaxHeight)**2 - ((Rearth + GroundAltitude)*np.sin(zenith))**2) \
        - (Rearth + GroundAltitude)*np.cos(zenith)
                
        return dist   
    
    def getXmaxPosition(self):
        
        uv = self.showerdirection()
        showerDistance = self.getGroundXmaxDistance()
        XmaxPosition = -uv*showerDistance 
        XmaxPosition[2] = XmaxPosition[2] + self.glevel  
                
        return XmaxPosition
    
# =============================================================================
#                 Cerenkov angle computation
# =============================================================================

    def GetZHSEffectiveactionIndex(self ,ns=325,kr=-0.1218,stepsize = 20000):
         
        
          XmaxPosition = self.xmaxpos
          core = self.get_center()
          xant, yant, zant = core[0], core[1], core[2]
          x0, y0, z0 = XmaxPosition[0], XmaxPosition[1], XmaxPosition[2]
          #rearth=6371007.0 #new aires
          rearth=6370949.0 #19.4.0
    #     Variable n integral calculation ///////////////////
          R02=x0*x0+y0*y0  #!notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada (entro con injz-zXmax -> injz-z0=zXmax
          h0=(np.sqrt( (z0+rearth)*(z0+rearth) + R02 ) - rearth)/1E3    #!altitude of emission
    
          rh0=ns*np.exp(kr*h0) #!refractivity at emission (this
          n_h0=1+1E-6*rh0 #!n at emission
    #        write(*,*) "n_h0",n_h0,ns,kr,x0,y0,injz-z0,h0,rh0
          modr=np.sqrt(R02)
    
          if(modr > 1000): #! if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers
    
    #         Vector from average point of track to observer.
              ux = xant-x0
              uy = yant-y0
              uz = zant-z0
    
    #         divided in nint pieces shorter than 10km
              nint=int((modr/stepsize)+1)
              kx=ux/nint
              ky=uy/nint       #k is vector from one point to the next
              kz=uz/nint
    #
              currpx=x0
              currpy=y0        #current point (1st is emission point)
              currpz=z0
              currh=h0
    #
              sum=0
              for iii in range(0,nint):
                nextpx=currpx+kx
                nextpy=currpy+ky #!this is the "next" point
                nextpz=currpz+kz
                nextR2=nextpx*nextpx + nextpy*nextpy
                nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1E3
    #c
                if(np.abs(nexth-currh) > 1E-10  ):
                  sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
                else:
                  sum=sum+np.exp(kr*currh)
    #            endif
    #c
                currpx=nextpx
                currpy=nextpy
                currpz=nextpz  #!Set new "current" point
                currh=nexth
    #c
              avn=ns*sum/nint
    #          print*,"avn:",avn
              n_eff=1+1E-6*avn #!average (effective) n
    #c
          else:
    #c         withouth integral
              hd=zant/1E3 #!detector altitude
    #
              if(np.abs(hd-h0) > 1E-10):
                avn=(ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
    #            print*,"avn2:",avn
              else:
                avn=ns*np.exp(kr*h0)
    #           print*,"avn3:",avn
    #            print *,"Effective n: h0=hd"
    #          endif
              n_eff=1+1E-6*avn #!average (effective) n
    #        endif
    #c     ///////////////////////////////////////////////////
          return n_eff
        
    def get_cerenkov_angle(self):
        
        n_refraction = self.GetZHSEffectiveactionIndex()
        cer_ang = np.rad2deg(np.arccos(1/n_refraction))
        
        return cer_ang
    
# =============================================================================
#                   Get in Geographic Frame
# =============================================================================
    
    def GetinGeographicFrame(self):
        
        # function that returns the trcaes in the shower plane (v, vxb, vxvxb) 
        #from the traces in the geographic plane (x, y, z)
    
        inclination = self.inclination*np.pi/180.0
        
        pos =  self.pos
        v, vxb, vxvxb = pos[:,0], pos[:,1], pos[:,2]
        n = len(vxb) # number of antennas
        
        # We move the core position in (0,0,0) before changing the reference frame
        #core = self.get_center(self.get_distplane)
        
        Traces = self.traces
        time_sample = len(Traces[:,0])
        
        # antennas positions in the  shower reference frame (v, vxB, vxvxB)
        x = np.zeros(n)   
        y = np.zeros(n)
        z = np.zeros(n)
        
        #Traces in the shower reference frame
        Traces_Ex = np.zeros([time_sample,n])
        Traces_Ey = np.zeros([time_sample,n])
        Traces_Ez = np.zeros([time_sample,n])
        Time = np.zeros([time_sample, n])
        
        # unit vectors 
        uv = self.showerdirection()
        uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field 
        
        uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
        uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
        
        uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
        uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
        
        P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
        
        # We calculate the positions in the shower plane
        Position_shower = np.array([v,vxb,vxvxb]) # position in the geographic reference frame
        Position_ground = np.dot(P, Position_shower) # position in the shower reference frame
        
        # We deduce the different components
        x = Position_ground[0, :] 
        y = Position_ground[1, :]
        z =  Position_ground[2, :]
        
        # We calulate the traces in the shower plane
        Traces_shower = np.zeros([time_sample,3])
        Traces_geo_temp = np.zeros([3, time_sample])
        
        for i in range(n):
            
            Traces_shower = np.array([Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i + 3*n]])
            
            Traces_geo_temp = np.dot(P, Traces_shower)
            
            Traces_Ex[:,i] = np.transpose(Traces_geo_temp[0,:]) # Ev component of the traces
            Traces_Ey[:,i] = np.transpose(Traces_geo_temp[1,:]) # Evxb component of the traces
            Traces_Ez[:,i] = np.transpose(Traces_geo_temp[2,:]) # Evxvxb component of the traces
            
            Time[:,i] = Traces[:,i]
        
        # We derive the traces in the shower plane
        
        Traces_ground = np.transpose(np.concatenate((np.transpose(Time), \
        np.transpose(Traces_Ex), np.transpose(Traces_Ey), np.transpose(Traces_Ez))))
        
        core = self.get_center()
        
        x = x + core[0]
        y = y + core[1] 
        z = z + core[2]
    
        Positions_ground = np.zeros([n,3]) # To check
        Positions_ground[:,0], Positions_ground[:,1], Positions_ground[:,2] = x, y, z
        
    
        return Positions_ground, Traces_ground
    
    
# =============================================================================
#                         Analysis functions
# =============================================================================
    
def CerenkovStretch(RefShower, TargetShower, GroundPlane = True):
    
    if(GroundPlane):
        TargetShower.pos, TargetShower.traces = TargetShower.GetinShowerPlane()
    
    Nant = RefShower.nant
    
    cerangle_ref = RefShower.get_cerenkov_angle()
    cerangle_target = TargetShower.get_cerenkov_angle()
    
    kstretch = cerangle_ref/cerangle_target

    w = RefShower.get_w()/kstretch
    
    v, vxb, vxvxb =  TargetShower.pos[:,0], \
    TargetShower.pos[:,1], TargetShower.pos[:,2]
    eta = np.arctan2(vxvxb, vxb)
    Distplane = TargetShower.distplane  
    d = Distplane*np.tan(w*np.pi/180.0)
    
    vxb_scaled = d*np.cos(eta) 
    vxvxb_scaled = d*np.sin(eta)
    
    scaled_pos = np.array([v,vxb_scaled, vxvxb_scaled]).T
    scaled_traces = TargetShower.traces[:,Nant:]*kstretch
                
    return scaled_pos, scaled_traces, kstretch
    
   
    
def CreateShowerfromHDF5(HDF5filepath):

    Primary, E0, zenith, azimuth, Bgeo, glevel, XmaxParam, Pos, Nant, Traces_C, Traces_G = LoadHDF5file(HDF5filepath)
    
    _Shower = Shower(Primary, E0, zenith, azimuth, 1e5, Nant, Bgeo, glevel, XmaxParam,
                        Pos, Traces_C, Traces_G)

    return _Shower  
