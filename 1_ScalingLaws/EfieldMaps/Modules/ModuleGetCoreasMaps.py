import numpy  as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import trapz, simps
from scipy.interpolate import Rbf



def GetCoreasTracesfromHDF5(HDF5filepath):
    k =0
    Traces_C = dict()
    with h5py.File(HDF5filepath, "r") as f:
        #print("keys")
        observers_coreas = f["CoREAS"]["observers"]  # Navigate to the group
        keys_coreas = list(observers_coreas.keys())  # List all datasets inside
        #print(keys_coreas)
        #print("keys")
        #print("Available keys:", keys)
        for key_c in keys_coreas:
            #print(key_c, key_g)
            Traces_C[k] = observers_coreas[key_c][()]
            k = k +1
        
        return Traces_C
    

def GetPeakTraces(Traces):

    Nant = len(Traces)
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

def PlotCoreasMaps(Shower, Etot):

    Pos = Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths() 
    for i in range(Nlay):
        sel = (Pos[:,2] == Depths[i])
        Elog = np.log10(Etot[sel] +1)

        plt.scatter(Pos[sel,0], Pos[sel,1], \
                    c= Etot[sel]+1, cmap = "jet", s=10, edgecolors='k', linewidth=0.2)
        cbar = plt.colorbar()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        cbar.set_label("$E$ [$\mu V/m$]")
        plt.legend(["Depth = %.f m" %(Depths[i])], loc ="upper right")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.title("Full band Efield (E =$%.2f\,$EeV, $\\theta=%.1f^{\circ}$)" %(Shower.energy, Shower.zenith), size =14)
        OutputFolder = "/Users/chiche/Desktop/AntennaGridMaps/"
        #plt.savefig(OutputFolder + "Efield_E%.2f_th%.1f_|z|%.d_smalldots.pdf"  %(Shower.energy, Shower.zenith, Depths[i]), bbox_inches = "tight")
        plt.show()

    return

def GetFluence(Traces):
    eps0 = 8.85e-12 # F.m^{-1}
    c = 3e8 # m.s^{-1}

    Nant = len(Traces)
    
    ftot = np.zeros(Nant)
    fx, fy, fz = np.zeros(Nant), np.zeros(Nant), np.zeros(Nant)
    binT = round((Traces[0][1,0] -Traces[0][0,0])*1e10)/1e10
    print(binT)

    for i in range(Nant):
        
        ftot_t = Traces[i][:,1]**2 + Traces[i][:,2]**2 + Traces[i][:,3]**2
        extent = 10000
        peak_id = np.argmax(ftot_t)
        minid = peak_id -extent
        maxid = peak_id + extent
        if(minid<0): minid = 0
        if(maxid>len( Traces[i][:,0])): maxid =len( Traces[i][:,0])
        
        time = np.arange(0, len(Traces[i][minid:maxid,0]))*binT
        
        fx[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,1]**2)), time)#/1e12
        fy[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,2]**2)), time)#/1e12
        fz[i] = eps0*c*simps(abs(hilbert(Traces[i][minid:maxid,3]**2)), time)#/1e12
        ftot[i] = eps0*c*(fx[i] + fy[i] + fz[i])
    print(fy,ftot)
    return fx, fy, fz, ftot


### Integral
def interpolate_rbf(x, y, z, grid_resolution=100, bounds=None, function='cubic'):
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



def GetRadiationEnergyFromInterpolation(Shower, ftot):
    
    Pos = Shower.pos
    Nlay, Nplane, Depths = Shower.GetDepths()

    Erad = np.zeros(len(Depths))
    for i in range(len(Depths)):
        sel = (Pos[:,2] == Depths[i])

        grid_x, grid_y, grid_z = \
        interpolate_rbf(Pos[:,0][sel], Pos[:,1][sel], ftot[sel])
        
        if(i ==0):
            plt.figure(figsize=(6, 5))
            plt.contourf(grid_x, grid_y, np.log10(grid_z), levels=100, cmap='jet')
            #plt.scatter(Pos[:,0][sel], Pos[:,1][sel], s =0.1)
            #plt.scatter(Pos[:729,0], Pos[:729,1], c=np.log10(EtotC_int[:729] +1), edgecolor='white', s=100)
            plt.colorbar(label="$\log_{10}(E)$ [$\mu Vs/m$]")
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.show()
            

        # First, compute the integral along one axis (e.g., x), then along the other (e.g., y)
        integral_x = trapz(grid_z, x=grid_x[0], axis=1)  # integrate over x (axis=1)
        total_integral = trapz(integral_x, x=grid_y[:,0])  # integrate over y (axis=0)
        Erad[i] = total_integral

        return Erad
    

def PlotEradvsDepths(Depths, EradAllDepths):

    plt.scatter(Depths, EradAllDepths)
    ymin = max(EradAllDepths/3)
    ymax = max(EradAllDepths*1.5)
    plt.ylim(ymin, ymax)
    plt.xlabel("Depth [m]")
    plt.ylabel("$E_{rad}$")
    plt.grid()
    #plt.savefig("/Users/chiche/Desktop/Rectangle_Erad_vs_depth_E0.32_th43.pdf", bbox_inches = "tight")
    plt.show()