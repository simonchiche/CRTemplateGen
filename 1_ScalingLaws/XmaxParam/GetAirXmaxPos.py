import numpy as np
from AirDensityParam import rho_corsika
from XmaxParam import Xmax_param_QGSJETII

def _get_CRzenith(zenith, glevel, injection):
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
    zen = zenith

    GdAlt = glevel
    injh = injection
            
    Re= 6370949 # m, Earth radius

    a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 *np.sin(np.pi-np.deg2rad(zen))**2) - (Re+GdAlt)*np.cos(np.pi-np.deg2rad(zen))
    zen_inj = np.rad2deg(np.pi-np.arccos((a**2 +(Re+injh)**2 -Re**2)/(2*a*(Re+injh))))
    
    
    return zen_inj 

def _getAirDensity(_height, model):

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


def _dist_decay_Xmax(Xmax_primary, zenith, glevel, injection): 
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
    AtmPath = "/Users/chiche/Desktop/FAERIEparam/1_ScalingLaws/XmaxParam/Data/Atmosphere.dat"
    zen = _get_CRzenith(zenith, glevel, injection)
    injh2 = injection
    zen2 = np.deg2rad(zen)
    
    hD=injh2
    step=10 #m
    if hD>10000:
        step=10 #m
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
        rho = rho_corsika(h, AtmPath) # g/cm3
        X=X+ rho * step*100. #(deltah*100) *abs(1./np.cos(np.pi-zen2)) # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta); density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
        
    return h, ai # Xmax_height in m, Xmax_distance in m    

def getGroundXmaxDistance(Xmax_primary, zenith, glevel, injection):

    # Xmax_primary in g/cm2
    # zenith in cosmic ray convention in degrees
    # ground altitude above sealevel in m
    # injection height above sealevel in m

    
    XmaxHeight, DistDecayXmax = _dist_decay_Xmax(Xmax_primary, zenith, glevel, injection)
    GroundAltitude = glevel
    zenith = np.deg2rad(zenith)
    Rearth = 6370949 
    
    dist = np.sqrt((Rearth+ XmaxHeight)**2 - ((Rearth + GroundAltitude)*np.sin(zenith))**2) \
    - (Rearth + GroundAltitude)*np.cos(zenith)

    return dist   

def showerdirection(zenith, azimuth):

    # zenith in cosmic ray in degrees
    # azimuth in degrees
    
    zenith = 180 - zenith

    zenith = zenith*np.pi/180.0
    azimuth = azimuth*np.pi/180.0
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), \
                    np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
    
    return uv

def getXmaxPosition(azimuth, zenith, glevel, injection, showerDistance, Xmax_primary =0):

    # Xmax_primary in g/cm2
    # zenith in cosmic ray convention in degrees
    # azimuth in degrees
    # ground altitude above sealevel in m
    # injection height above sealevel in m

    
    uv = showerdirection(zenith, azimuth)
    print(Xmax_primary)
    if(Xmax_primary != 0):
        showerDistance = getGroundXmaxDistance(Xmax_primary, zenith, glevel, injection)
    print(showerDistance, "shower distance")
    XmaxPosition = -uv*showerDistance 
    XmaxPosition[2] = XmaxPosition[2] + glevel  
            
    return XmaxPosition, showerDistance


def Xmax_param(primary, energy, fluctuations):

    
    if(primary == 'Iron'):
        a =65.2
        c =270.6
        
        Xmax = a*np.log10(energy*1e6) + c

    elif(primary == 'Proton'):
        a = 57.4
        c = 421.9
        Xmax = a*np.log10(energy*1e6) + c

    return Xmax
    

# Iron 661
# proton 766


### CASE 1 ###

Azimuth = 134
Zenith = 34
injection = 1e6
glevel = 2800
Dxmax = 0

primary = "Proton"

E = 1e18 # eV
Xmax_proton_QGSJETII, sigma_proton_QGSJETII = Xmax_param_QGSJETII(E, 1) 
Xmax = Xmax_proton_QGSJETII
XmaxPosition, Dxmax_p_E18 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)

E= np.sqrt(10)*1e18
Xmax_proton_QGSJETII, sigma_proton_QGSJETII = Xmax_param_QGSJETII(E, 1) 
Xmax = Xmax_proton_QGSJETII
XmaxPosition, Dxmax_p_E18_5 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)

E= 1e19
Xmax_proton_QGSJETII, sigma_proton_QGSJETII = Xmax_param_QGSJETII(E, 1) 
Xmax = Xmax_proton_QGSJETII
XmaxPosition, Dxmax_p_E19 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)

primary = "Iron"
E= 1e18
Xmax_iron_QGSJETII, sigma_iron_QGSJETII = Xmax_param_QGSJETII(E, 56) 
Xmax = Xmax_iron_QGSJETII
XmaxPosition, Dxmax_fe_E18 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)


E= np.sqrt(10)*1e18
Xmax_iron_QGSJETII, sigma_iron_QGSJETII = Xmax_param_QGSJETII(E, 56) 
Xmax = Xmax_iron_QGSJETII
XmaxPosition, Dxmax_fe_E18_5 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)

E= 1e19
Xmax_iron_QGSJETII, sigma_iron_QGSJETII = Xmax_param_QGSJETII(E, 56) 
Xmax = Xmax_iron_QGSJETII
XmaxPosition, Dxmax_fe_E19 = getXmaxPosition(Azimuth, Zenith, glevel, injection, Dxmax, Xmax)


Dxmax_p = np.array([Dxmax_p_E18, Dxmax_p_E18_5, Dxmax_p_E19])
Dxmax_fe = np.array([Dxmax_fe_E18, Dxmax_fe_E18_5, Dxmax_fe_E19])

DataPath = "/Users/chiche/Desktop/FAERIEparam/1_ScalingLaws/XmaxParam/Data/"
np.savetxt(DataPath + "Dxmax_p.txt", Dxmax_p)
np.savetxt(DataPath + "Dxmax_fe.txt", Dxmax_fe)