import numpy as np
import matplotlib.pyplot as plt

ZenithBins = np.array([0,20,28,34,39,43,47,50])
azimuth = 90*np.pi/180.0
Bx = 7.705
By = 0
Bz = -54.111

B = np.sqrt(Bx**2 + By**2 + Bz**2)

ub = np.array([Bx, By, Bz])/B

sin_alpha = np.zeros(len(ZenithBins))

sin_alpha_all_phi = dict()
phi_all = np.array([0,90,270])*np.pi/180.0

for j in range(len(phi_all)):
    azimuth = phi_all[j]
    sin_alpha = np.zeros(len(ZenithBins))

    for i in range(len(ZenithBins)):
        print(azimuth)
        zenith = ZenithBins[i]*np.pi/180.0
        ux = np.sin(zenith)*np.cos(azimuth)
        uy = np.sin(zenith)*np.sin(azimuth)
        uz = np.cos(zenith)
        uv = np.array([ux, uy, uz])

        scalar = np.dot(uv, ub)
        alpha_rad = np.arccos(scalar)
        sin_alpha[i] = np.sin(alpha_rad)
    sin_alpha_all_phi[j]= sin_alpha

plt.scatter(ZenithBins, sin_alpha, label = "$B_{geo}$ = $B_{summit}$")
plt.xlabel("zenith [Deg.]")
plt.ylabel("$\sin{\\alpha}$")
plt.legend()
#plt.savefig("/Users/chiche/Desktop/sin_alpha_vs_zen.pdf", bbox_inches="tight")
plt.show()


phi_all = phi_all*180/np.pi
plt.scatter(ZenithBins, sin_alpha_all_phi[0], label = r"$\varphi = %.d$" %phi_all[0])
plt.scatter(ZenithBins, sin_alpha_all_phi[1], label = r"$\varphi = %.d$" %phi_all[1])
plt.scatter(ZenithBins, sin_alpha_all_phi[2], label = r"$\varphi = %.d$" %phi_all[2])
plt.xlabel("zenith [Deg.]")
plt.ylabel("$\sin{\\alpha}$")
plt.legend()
#plt.savefig("/Users/chiche/Desktop/sin_alpha_vs_zen.pdf", bbox_inches="tight")
plt.show()
