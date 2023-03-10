#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:12:36 2023

@author: akimlavrinenko
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import nsum, inf

def R1 (k):
    ss = nsum(lambda m: np.e**(-((z[k] + 2*m*Lxi - z0)**2 / (4*D*tt))) + np.e**(-((z[k] + 2*m*Lxi + z0)**2 / (4*D*tt))), [-inf, inf])
    return ss

def R2 (j):
    ss = nsum(lambda m: np.e**(-((eta[j]  + 2*m*Lxi - eta0)**2 / (4*D*tt))) + np.e**(-((eta[j]  + 2*m*Lxi + eta0)**2 / (4*D*tt))), [-inf, inf])
    return ss

S = 1      # generation rate
D = 0.0152 # turbulent diffusion coefficient
u = 0.5 #0.0856 # constant velocity
# Dimensions of the cube
Lxi = 1
Leta = 1
Lz = 1

#source position
xi0 = 0.3  # initial xi position of the source
eta0= 0.3  # initial eta position of the source
z0  = 0.5     # initial z position of the source

# grid for xi
dxi = 0.025 # grid spacing
nx = int(np.ceil(Lxi/dxi)) # four cubes
nxi = int(4*nx) # four cubes
xi = np.linspace(0, 4*Lxi, nxi) # four cubes


# grid for eta
neta = nx  # number of points
eta = np.linspace(0, Leta, neta)

# grid for z
nz = neta
z = np.linspace(0, Lz, nz)

# grid for x and y (equal to the grid of eta)
ny = neta
y = eta
x = eta

X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y

# positions of the grid nodes of the mesh xi,eta,z
ETA,XI,Z = np.meshgrid(eta, xi, z)

# times to perform the calculations
tmin = 0
tmax = 100
dt = 1
t = np.linspace(tmin, tmax, int(tmax/dt)+1)
t = t[1:]
C = np.zeros((nxi, neta, nz))
C1 = np.zeros((nx, ny, nz))
C2 = np.zeros((nx, ny, nz))
C3 = np.zeros((nx, ny, nz))
C4 = np.zeros((nx, ny, nz))
C2m = np.zeros((nx, ny, nz))
C3m = np.zeros((nx, ny, nz))
C4m = np.zeros((nx, ny, nz))

vmax = 1
levels = np.linspace(0, vmax, nx+1)

linestyle = 'o'
# label = 'K = ' + str(klist[0])
color = 'k'

mRange = 5

c = 0
n_t = 0

sigmaTsigma0list = []
sigmalist = []
for tt in t:
    sigma = np.zeros((nx, ny, nz))

    for i in range(nxi):
        for j in range(neta):
            for k in range(nz):
                
                if c < 2:
                    t1_1 = ((xi[i] - xi0) - u*tt)**2 / (4*D*tt)
                    t1_2 = ((xi[i] - xi0 + 4*Lxi) - u*tt)**2 / (4*D*tt)
                
                elif c >= 2:
                   # print('hi')
                    t1_1 = ((xi[i] - xi0) - u*(tt - n_t))**2 / (4*D*tt)
                    t1_2 = ((xi[i] + xi0 + 4*Lxi) - u*(tt - n_t))**2 / (4*D*tt)
                    
                #refklections along z
                Rz = 0
                for m in range(-3,4):
                    Rz += np.exp(-((z[k] + 2*m*Lxi - z0)**2 / (4*D*tt))) + np.exp(-((z[k] + 2*m*Lxi + z0)**2 / (4*D*tt)))
                    # print(m, Rz)
                #refklections along eta
                Reta = 0
                for m in range(-3,4):
                    # print(m)
                    Reta += np.exp(-((eta[j]  + 2*m*Lxi - eta0)**2 / (4*D*tt))) + np.exp(-((eta[j]  + 2*m*Lxi + eta0)**2 / (4*D*tt)))
                                   
                t1 = np.exp(-t1_1) + np.exp(-t1_2)
               
                
                C[i,j,k] = (S/(8*(np.pi*tt*D)**(3/2))) * t1 * Rz * Reta
        # print(Reta1, Reta)
    # print(c, sum(sum(sum(C)))*dxi**3 / (Lxi*Leta*Lz))
    if (xi0+u*tt)  > xi[-1]*c:
        c += 1
        n_t = tt
                
            
    time = str(round(tt,3)).zfill(3)
    fig,ax = plt.subplots(1,1, figsize=(8, 2))
    ax.set_title('V: ' + str(u) + ', time: ' +  str(time) + ' s')
    cp = ax.contourf(XI[:,:,int((nz + 1) / 2)], ETA[:,:,int((nz + 1) / 2)], C[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # plt.colorbar(cp)
    # plt.axis('square')
    plt.ylabel('y, m')
    plt.xlabel('x, m')
    fig.tight_layout()
    # plt.savefig('./conc_' + tt + '.png', dpi = 100)
    plt.show()
                
    
    C1 = C[:int(np.ceil(len(C)/4)),:,:]
    C2 = C[int(np.ceil(len(C)/4)):int(np.ceil(len(C)/4))*2,:,:]
    C3 = C[2*int(np.ceil(len(C)/4)):(int(np.ceil(len(C)/4)))*3,:,:]
    C4 = C[3*int(np.ceil(len(C)/4)):,:,:]
    
    C2m = np.rot90(C2,k=1, axes = (0,1))
    C3m = np.rot90(C3,k=2, axes = (0,1))
    C4m = np.rot90(C4,k=3, axes = (0,1))

    Ctot = C1 + C2m + C3m + C4m
    # ctot2d = Ctot[:, :, int((nz + 1) / 2)]
    
    
    # time = str(round(tt,3)).zfill(3)
    # fig,ax = plt.subplots(figsize=(5, 5))
    # fig.suptitle('V: ' + str(u) + ', time: ' +  str(time) + ' s')
    # ax.contourf(Y, X, Ctot[:,:,-3], levels=levels, vmin=0, vmax=vmax)
    # ax[0,0].contourf(Y, X, C1[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # ax[0,1].contourf(Y, X, C2[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # ax[1,0].contourf(Y, X, C3[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # ax[1,1].contourf(Y, X, C4[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # ax[2,0].contourf(Y, X, ctot2d, levels=levels, vmin=0, vmax=vmax)
    # plt.colorbar(cp)
    # plt.axis('square')
    # plt.ylabel('y, m')
    # plt.xlabel('x, m')
    # fig.tight_layout()
    # plt.savefig('./conc_' + tt + '.png', dpi = 100)
    # plt.show()
    
    # if tt == t[0]:
    Cinf = sum(sum(sum(Ctot)))*dxi**3 / (Lxi*Leta*Lz)
    # Cinf = Cinf    
    print(tt, Cinf)
    # for i in range(nx):
    #     for j in range(ny):
    #         for k in range(nz):
    #             sigma[i][j][k] = (Ctot[i,j,k] - Cinf)**2
    
    # sigma = np.sqrt(sum(sum(sum(sigma))) / (nx*ny*nz))
    # sigmalist.append(sigma)
    # itemindex = np.where(t == tt)[0][0]
    # # print(itemindex,tt)
    # sss = sigmalist[itemindex]/sigmalist[0]
    # sigmaTsigma0list.append(sss)
    
    #print(tt, 'sigmaTsigma0: ', sss)
    # if D == klist[0]:
    #     linestyle = 'o'
    #     label = 'K = ' + str(klist[0])
    #     color = 'k'
    # elif D == klist[1]:
    #     linestyle = 's'
    #     label = 'K = ' + str(klist[1])
    #     color = 'r'
    # elif D == klist[2]:
    #     linestyle = '*'
    #     label = 'K = ' + str(klist[2])
    #     color = 'b'
    # elif D == klist[3]:
    #     linestyle = 'v'
    #     label = 'K = ' + str(klist[3])
    #     color = 'c'
    # elif D == klist[4]:
    #     linestyle = "^"
    #     label = 'K = ' + str(klist[4])
    #     color = 'y'
    
#     if tt == t[0]:
       
#         plt.plot(tt, sss, marker = linestyle, color = color, markersize = 3, label = label)
#     else:
#         plt.plot(tt, sss, marker = linestyle, color = color, markersize = 3)
    # plt.plot(tt,Cinf, 'ro')
# res = np.asarray([t,sigmaTsigma0list]).T
# np.savetxt('./sigma_pallares_' + str(D) + '_' + str(dxi) +  '.dat', res)
# plt.ylabel('sigma[t]/sigma[t=0]')
# plt.ylabel('sum(Ctot)*dxi**3 / (Lxi*Leta*Lz)')
# plt.xlabel('t, s')
# plt.legend(loc="upper right")
# plt.ylim(0,1.1)
# plt.savefig('./sigma_pallares.png', dpi = 200)
# plt.show()

