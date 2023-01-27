#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:12:36 2023

@author: akimlavrinenko
"""

import numpy as np
import matplotlib.pyplot as plt


S = 1      # generation rate
# D = 0.001 # turbulent diffusion coefficient
u = 0.0856   # constant velocity
# Dimensions of the cube
Lxi = 1
Leta = 1
Lz = 1 
#source position
xi0 =0.5  # initial xi position of the source
eta0=0.5 # initial eta position of the source
z0  =0   # initial z position of the source

# grid for xi
dxi = 0.05 # grid spacing
nxi = int((4/dxi))+1 # four cubes
xi = np.linspace(0, 4, nxi) # four cubes
nx = int(((nxi-1)/4))+1 # four cubes

# grid for eta
neta = nx  # number of points
eta = np.linspace(0, 1, neta)

# grid for z
nz = neta
z = np.linspace(Lz/-2, Lz/2, nz)

# grid for x and y (equal to the grid of eta)
ny = neta
y = eta
x = eta

X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y

# positions of the grid nodes of the mesh xi,eta,z
ETA,XI,Z = np.meshgrid(eta, xi, z)

# times to perform the calculations
tmin = 0
tmax = 200
dt = 0.1
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

vmax = 25
levels = np.linspace(0, vmax, nx+1)

linestyle = 'o'
# label = 'K = ' + str(klist[0])
color = 'k'
            
klist = [1e-1, 1e-2, 1e-3, 1e-4]
for D in klist:
    sigmalist = []
    sigmaTsigma0list = []
    for tt in t:
        sigma = np.zeros((nx, ny, nz))
        for i in range(nxi):
            for j in range(neta):
                for k in range(nz):
                    
                    t1 = (xi[i] - xi0 - u*tt)**2 / (4*D*tt)
                    #refklections along z
                    t1Rz = (z[k] - z0)**2 / (4*D*tt)
                    t2Rz = (z[k] + z0 - Lz)**2 / (4*D*tt)
                    t3Rz = (z[k] + z0 + Lz)**2
                    #refklections along eta
                    t1Reta = (eta[j] - eta0)**2 / (4*D*tt)
                    t2Reta = (eta[j] + eta0)**2 / (4*D*tt)
                    t3Reta = (eta[j] + eta0 - 2*Leta)**2 / (4*D*tt)
                    
                    
                    Rz = np.exp(-t1Rz) + np.exp(-t2Rz) + np.exp(-t3Rz)
                    Reta = np.exp(-t1Reta) + np.exp(-t2Reta) + np.exp(-t3Reta)
                    
                    C[i,j,k] = (S/(8*(np.pi*tt*D)**(3/2))) * np.exp(-t1) * Rz * Reta
                    
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    C1[i,j,k]=C[i,j,k]
                    C2[i,j,k]=C[nx-1+i,j,k]
                    C3[i,j,k]=C[2*(nx-1)+i,j,k]
                    C4[i,j,k]=C[3*(nx-1)+i,j,k]
                    
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    C2m[i,j,k]=C2[j,nx-1-i,k]
                    C3m[i,j,k]=C3[nx-1-i,ny-1-j,k]
                    C4m[i,j,k]=C4[ny-1-j,i,k]
    
        Ctot = C1 + C2m + C3m + C4m
        ctot2d = Ctot[:, :, int((nz + 1) / 2)]
        
        # fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
        # time = str(int(round(tt,3))).zfill(3)
        # ax.set_title('V: ' + str(u) + ', time: ' +  str(time) + ' s')
        # cp = ax.contourf(X, Y, ctot2d, levels=levels, vmin=0, vmax=vmax)
        # plt.colorbar(cp)
        # plt.axis('square')
        # plt.ylabel('y, m')
        # plt.xlabel('x, m')
        # fig.tight_layout()
        # # plt.savefig('./conc_' + tt + '.png', dpi = 100)
        # plt.show()
        
        if tt == t[0]:
            Cinf = sum(sum(sum(Ctot)))*dxi**3 / (Lxi*Leta*Lz)
            
        # print(Cinf)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                
                    sigma[i][j][k] = (Ctot[i,j,k] - Cinf)**2
        
        sigma = np.sqrt(sum(sum(sum(sigma))) / (dxi**3))
        print(tt, 'sigma: ', sigma)
        sigmalist.append(sigma)
        itemindex = np.where(t == tt)[0][0]
        sss = sigmalist[itemindex]/sigmalist[0]
        sigmaTsigma0list.append(sss)
        plt.plot(tt, sss, marker = linestyle, color = color, markersize = 3)
    plt.ylabel('sigma[t]/sigma[t=0.4]')
    plt.xlabel('t, s')
    plt.legend(loc="upper right")
    plt.ylim(0,1.1)
    # plt.savefig('./sigma2.png', dpi = 200)
    plt.show()