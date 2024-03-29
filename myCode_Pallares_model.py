#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:12:36 2023

@author: akimlavrinenko
"""

import numpy as np
import matplotlib.pyplot as plt


S = 30   # generation rate
D = 0.02 # turbulent diffusion coefficient
u = 0.0856 # constant velocity
# Dimensions of the cube
Lxi = 3.1
Leta = 3.1
Lz = 3.1

#source position
xi0 = 0.6  # initial xi position of the source
eta0= 0.6  # initial eta position of the source
z0  = 2.5     # initial z position of the source

tmin = 0
tmax = 10
dt = 0.1

# grid for xi
dxi = 0.15 # grid spacing
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
color = 'k'

c1 = 0
c2 = Lxi*4
c3 = Lxi*8
c4 = Lxi*12
cc = 0
c = 0

r = int(np.ceil(tmax * u + xi0 / 4))
n = int(np.ceil(r / 4) * 4)
cList = np.linspace(1, n, n)
cArr = np.reshape(cList, (int(n / 4), 4)).T
ccArr = np.zeros(np.shape(cArr))
ccArr[0,0] = c1
ccArr[1,0:2] = c2
ccArr[2,0:3] = c3
ccArr[3,0:4] = c4



for i in range(0,len(ccArr[0,:])):
    for j in range(0, len(ccArr[:,0])):
        ccArr[j,i*4+1+j:i*4+5+j] = (i+1)*Lxi*16 + ccArr[j,0]

sigmaTsigma0list = []
sigmalist = []

CinfList= [] 
for tt in t:
    sigma = np.zeros((nx, ny, nz))
    C = np.zeros((nxi, neta, nz))
    C1 = np.zeros((nx, ny, nz))
    C2 = np.zeros((nx, ny, nz))
    C3 = np.zeros((nx, ny, nz))
    C4 = np.zeros((nx, ny, nz))
    C2m = np.zeros((nx, ny, nz))
    C3m = np.zeros((nx, ny, nz))
    C4m = np.zeros((nx, ny, nz))

    for i in range(nxi):
        for j in range(neta):
            for k in range(nz):
                c1 = ccArr[0,c]
                c2 = ccArr[1,c]
                c3 = ccArr[2,c]
                c4 = ccArr[3,c]
                
                t1_0 = ((xi[i] - xi0 + c1) - u*tt)**2 / (4*D*tt)
                t1_1 = ((xi[i] - xi0 + c2) - u*tt)**2 / (4*D*tt)
                t1_2 = ((xi[i] - xi0 + c3) - u*tt)**2 / (4*D*tt)
                t1_3 = ((xi[i] - xi0 + c4) - u*tt)**2 / (4*D*tt)
                
                t1 = np.exp(-t1_0) + np.exp(-t1_1) + np.exp(-t1_2) + np.exp(-t1_3)
                # #refklections along z
                Rz = 0
                for m in range(-3,4):
                    Rz += np.exp(-((z[k] + 2*m*Lxi - z0)**2 / (4*D*tt))) + np.exp(-((z[k] + 2*m*Lxi + z0)**2 / (4*D*tt)))
                #refklections along eta
                Reta = 0
                for m in range(-3,4):
                    Reta += np.exp(-((eta[j]  + 2*m*Lxi - eta0)**2 / (4*D*tt))) + np.exp(-((eta[j]  + 2*m*Lxi + eta0)**2 / (4*D*tt)))
                    
                C[i,j,k] = (S/(8*(np.pi*tt*D)**(3/2))) * t1 * Rz * Reta
                
                C[C<S*1e-15] = 0
    
    if (xi0+u*tt)  >= xi[-1] * (c+2):
            c += 1
            
    # time = str(round(tt,3)).zfill(3)
    # fig,ax = plt.subplots(1,1, figsize=(8, 2))
    # ax.set_title('V: ' + str(u) + ', time: ' +  str(time) + ' s')
    # # cp = ax.contourf(XI[:,:,int((nz + 1) / 2)], ETA[:,:,int((nz + 1) / 2)], C[:,:,int((nz + 1) / 2)], levels=levels, vmin=0, vmax=vmax)
    # cp = ax.contourf(XI[:,:,13], ETA[:,:,13], C[:,:,13], levels=levels, vmin=0, vmax=vmax)
    # # plt.colorbar(cp)
    # # plt.axis('square')
    # plt.ylabel('y, m')
    # plt.xlabel('x, m')
    # fig.tight_layout()
    # # plt.savefig('.mcPm_figs/conc_' + str(int(tt)) + '.png', dpi = 100)
    # plt.show()
                
    
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
    
    Cinf = sum(sum(sum(Ctot)))*dxi**3 / (Lxi*Leta*Lz)
    # print(tt, np.round(Cinf,4), np.round((xi0+u*tt),4), np.round(xi[-1] * (c+2), 4),'|', c, c1, c2, c3, c4)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                sigma[i][j][k] = (Ctot[i,j,k] - Cinf)**2
    
    sigma = np.sqrt(sum(sum(sum(sigma))) / (nx*ny*nz))
    print(sigma)
    print('------')
    sigmalist.append(sigma)
    
res = np.asarray([t,sigmalist]).T
np.savetxt('./sigma_pallares_' + str(D) + '_' + str(dxi) + '_' + str(dxi) + '.dat', res)
