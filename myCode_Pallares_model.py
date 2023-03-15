#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:12:36 2023

@author: akimlavrinenko
"""

import numpy as np
import matplotlib.pyplot as plt


S = 1      # generation rate
D = 0.00152 # turbulent diffusion coefficient
u = 0.1 #0.0856 # constant velocity
# Dimensions of the cube
Lxi = 1
Leta = 1
Lz = 1

#source position
xi0 = 0.2  # initial xi position of the source
eta0= 0.2  # initial eta position of the source
z0  = Lz/2     # initial z position of the source

# grid for xi
dxi = 0.05 # grid spacing
nx = int(np.ceil(Lxi/dxi)) # four cubes
nxi = int(4*nx) # four cubes
xi = np.linspace(0, 4*Lxi, nxi) # four cubes


# grid for eta
neta = nx  # number of points
eta = np.linspace(0, Leta, neta)

# grid for z
nz = 5 #neta
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
tmax = 200
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

c1 = 0
c2 = 4
c3 = 8
c4 = 12
cc = 0
c = 0

r = int(np.ceil(tmax * u + xi0 / 4))
n = int(np.ceil(r / 3) * 3)
cList = np.linspace(1, n, n)
cArr = np.reshape(cList, (int(n / 3), 3)).T
ccArr = np.zeros(np.shape(cArr))
ccArr[0,0] = c1
ccArr[1,0:2] = c2
ccArr[2,0:3] = c3


for i in range(0,len(ccArr[0,:])):
    for j in range(0, len(ccArr[:,0])):
        ccArr[j,i*3+1+j:i*3+4+j] = (i+1)*12 + ccArr[j,0]

sigmaTsigma0list = []
sigmalist = []

CinfList= [] 
for tt in t:
    sigma = np.zeros((nx, ny, nz))

    for i in range(nxi):
        for j in range(neta):
            for k in range(nz):
                c1 = ccArr[0,c]
                c2 = ccArr[1,c]
                c3 = ccArr[2,c]
                
                t1_0 = ((xi[i] - xi0 + c1*Lxi) - u*tt)**2 / (4*D*tt)
                t1_1 = ((xi[i] - xi0 + c2*Lxi) - u*tt)**2 / (4*D*tt)
                t1_2 = ((xi[i] - xi0 + c3*Lxi) - u*tt)**2 / (4*D*tt)
                
                t1 = np.exp(-t1_0) + np.exp(-t1_1) + np.exp(-t1_2)
                # #refklections along z
                Rz = 0
                for m in range(-3,4):
                    Rz += np.exp(-((z[k] + 2*m*Lxi - z0)**2 / (4*D*tt))) + np.exp(-((z[k] + 2*m*Lxi + z0)**2 / (4*D*tt)))
                #refklections along eta
                Reta = 0
                for m in range(-3,4):
                    Reta += np.exp(-((eta[j]  + 2*m*Lxi - eta0)**2 / (4*D*tt))) + np.exp(-((eta[j]  + 2*m*Lxi + eta0)**2 / (4*D*tt)))
                    
                C[i,j,k] = (S/(8*(np.pi*tt*D)**(3/2))) * t1 * Rz * Reta
    
    if (xi0+u*tt)  >= xi[-1] * (c+2):
            c += 1
            
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
    print(tt ,(xi0+u*tt), xi[-1] * (c+2),'|', c, c1, c2, c3)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                sigma[i][j][k] = (Ctot[i,j,k] - Cinf)**2
    
    sigma = np.sqrt(sum(sum(sum(sigma))) / (nx*ny*nz))
    sigmalist.append(sigma)
    itemindex = np.where(t == tt)[0][0]
    # print(itemindex,tt)
    sss = sigmalist[itemindex]/sigmalist[0]
    sigmaTsigma0list.append(sss)
    
    CinfList.append(Cinf)
    
res = np.asarray([t,sigmaTsigma0list]).T
np.savetxt('./sigma_pallares_' + str(D) + '_' + str(dxi) +  '.dat', res)

plt.plot(t,CinfList, 'ro')
plt.ylabel('sum(Ctot)*dxi**3 / (Lxi*Leta*Lz)')
plt.xlabel('t, s')
plt.savefig('./sumC_pallares.png', dpi = 200)
plt.show()


plt.plot(res[:,0], res[:,0])
plt.ylabel('sigma[t]/sigma[t=0]')
plt.xlabel('t, s')
plt.legend(loc="upper right")
plt.ylim(0,1.1)
plt.savefig('./sigma_pallares.png', dpi = 200)
plt.show()

