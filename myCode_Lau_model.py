#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:48:46 2023

@author: akimlavrinenko
"""

import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library

#Parameters values -- User inputs floats
time = 1 #event duration, in seconds
time2 = 3
delta_t = 1 #(s) time-steps
delta_t2 = 1 #(s) time-steps
# time2 = 5
l = 3 #x-length (m) of room
w = 3 #y-length (m) of room
h = 3 #z-length (m) of room
x_o = 0.5 #x-coordinate of source
y_o = 1.5 #y-coordinate of source
v = 0.5 #air velocity (m/s) from left to right. 
R = 10 #aerosol emission rate (particles/s)
Q = 0 #0.002 # Air exchange rate (s^-1)
K = 0.0053 # Eddy diffusion coefficient (m^2/s)
d = 0 #1.7*10**(-4) #deactivation rate (s^-1)
s = 0 # 1.1*10**(-4) #settling rate (s^-1)
delta_x = 0.05 #(m) mesh-size


#set up mesh
n_x = int(l / delta_x) + 1 #int: calculate number of x-steps
n_y = int(w / delta_x) + 1 #int:calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
 #Initialise numpy array of same size as X for C (the concentration)

timelist = [time, time2]
reslist = []
deltaTlist = [delta_t, delta_t2]
vmax = 100
levels = np.linspace(0, vmax, n_x+1)


for tt in range(len(timelist)):
    C = np.zeros_like(X)
    term1temp = []
    term2temp = []
    term3temp = []
    #time-axis
    t_end = timelist[tt]
    n_t = int(t_end/deltaTlist[tt])
    t_arr = np.linspace(deltaTlist[tt],t_end,n_t) 
    S = delta_t * np.full(len(t_arr), R)
    m = int(v/(2*l) *time) 
    print(m)
    for t in range(1,len(t_arr)+1):
        # print(round(t_arr[t-1],3))
        t1 = np.zeros_like(X)
        t2 = np.zeros_like(Y)
        for i in range(len(x)):
            for j in range(len(y)):
                t1[i][j] = np.exp(-((X[i][j]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((X[i][j]+x_o+v*t)**2)/(4*K*t))
                t2[i][j] = np.exp(-((Y[i][j]-y_o)**2)/(4*K*t)) + np.exp(-((Y[i][j]+y_o)**2)/(4*K*t))
                for n in range(1,m+1):
                    t1[i][j] += np.exp(-((X[i][j]-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((X[i][j]+x_o+v*t - 2*n*l)**2)/(4*K*t))
                    t1[i][j] += np.exp(-((X[i][j]-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((X[i][j]+x_o+v*t + 2*n*l)**2)/(4*K*t))
                for n in range(1,5):
                    t2[i][j] += np.exp(-((Y[i][j]-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((Y[i][j]+y_o + 2*n*w)**2)/(4*K*t))
                    t2[i][j] += np.exp(-((Y[i][j]-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((Y[i][j]+y_o - 2*n*w)**2)/(4*K*t))
        
        t3 = np.exp(-(Q+d+s)*t)
        
        term1temp.append(t1)
        term2temp.append(t2)
        term3temp.append(t3)
            
    term1tempArr = np.dstack(term1temp)
    term2tempArr = np.dstack(term2temp)
        
    for i in range(len(x)):
        for j in range(len(y)):
            integ = 1/(4*np.pi*K*t_arr) * term1tempArr[0,i,:] * term2tempArr[j,0,:] * term3temp
            C[j][i] = convolve(S,integ,mode='valid') / (h/2)
                
    reslist.append(C)
    cc = np.unravel_index(C.argmax(), C.shape)
    ccc = np.subtract(np.asarray(t1.shape),1)/2
    print('ps', sum(sum(C)), '|', C[int(ccc[0]),int(ccc[1])], 
          'max value: ',cc, C[cc[0]][cc[1]])
    
    fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
    cp = ax.contourf(X, Y, C, levels=levels, vmin=0, vmax=vmax)
    plt.colorbar(cp)
    fig.tight_layout()
    plt.axis('square') 
    plt.show()
    
# a = np.subtract(reslist[1],reslist[0])
R = np.subtract(reslist[1], reslist[0])
RR =np.subtract(reslist[1], R)
rr = np.unravel_index(RR.argmax(), RR.shape)
print(sum(sum(RR)), RR[int(ccc[0]),int(ccc[1])], rr, RR[rr[0]][rr[1]])

fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
cp = ax.contourf(X, Y, RR, levels=levels, vmin=0, vmax=100)
plt.colorbar(cp)
fig.tight_layout()
plt.axis('square') 
plt.show()