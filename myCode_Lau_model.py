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
time = 10 #event duration, in seconds
# time2 = 3
delta_t = 0.1 #(s) time-steps
# delta_t2 = 1 #(s) time-steps
# time2 = 5
l = 2 #x-length (m) of room
w = 2 #y-length (m) of room
h = 2 #z-length (m) of room
x_o = 1 #x-coordinate of source
y_o = 1 #y-coordinate of source
v = 1 #air velocity (m/s) from left to right. 
R = 10 #aerosol emission rate (particles/s)
Q = 0 #0.002 # Air exchange rate (s^-1)
K = 5e-3 #0.0053 # Eddy diffusion coefficient (m^2/s)
d = 0 #1.7*10**(-4) #deactivation rate (s^-1)
s = 0 # 1.1*10**(-4) #settling rate (s^-1)
delta_x = 0.025 #(m) mesh-size


#set up mesh
n_x = int(l / delta_x) + 1 #int: calculate number of x-steps
n_y = int(w / delta_x) + 1 #int:calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
 #Initialise numpy array of same size as X for C (the concentration)

# timelist = [time, time2]
reslist = []
# deltaTlist = [delta_t, delta_t2]
vmax = 300
levels = np.linspace(0, vmax, n_x+1)


# for tt in range(len(timelist)):
C = np.zeros_like(X)
term1temp = []
term2temp = []
term3temp = []
#time-axis
t_end = time
n_t = int(t_end/delta_t)
t_arr = np.linspace(delta_t,t_end,n_t)
# S = delta_t * np.full(len(t_arr), R)
m = 10 #int(v/(2*l) *time) 

for t in range(0,len(t_arr)):
    print(round(t_arr[t],3))
    t1 = np.zeros_like(X)
    t2 = np.zeros_like(Y)
    for i in range(len(x)):
        for j in range(len(y)):
            t1[i][j] = np.exp(-((X[i][j]-x_o-v*t_arr[t])**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t])**2)/(4*K*t_arr[t]))
            t2[i][j] = np.exp(-((Y[i][j]-y_o)**2)/(4*K*t_arr[t])) + np.exp(-((Y[i][j]+y_o)**2)/(4*K*t_arr[t]))
            for n in range(1,m+1):
                t1[i][j] += np.exp(-((X[i][j]-x_o -v*t_arr[t] + 2*n*l)**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t] - 2*n*l)**2)/(4*K*t_arr[t]))
                t1[i][j] += np.exp(-((X[i][j]-x_o -v*t_arr[t] - 2*n*l)**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t] + 2*n*l)**2)/(4*K*t_arr[t]))
            for n in range(1,5):
                t2[i][j] += np.exp(-((Y[i][j]-y_o - 2*n*w)**2)/(4*K*t_arr[t])) + np.exp(-((Y[i][j]+y_o + 2*n*w)**2)/(4*K*t_arr[t]))
                t2[i][j] += np.exp(-((Y[i][j]-y_o + 2*n*w)**2)/(4*K*t_arr[t])) + np.exp(-((Y[i][j]+y_o - 2*n*w)**2)/(4*K*t_arr[t]))
    
    t3 = np.exp(-(Q+d+s)*t_arr[t])
    
    term1temp.append(t1)
    term2temp.append(t2)
    term3temp.append(t3)
    
    term1tempArr = np.dstack(term1temp)
    term2tempArr = np.dstack(term2temp)
    
    term1tempArr[np.isnan(term1tempArr)] = 0
    term2tempArr[np.isnan(term2tempArr)] = 0
    if t_arr[t] > 0.4:
    #     # t1 = np.zeros_like(X)
    #     # t2 = np.zeros_like(Y)
    #     t1 = 
    #     t2 = 
        S = np.full(np.shape(term1temp)[0], 1e-10)#R)
    else:
        S = np.full(np.shape(term1temp)[0], R)
    C = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            integ = 1/(4*np.pi*K*t_arr[t]) * term1tempArr[0,i,:] * term2tempArr[j,0,:] * term3temp[t]
            C[j][i] = convolve(S,integ,mode='valid') / (h/2)
                    
    # reslist.append(C)
    # if t_arr[t] > 0.4:
        # newres = np.subtract(reslist[-1], np.subtract(reslist[3],reslist[4]))
    # if t > 4:
    #     newres = 
    
    cc = np.unravel_index(C.argmax(), C.shape)
    ccc = np.subtract(np.asarray(t1.shape),1)/2
    print('ps', sum(sum(C)), '|', C[int(ccc[0]), int(ccc[1])],'|', 
          'max value: ',cc, C[cc[0]][cc[1]])
    
    # A = reslist[16] + np.subtract(reslist[15], reslist[16])
    fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
    cp = ax.contourf(X, Y, C, levels=levels, vmin=0, vmax=vmax)
    plt.colorbar(cp)
    fig.tight_layout()
    plt.axis('square') 
    plt.show()
    
# a = np.subtract(reslist[1],reslist[0])
# R = np.subtract(reslist[1], reslist[0])
# RR =np.subtract(reslist[1], R)
# rr = np.unravel_index(RR.argmax(), RR.shape)
# print(sum(sum(RR)), RR[int(ccc[0]),int(ccc[1])], rr, RR[rr[0]][rr[1]])

# fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
# cp = ax.contourf(X, Y, RR, levels=levels, vmin=0, vmax=100)
# plt.colorbar(cp)
# fig.tight_layout()
# plt.axis('square') 
# plt.show()