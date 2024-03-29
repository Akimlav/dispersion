#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:48:46 2023

@author: akimlavrinenko
"""

import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import fftconvolve #import convolve function from scipy.signal library
import time as timer

start = timer.time()
#Parameters values -- User inputs floats
time = 10 #event duration, in seconds
t_inj = 1 #inhection duration, in seconds
delta_t = 0.1 #(s) time-steps

delta_x = 0.125 #(m) mesh-size

l = 3.1 #x-length (m) of room
w = 3.1 #y-length (m) of room
h = 3.1 #z-length (m) of room

x_o = 0.6 #x-coordinate of source
y_o = 0.6 #y-coordinate of source

v = 0.0856 #air velocity (m/s) from left to right. 
R = 1370 #aerosol emission rate (particles/s)
Q = 0 #0.002 # Air exchange rate (s^-1)
# K = 5e-3 #0.0053 # Eddy diffusion coefficient (m^2/s)
d = 0 #1.7*10**(-4) #deactivation rate (s^-1)
s = 0 # 1.1*10**(-4) #settling rate (s^-1)

#set up mesh
n_x = int(l / delta_x) + 1 #int: calculate number of x-steps
n_y = int(w / delta_x) + 1 #int:calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
 #Initialise numpy array of same size as X for C (the concentration)


vmax = 1
levels = np.linspace(0, vmax, n_x+1)


t_end = time
n_t = int(t_end/delta_t)
t_arr = np.asarray(np.linspace(delta_t,t_end,n_t))



klist = [0.01, 0.02, 0.03]

for K in klist:
    sigmalist = []
    reslist = []
    t1sumsum = []
    term1temp = []
    term2temp = []
    term3temp = []
    Cinflist = []
    for t in range(0,len(t_arr)):
        print('_____________________________________________________________')
        print(round(t_arr[t],3))
        m = int(v/(2*l)*t) 
        t1 = np.zeros_like(X)
        t2 = np.zeros_like(Y)
        t1sum = []
        for i in range(len(x)):
            for j in range(len(y)):
                
                t1[i][j] = np.exp(-((X[i][j]-x_o-v*t_arr[t])**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t])**2)/(4*K*t_arr[t]))
                t2[i][j] = np.exp(-((Y[i][j]-y_o)**2)/(4*K*t_arr[t])) + np.exp(-((Y[i][j]+y_o)**2)/(4*K*t_arr[t]))
                for n in range(1,m+1):
                    t1[i][j] += np.exp(-((X[i][j]-x_o-v*t_arr[t] + 2*n*l)**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t] - 2*n*l)**2)/(4*K*t_arr[t]))
                    t1[i][j] += np.exp(-((X[i][j]-x_o-v*t_arr[t] - 2*n*l)**2)/(4*K*t_arr[t])) + np.exp(-((X[i][j]+x_o+v*t_arr[t] + 2*n*l)**2)/(4*K*t_arr[t]))
                for n in range(1,4):
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
    
        S = np.full(np.shape(term1temp)[0], R)*delta_t
        C = np.zeros_like(X)
        
        integlist = []
        
        for i in range(len(x)):
            for j in range(len(y)):
                integ = 1/(4*np.pi*K*t_arr[:t+1]) * term1tempArr[0,i,:] * term2tempArr[j,0,:] * term3temp[:t+1]
                C[j][i] = fftconvolve(S,integ,mode='valid') / (h/2) #* delta_x**2
                

        reslist.append(C)
        # cc = np.unravel_index(C.argmax(), C.shape)
        # ccc = np.subtract(np.asarray(t1.shape),1)/2
        
        if t_arr[t] > t_inj:
            sigma = np.zeros_like(X)
            itemindex = np.where(t_arr == t_inj+delta_t)
            C = np.subtract(reslist[t],reslist[t-itemindex[0][0]])
            C[C<1e-15] = 0
            cc_max = np.unravel_index(C.argmax(), C.shape)
            
            Cinf = sum(sum(C))*delta_x**2 / (l*w)
            for i in range(len(x)):
                for j in range(len(y)):
                    sigma[j][i] = (C[j][i] - Cinf)**2
            
            sigma = np.sqrt(sum(sum(sigma)) / (n_x*n_y))
            print('sigma: ', sigma)
            sigmalist.append(sigma)
            Cinflist.append(Cinf)
            
    new_t_arr = t_arr[np.where(t_arr == t_inj)[0][0] + 1:]
    res = np.asarray([new_t_arr,sigmalist, Cinflist]).T
    np.savetxt('./lau_sigma_D' + str(K) + '_dx' + str(delta_x) + '_dt' + str(delta_t) + '.dat', res)
        