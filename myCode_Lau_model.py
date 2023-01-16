#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:48:46 2023

@author: akimlavrinenko
"""

import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library
import scipy.fftpack

#Parameters values -- User inputs floats
time = 1000 #event duration, in seconds
t_inj = 0.4 #inhection duration, in seconds
delta_t = 0.1 #(s) time-steps

delta_x = 0.05  #(m) mesh-size

l = 3.14 #x-length (m) of room
w = 3.14 #y-length (m) of room
h = 3.14 #z-length (m) of room

x_o = 1.57 #x-coordinate of source
y_o = 1.57 #y-coordinate of source

v = 0.2 #air velocity (m/s) from left to right. 
R = 100 #aerosol emission rate (particles/s)
Q = 0 #0.002 # Air exchange rate (s^-1)
K = 5e-3 #0.0053 # Eddy diffusion coefficient (m^2/s)
d = 0 #1.7*10**(-4) #deactivation rate (s^-1)
s = 0 # 1.1*10**(-4) #settling rate (s^-1)

#set up mesh
n_x = int(l / delta_x) + 1 #int: calculate number of x-steps
n_y = int(w / delta_x) + 1 #int:calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
 #Initialise numpy array of same size as X for C (the concentration)

reslist = []
vmax = 20
levels = np.linspace(0, vmax, n_x+1)

term1temp = []
term2temp = []
term3temp = []
t_end = time
n_t = int(t_end/delta_t)
t_arr = np.linspace(delta_t,t_end,n_t)
m = 20 #int(v/(2*l) *time) 

for t in range(0,len(t_arr)):
    print('_____________________________________________________________')
    print(round(t_arr[t],3))
    t1 = np.zeros_like(X)
    t2 = np.zeros_like(Y)
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
    # S = np.full(np.shape(term1temp)[0], R)
    C = np.zeros_like(X)
    integlist = []
    
    for i in range(len(x)):
        for j in range(len(y)):
            integ = 1/(4*np.pi*K*t_arr[:t+1]) * term1tempArr[0,i,:] * term2tempArr[j,0,:] * term3temp[:t+1]
            C[j][i] = convolve(S,integ,mode='valid') / (h/2)
            
            # r = scipy.fftpack.fft(S).real * scipy.fftpack.fft(integ).real
            # rr = scipy.fftpack.ifft(r).real
            # print(rr)
            # C[j][i] = rr[0]
            # c= 0
            # for ii in range(len(newS)):
                # c += integ[ii]*delta_t
            # CC[j][i] = c
            
    reslist.append(C)
    cc = np.unravel_index(C.argmax(), C.shape)
    ccc = np.subtract(np.asarray(t1.shape),1)/2
    
    
    if t_arr[t] > t_inj:
        itemindex = np.where(t_arr == t_inj+delta_t)
        CC = np.subtract(reslist[t],reslist[t-itemindex[0][0]])
        cc_max = np.unravel_index(CC.argmax(), CC.shape)
        
        fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
        tt = np.round(t_arr[t],3)
        ax.set_title('V: ' + str(v) + ', time: ' +  str(tt) + ' s')
        cp = ax.contourf(X, Y, CC, levels=levels, vmin=0, vmax=vmax)
        plt.colorbar(cp)
        plt.axis('square')
        plt.ylabel('y, m')
        plt.xlabel('x, m')
        fig.tight_layout()
        plt.savefig('./conc_' + str(tt) + '.png', dpi = 100)
        plt.show()
        
        print('ps', sum(sum(CC)),'|', 'max value: ',cc_max, CC[cc_max[0]][cc_max[1]])
        diff0 = abs(CC[cc_max[0]][cc_max[1]] - CC[0,0])
        diff1 = abs(CC[cc_max[0]][cc_max[1]] - CC[0,np.asarray(t1.shape)[0]-1])
        diff2 = abs(CC[cc_max[0]][cc_max[1]] - CC[np.asarray(t1.shape)[0]-1,0])
        diff3 = abs(CC[cc_max[0]][cc_max[1]] - CC[np.asarray(t1.shape)[0]-1, np.asarray(t1.shape)[0]-1])
        if diff0 and diff1 and diff2 and diff3 < 1e-2:
            print('Converged!')
            print(diff0, diff1, diff2, diff3)
            break
    else:
        fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
        tt = np.round(t_arr[t],3)
        ax.set_title('V: ' + str(v) + ', time: ' +  str(tt) + ' s')
        cp = ax.contourf(X, Y, C, levels=levels, vmin=0, vmax=vmax)
        plt.colorbar(cp)
        plt.axis('square')
        plt.ylabel('y, m')
        plt.xlabel('x, m')
        fig.tight_layout()
        plt.savefig('./conc_' + str(tt) + '.png', dpi = 100)
        plt.show()
        print('ps', sum(sum(C)), '|', C[int(ccc[0]), int(ccc[1])],'|', 
              'max value: ',cc, C[cc[0]][cc[1]])