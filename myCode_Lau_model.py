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
time = 5#event duration, in seconds
t_inj = 0.4 #inhection duration, in seconds
delta_t = 0.1 #(s) time-steps

delta_x = 0.05 #(m) mesh-size

l = 3.14 #x-length (m) of room
w = 3.14 #y-length (m) of room
h = 3.14 #z-length (m) of room

x_o = 1 #x-coordinate of source
y_o = 1.57 #y-coordinate of source

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


sigmalist = []
klist = [1e-1, 5e-3]
colorlist = ['r','b','k','m','c','y']

for K in klist:
    sigmaTsigma0list = []
    reslist = []
    t1sumsum = []
    term1temp = []
    term2temp = []
    term3temp = []

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
        # c0max = reslist[0].argmax()
        
        if t_arr[t] > t_inj:
            sigma = np.zeros_like(X)
            itemindex = np.where(t_arr == t_inj+delta_t)
            C = np.subtract(reslist[t],reslist[t-itemindex[0][0]])
            C[C<0] = 0
            cc_max = np.unravel_index(C.argmax(), C.shape)
            
            Cinf = sum(sum(C))*delta_x**2 / (l*w)
            for i in range(len(x)):
                for j in range(len(y)):
                    sigma[j][i] = (C[j][i] - Cinf)**2
            
            sigma = np.sqrt(sum(sum(sigma)) / (n_x*n_y))
            print('sigma: ', sigma)
            sigmalist.append(sigma)
            
            # fig,ax = plt.subplots(1,1,  figsize=(5.5, 5))
            # tt = str(int(round(t_arr[t],3)*10)).zfill(5)
            # ax.set_title('V: ' + str(v) + ', time: ' +  str(round(t_arr[t],3)) + ' s')
            # cp = ax.contourf(X, Y, C, levels=levels, vmin=0, vmax=vmax)
            # plt.colorbar(cp)
            # plt.axis('square')
            # plt.ylabel('y, m')
            # plt.xlabel('x, m')
            # fig.tight_layout()
            # plt.savefig('./conc_' + tt + '.png', dpi = 100)
            # plt.show()
            
            print('ps', sum(sum(C))/(l*w),'|', 'max value: ',cc_max, C[cc_max[0]][cc_max[1]])
            diff0 = abs(C[cc_max[0]][cc_max[1]] - C[0,0])
            diff1 = abs(C[cc_max[0]][cc_max[1]] - C[0,np.asarray(t1.shape)[0]-1])
            diff2 = abs(C[cc_max[0]][cc_max[1]] - C[np.asarray(t1.shape)[0]-1,0])
            diff3 = abs(C[cc_max[0]][cc_max[1]] - C[np.asarray(t1.shape)[0]-1, np.asarray(t1.shape)[0]-1])
          
            # c0max = np.unravel_index(reslist[3].argmax(), reslist[3].shape)
            # print(c0max)
            
            # if diff0 and diff1 and diff2 and diff3 < 1e-4:
            #     print('Converged!')
            #     print(diff0, diff1, diff2, diff3)
            #     break
        # else:
            # fig,ax = plt.subplots(1,1, figsize=(5.5, 5))
            # tt = str(int(round(t_arr[t],3)*10)).zfill(5)
            # ax.set_title('V: ' + str(v) + ', time: ' +  str(round(t_arr[t],3)) + ' s')
            # cp = ax.contourf(X, Y, C, levels=levels, vmin=0, vmax=vmax)
            # plt.colorbar(cp)
            # plt.axis('square')
            # plt.ylabel('y, m')
            # plt.xlabel('x, m')
            # fig.tight_layout()
            # plt.savefig('./conc_' + tt + '.png', dpi = 100)
            # plt.show()
            # print('ps', sum(sum(C))/(l*w), '|', C[int(ccc[0]), int(ccc[1])],'|', 
                  # 'max value: ',cc, C[cc[0]][cc[1]])
        
        # c0max = np.unravel_index(reslist[0].argmax(), reslist[0].shape)
        # cc_max = np.unravel_index(C.argmax(), C.shape)
        # ctmax = C[cc_max[0]][cc_max[1]]
        # val = C[int(ccc[0]),:]
        # vallist.append(val)
        # print('val', val)
        if K == klist[0]:
            linestyle = 'o'
            label = 'K = ' + str(klist[0])
            color = 'k'
        elif K == klist[1]:
            linestyle = 's'
            label = 'K = ' + str(klist[1])
            color = 'r'
        elif K == klist[2]:
            linestyle = '*'
            label = 'K = ' + str(klist[2])
            color = 'b'
        elif K == klist[3]:
            linestyle = 'v'
            label = 'K = ' + str(klist[3])
            color = 'c'
        elif K == klist[4]:
            linestyle = "^"
            label = 'K = ' + str(klist[4])
            color = 'y'
        
        # sss = sum(sum(C))*delta_x**2
        # if t == 0:
            # plt.plot(t_arr[t], sss, marker = linestyle, label = label, color = color, markersize = 1)
        # else:
            # plt.plot(t_arr[t], sss, marker = linestyle, color = color, markersize = 1)
        if t_arr[t] == t_inj+delta_t:
            sss = sigmalist[t-itemindex[0][0]]/sigmalist[0]
            sigmaTsigma0list.append(sss)
            plt.plot(t_arr[t], sss, marker = linestyle, color = color, markersize = 1, label = label)
        elif t_arr[t] > t_inj+delta_t:
            sss = sigmalist[t-itemindex[0][0]]/sigmalist[0]
            sigmaTsigma0list.append(sss)
            plt.plot(t_arr[t], sss, marker = linestyle, color = color, markersize = 1)
        end = timer.time()
        print('loop time ', round(end - start, 3), 's')
        
    res = np.asarray([t_arr[4:],sigmaTsigma0list]).T
    np.savetxt('./sigma_' + str(K) + '.dat', res)
        
plt.ylabel('sigma[t]/sigma[t=0.4]')
plt.xlabel('t, s')
plt.legend(loc="upper right")
plt.ylim(0,1)
plt.savefig('./sigma2.png', dpi = 200)
plt.show()
        
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# newlist = chunkIt(vallist, len(klist))

# for k in range(len(klist)):
#     plt.plot(t_arr[:len(newlist[k])], newlist[k], 'r-o', markersize = 0.5, label = 'K = ' + str(klist[k]), color = colorlist[k])
# plt.legend(loc="upper right")
# plt.ylabel('C_max/C0_max')
# plt.xlabel('t, s')
# plt.savefig('./k_cmax_c0max.png')
# plt.show()

