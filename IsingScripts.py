# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:48:58 2017

@author: Benedict

Version 3
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import Ising1DFuncs as ising


#%%

L = 100
J = 1
T = 2
mcSteps = 10

isingArray = ising.isingGrid(L)

plotArray, gridims, enArray, magArray = ising.metropolisAlg(isingArray, J, T, mcSteps)

#show animation
fig = plt.figure(1)
ani = animation.ArtistAnimation(fig, gridims, interval=5, blit=True, repeat_delay=1000)
plt.show()

fig2 = plt.figure(2)
plt.plot(np.arange(len(enArray)), enArray)

fig3 = plt.figure(3)
plt.plot(np.arange(len(magArray)), magArray)