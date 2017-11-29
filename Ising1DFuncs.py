# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:02:00 2017

@author: Benedict

Version 3
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

def isingGrid(n):
    '''Returns a 1D array of ones to simulate 1D ising model'''
    return np.ones((n,1))

#%%
    
def pointEn(isingGrid, coord, J):
    '''Finds the energy contributed to the system by one point on the array'''
    
    #length of the array
    n = len(isingGrid)
    
    #coordinate changes to find nearest neighbour points
    coordChange = np.array([1, -1])
    #apply coord changes
    neighbourPoints = coord + coordChange
    
    #impose periodic boundary condiitons
    for i in range(2):
        if neighbourPoints[i] > n - 1:
            neighbourPoints[i] = 0
            
    enPoint = 0
    #iterate over the grid using energy equation
    for i in range(2):
        enPoint += -J * isingGrid[coord, 0] * isingGrid[neighbourPoints[i], 0]
    
    return enPoint



#%%
    
def configEn(isingGrid, J):
    '''Iterates over 1D array and returns total configuration energy'''
    
    n = len(isingGrid)
    
    en = 0
    
    #use for loop to run across line
    for i in range(n):
        en += pointEn(isingGrid, i, J)
    
    #divide by 2 in the result to compensate for double counting
    return en/2

#%%
    
def enDelta(isingGrid, coord, J):
    '''Takes a grid, point and coupling const and finds energy change if
    a point flips'''
    
    #difference in energy when a point flips
    delta = -2 * pointEn(isingGrid, coord, J)
    
    return delta

#%%
    
def toFlip(delta, T):
    '''Takes in the change in energy due to a flip and decides whether to flip
    the point. Uses boltzmann distribution to decide if positive changes
    should flip. Returns True or False.'''
    #condition for loss of energy under flip
    if delta <= 0:
        return True
    #condition for gain in energy
    else:
        i = np.random.uniform()
        #decide whether to flip based on boltzmann distribution
        if i < np.exp(-delta/T):
            return True
        else:
            return False

#%%

def metropolisAlg(isingGrid, J, T, mcSteps):
    '''Simulates the ising model using metropolis-hastings algorithm'''
    
    #Find number of iterations from mc steps
    n = len(isingGrid)
    noIt = mcSteps * n
    
    #find energy of initial state
    energy = np.array(configEn(isingGrid, J))
    enArray = np.array(energy)
    
    #find magnetisation
    mag = np.sum(isingGrid)
    magArray = np.array(mag)
    
    plotArray = np.copy(isingGrid)
    figLength = 100
    gridims = []
    
    
    #iterate over array
    for i in np.arange(noIt):
        #generate random coordinate on the grid
        x = np.random.randint(n)
        #find energy change for fliping x
        delta = enDelta(isingGrid, x, J)
        #decide whether to flip the point
        yn = toFlip(delta, T)
        
        #apply result of toFlip
        if yn:
            isingGrid[x] = -isingGrid[x]
            energy += delta
        
        mag = np.sum(isingGrid)
        magArray = np.hstack((magArray, mag))
        
        enArray = np.hstack((enArray, energy))
            
        plotArray = np.hstack((plotArray, isingGrid))
        
        if len(plotArray) < figLength:
            gridim = plt.imshow(plotArray, animated=True)
        else:
            gridim = plt.imshow(plotArray[:,-figLength:], animated=True)
            
            
        gridims.append([gridim])
    
    #print(enArray)    
    return plotArray, gridims, enArray, magArray
    
#%%
