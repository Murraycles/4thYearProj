# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:25:34 2017

@author: Benedict
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

class Ising1D(object):
    """A 1D grid of points of value 1 or -1, to simulate the ising model"""
    
    def __init__(self, size, J):
        self.grid = np.ones((size, 1))
        self.coupling = J
        self.size = size
    
    def pointEn(self, point):
        
        coordChange = np.array([1, -1])
        neighbourPoints = point + coordChange
        
        for i, P in enumerate(neighbourPoints):
            if P > self.size - 1:
                neighbourPoints[i] = 0
        
        enPoint = 0
        
        for i, _ in enumerate(neighbourPoints):
            enPoint += -self.coupling * self.grid[i] * self.grid[point]
        
        return enPoint
    
    def configEn(self):
        
        en = 0
        
        for i in range(self.size):
            en += self.pointEn(i)
            
        return en/2
    
    def enDelta(self, point):
        
        delta = -2 * self.pointEn(point)
        
        return delta
    
    def toFlip(self, point, T):
        
        delta = self.enDelta(point)
        
        if delta <= 0:
            return True
        else:
            i = np.random.uniform()
            
            if i < np.exp(-delta / T):
                return True
            else:
                return False
    
    def configMag(self):
        return np.sum(self.grid)
    
            
#%%

class IsingSim(object):
    
    def __init__(self, size, T, J):
        
        self.ising = Ising1D(size, J)
        self.size = size
        self.temp = T
        self.coupling = J
        
        self.noIt = np.empty((1, 2))
        
        self.energy = self.ising.configEn()
        self.mag = self.ising.configMag()
    
    def iterate(self, MCsteps):
        iters = MCsteps * self.size
        
        for i in np.arange(iters):
            
            x = np.random.randint(self.size)
            
            yn = self.ising.toFlip(x, self.temp)
            
            if yn:
                self.energy += self.ising.enDelta(x)
                self.ising.grid[x] = -self.ising.grid[x]
                self.mag = self.ising.configMag()
        
        self.noIt += np.array([MCsteps, iters])
        