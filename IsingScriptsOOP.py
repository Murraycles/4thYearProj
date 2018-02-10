# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:56:14 2017

@author: Benedict
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import IsingClasses as ic

#%%

ising = ic.IsingSim(10, 2, 1)

ising.iterate(10)

print(ising.ising.grid)
print(ising.noIt)