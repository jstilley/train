#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:31:22 2020

@author: JohnStilley
"""

import numpy as np
from rk4 import rk4
from train_motion import train_motion
import matplotlib.pyplot as plt

# Parameters
Ls = 0.1 # Piston stroke length (m)
wR = 0.025 # Wheel radius (m)
gR = 0.01 # Gear radius (m)
pR = 0.01 # Piston radius (m)
mW = 0.1 # Wheel mass (kg)
kPa = 100000.0 # Tank Gauge pressure (Pa)
g = 9.8 # gravitational acceleration (m/s^2)
p = 1.0 # air density (kg/m^3)
m = 10.0 # train mass (kg)
A = 0.05 # frontal area (m^2)
mu = 0.7 # coefficient of static friction
Cd = 0.8 # Drag coefficient
Crr = 0.03 # Rolling resistance coefficient

params_dict = {'Ls': Ls, 'wR': wR, 'gR': gR, 'pR': pR,
               'mW': mW, 'kPa': kPa, 'g': g, 'p': p,
               'm': m, 'A': A, 'mu': mu, 'Cd': Cd, 'Crr': Crr}

# Creates ODE function call
ode = lambda t, y : train_motion(t, y, params_dict)

# Utilizes Runge Kutta to solve the ODEs
N = 0.04
tspan = np.arange(0.0, 0.95, N)
y0 = np.array([0.0,0.0])
tr, yr = rk4(ode, tspan, y0)


plt.figure()
plt.plot(tr, yr[:,0], 'r-')
plt.title('Position Over Time')
plt.xlabel('time (secs)')
plt.figure()
plt.plot(tr, yr[:,1], 'g-')
plt.title('Velocity Over Time')
plt.xlabel('time (secs)')



