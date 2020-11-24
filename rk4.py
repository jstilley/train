#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:39:07 2020

@author: JohnStilley
"""

def rk4(ode, xspan, y0):
    # runge_kutta_4 utilizes the Ruge Kutta method to solve an ordinary differential equation
    # Inputs: ODE function call, domain over which function is solved, initial condition
    # Output: domain, y values for given domain
    
    import numpy as np
    
    x = xspan
    n = len(xspan) - 1
    
    y = np.zeros([len(xspan), len(y0)])
    
    for j in range(len(y0)):
        y[0,j] = y0[j]
        
    for i in range(n):
        h = x[i+1] - x[i]
    
        k1 = ode(x[i], y[i])
        k2 = ode(x[i] + 0.5*h, y[i] + 0.5*k1*h)
        k3 = ode(x[i] + 0.5*h, y[i] + 0.5*k2*h)
        k4 = ode(x[i] + h, y[i] + k3*h)
        
        y[i+1] = y[i] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
    return x, y