#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:15:06 2020

@author: JohnStilley
"""
import numpy as np

def train_motion(t, y, par):
    # train_motion creates a vector of functions that describe
    # the motion of a metaphysical train
    # Inputs: time, position and velocity of the train, function parameter dictionary
    # Outputs: dydt(vector), evaluated functions at given values
    
    # Assigns initial variables
    x = y[0]
    v = y[1]
    
    # Solves for torque
    Ap = np.pi * (par['pR'])**2
    Fp = Ap * par['kPa']
    T = Fp * par['gR']

    # Solves for the forces in acceleration/deceleration equations
    F = T/par['wR']
    Fd = 0.5 * par['Cd'] * par['p'] * par['A'] * (v**2)
    Fr = par['m'] * par['g'] * par['Crr']
    
    # Solves for the acceleration distance
    La = (par['Ls'] * par['wR'])/par['gR']
    
    dxdt = v
    # Determines which equation to use for acceleration
    # depending on the distance that the train has traveled
    if x <= La:
        dvdt = (F - Fd - Fr)/(par['m'] + par['mW'])
        
    elif x > La: 
        dvdt = ((-1*Fd) - Fr)
    
    Ft = F - par['mW'] * dvdt
    
    if Ft > (par['mu'] * (par['m']/2) * par['g']):
        raise ValueError('Torque Force exceed wheel slip criterion')
             
    return np.array([dxdt, dvdt])


