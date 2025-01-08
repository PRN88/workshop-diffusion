#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# Here we develop a one dimensional model of diffusion.
# It assumes a constant diffusivity.
# It uses regular grid.
# It has fixed boundary conditions.

#  The diffusion equation:
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# The discretized version of the diffusion equation that we'll solve with our model:
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

import numpy as np
import matplotlib.pyplot as plt



D = 100 #diffusivity
Lx = 300 #domain size



dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx= len(x)

C=np.zeros_like(x)
C_left=500
C_right=0
C[x<=Lx//2]=C_left
C[x>Lx//2]=C_right


plt.figure()
plt.plot(x,C,"r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("initial concentration profile")


time=0
nt=5000
dt=0.5*(dx**2/D)

for t in range(0,nt):
    C+=D*dt/dx**2*(np.roll(C,-1)-2*C+np.roll(C,1))
    C[0]=C_left
    C[-1]=C_right


plt.figure()
plt.plot(x,C,"b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")

