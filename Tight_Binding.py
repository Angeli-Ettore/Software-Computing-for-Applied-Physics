# -*- coding: utf-8 -*-

# Band structure and Density Of States calculation for a triangular lattice
# in different cases of dimensionality and neighbor interaction

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

# parameters
N = 200 #number of lattice sites
t1 = 1.0 # nearest neighbor hopping parameter
t2 = 0.125 * t1 # next-nearest neighbor hopping parameter
a = 1.0 #lattice constant

#initialization of the wave vector for both the 1D and 2D cases
k_vec = np.linspace(-pi/a, pi/a, N) # wave vector k
kx_vec = np.linspace(-pi/a, pi/a, N) # x component of the wave vector k
ky_vec = np.linspace(-pi/a, pi/a, N) # y component of the wave vector k


#Tight Binding model energy function for nearest neighbors 1D case 
def TB_near_1D(k,t1,a):
    return -2*t1*np.cos(k*a)

#Tight Binding model energy function for next-nearest neighbors 1D case 
def TB_next_1D(k,t1,t2,a):
    return TB_near_1D(k,t1,a)-2*t2*np.cos(k*a)

#Tight Binding model energy function for nearest neighbors 2D case 
def TB_near_2D(kx,ky,t1,a):
    return -2*t1*(np.cos(kx*a)+2*np.cos(kx*a/2)*np.cos(ky*a*sqrt(3)/2))

#Tight Binding model energy function for next-nearest neighbors 2D case 
def TB_next_2D(kx,ky,t1,t2,a):
    return TB_near_2D(kx,ky,t1,a) - 2*t2*(np.cos(ky*a*sqrt(3))+2*np.cos(ky*a*sqrt(3)/2)*np.cos(ky*a*3/2))

#energy band for the 1D nearest neighbors case
energies_1D_near = TB_near_1D(k_vec, t1, a)

#energy band for the 1D next-nearest neighbors case
energies_1D_next = TB_next_1D(k_vec, t1, t2, a)

# plots of the energy bands for the 1D case
fig, (near1D, next1D) = plt.subplots(1, 2, figsize=(10, 4))

#nearest neighbors 1D case plot
near1D.plot(k_vec, energies_1D_near, label=r'nearest neighbors 1D case', color='b')
near1D.set_xlabel(r'wave vector k')
near1D.set_ylabel(r'energy $\epsilon$(k)')
near1D.legend()

#next-nearest neighbors 1D case plot
next1D.plot(k_vec, energies_1D_next, label=r'next-nearest neighbors 1D case', color='b')
next1D.set_xlabel(r'wave vector k')
next1D.set_ylabel(r'energy $\epsilon$(k)')
next1D.legend()

plt.tight_layout()
plt.show()


#energy band for the 2D nearest neighbors case
energies_2D_near = np.zeros((N,N))
for i, kx in enumerate(kx_vec):
    for j, ky in enumerate(ky_vec):
        energies_2D_near[i,j] = TB_near_2D(kx,ky,t1,a)

#energy band for the 2D next-nearest neighbors case
energies_2D_next = np.zeros((N,N))
for i, kx in enumerate(kx_vec):
    for j, ky in enumerate(ky_vec):
        energies_2D_next[i,j] = TB_next_2D(kx,ky,t1,t2,a)

'''
#general check
print(energies_2D_near)
print(energies_2D_next)
'''

x1, y1 = np.meshgrid(kx_vec, ky_vec)
plt.contourf(x1, y1, energies_2D_near, cmap="viridis")
plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
plt.xlabel("Wave vector $k_x$")
plt.ylabel("Wave vector $k_y$")
plt.title("Energy Band Structure for NN 2D")
plt.show()

x2, y2 = np.meshgrid(kx_vec, ky_vec)
plt.contourf(x2, y2, energies_2D_next, cmap="viridis")
plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
plt.xlabel("Wave vector $k_x$")
plt.ylabel("Wave vector $k_y$")
plt.title("Energy Band Structure for NNN 2D")
plt.show()
