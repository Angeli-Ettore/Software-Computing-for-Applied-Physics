# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt



#Tight Binding model energy function for nearest neighbors 1D case 
def TB_1D_nn(k,t1,a):
    return -2*t1*np.cos(k*a)

#Tight Binding model energy function for next-nearest neighbors 1D case 
def TB_1D_nnn(k,t1,t2,a):
    return TB_1D_nn(k,t1,a)-2*t2*np.cos(k*a)

#Tight Binding model energy function for nearest neighbors 2D case 
def TB_2D_nn(kx,ky,t1,a):
    return -2*t1*(np.cos(kx*a)+2*np.cos(kx*a/2)*np.cos(ky*a*np.sqrt(3)/2))

#Tight Binding model energy function for next-nearest neighbors 2D case 
def TB_2D_nnn(kx,ky,t1,t2,a):
    return TB_2D_nn(kx,ky,t1,a) - 2*t2*(np.cos(ky*a*np.sqrt(3))+2*np.cos(ky*a*np.sqrt(3)/2)*np.cos(ky*a*3/2))


#calculation of the DOS (2D NN case)
def DOS_1D_nn(t, a, k, eta, bounds=(-10,10)):
    #energy values
    energies = TB_1D_nn(k, t, a)
    
    #energy values for which the DOS is calculated
    rng = np.linspace(bounds[0], bounds[1], len(k))
    dos = np.zeros(len(k))
    
    for i, E in enumerate(rng):
        gaussian_weights = np.exp(-((E - energies) ** 2) / (eta ** 2)) / (np.sqrt(np.pi) * eta)
        dos[i] = np.sum(gaussian_weights) / (len(k) ** 2)
    
    return rng, dos


#calculation of the DOS (2D NN case)
def DOS_1D_nnn(t, tn, a, k, eta, bounds=(-10,10)):
    #energy values
    energies = TB_1D_nnn(k, t, tn, a)
    
    #energy values for which the DOS is calculated
    rng = np.linspace(bounds[0], bounds[1], len(k))
    dos = np.zeros(len(k))
    
    for i, E in enumerate(rng):
        gaussian_weights = np.exp(-((E - energies) ** 2) / (eta ** 2)) / (np.sqrt(np.pi) * eta)
        dos[i] = np.sum(gaussian_weights) / (len(k) ** 2)
    
    return rng, dos



#calculation of the DOS (2D NN case)
def DOS_2D_nn(t, a, kx, ky, eta, bounds=(-10,10)):
    #energy values for 2D NN case
    energies = TB_2D_nn(kx, ky, t, a).flatten()
    
    #energy values for which the DOS is calculated
    rng = np.linspace(bounds[0], bounds[1], len(kx))
    dos = np.zeros(len(kx))
    
    for i, E in enumerate(rng):
        gaussian_weights = np.exp(-((E - energies) ** 2) / (eta ** 2)) / (np.sqrt(np.pi) * eta)
        dos[i] = np.sum(gaussian_weights) / (len(kx) ** 2)
    
    return rng, dos

#calculation of the DOS (2D NNN case)
def DOS_2D_nnn(t, tn, a, kx, ky, eta, bounds=(-10,10)):
    #energy values for 2D NNN case
    energies = TB_2D_nnn(kx, ky, t, tn, a).flatten()
    
    #energy values for which the DOS is calculated
    rng = np.linspace(bounds[0], bounds[1], len(kx))
    dos = np.zeros(len(kx))
    
    for i, E in enumerate(rng):
        gaussian_weights = np.exp(-((E - energies) ** 2) / (eta ** 2)) / (np.sqrt(np.pi) * eta)
        dos[i] = np.sum(gaussian_weights) / (len(kx) ** 2)
    
    return rng, dos

def color_map(kx, ky, energies, case):
    plt.contourf(ky, kx, energies, cmap="viridis") #2D nearest neighbor case
    plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
    plt.xlabel("Wave Vector $k_x$")
    plt.ylabel("Wave Vector $k_y$")
    plt.title(f"Energy Band for 2D {case}")
    plt.show()
    return 

def dos_plotter(x, y, label):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=label)
    plt.xlabel("Energy")
    plt.ylabel("Density of States")
    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.show()



