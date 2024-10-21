# Band structure and Density Of States calculation for a triangular lattice in different cases of dimensionality and neighbor interaction

import numpy as np
import plots as plot
from time import time

'''
Definition of working parameters
'''
N = 500 #number of lattice sites
t_nn = 1.0 # nearest neighbor hopping parameter
t_nnn = 0.1 * t_nn # next-nearest neighbor hopping parameter
a = 5.0 #lattice constant
width = 0.01 # parameter of the gaussian\lorentzian that approximate delta function in DOS calculation
bounds = [-12,12] #energy values for DOS calculation and plotting
method = "lorentzian" # method for approximating Dirac's delta in DOS calculation (use only gaussian or lorentzian)
tempo = time() # used time for the entire code


'''
definition of working parameter list "params"
'''
params = [0, t_nn, t_nnn, a, N, width, *bounds, method] 


'''
Initialization of the wave vectors (1D & 2D)
'''
k_vec = np.linspace(-np.pi / a, np.pi / a , N) # wave vector k (1D case)
kx_vec = np.linspace(-4 * np.pi / (3 * a), 4 * np.pi / (3 * a), N) # x component of the wave vector k (2D case)
ky_vec = np.linspace(-4 * np.pi / (3 * a), 4 * np.pi / (3 * a), N) # y component of the wave vector k (2D case)
kx_grid, ky_grid = np.meshgrid(kx_vec, ky_vec) # mesh grid for definition of the 2D energy array


params[0] = 1 #1D NN case
plot.Energy_and_DOS_1D_plotter(params, k_vec)

params[0] = 2 #1D NNN case
plot.Energy_and_DOS_1D_plotter(params, k_vec)

params[0] = 3 #2D NN case
plot.Energy_and_DOS_2D_plotter(params, kx_grid, ky_grid)
plot.color_map_plotter(params, kx_grid, ky_grid)

params[0] = 4 #2D NNN case
plot.Energy_and_DOS_2D_plotter(params, kx_grid, ky_grid)
plot.color_map_plotter(params, kx_grid, ky_grid)


print(f"Code executed successfully in {time() - tempo:.2f} seconds!")

