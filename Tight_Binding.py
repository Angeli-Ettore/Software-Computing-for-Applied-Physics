# Band structure and Density Of States calculation for a triangular lattice in different cases of dimensionality and neighbor interaction

import numpy as np
import plots as plot
import calculations as calc
from time import time

''' --------definition of working parameters-------- '''
t_nn = 1.0 # nearest neighbor hopping parameter
t_nnn = 0.2 * t_nn # next-nearest neighbor hopping parameter
a = 1.0 #lattice constant
N = 1000 #number of lattice sites
width = 0.01 # parameter of the gaussian\lorentzian that approximate delta function in DOS calculation
method = "lorentzian" # method for approximating Dirac's delta in DOS calculation (use only gaussian or lorentzian)
tempo = time() # used time for the entire code
''' ------------------------------------------------ '''


''' --------crafting and checking of params-------- '''
params = [0, t_nn, t_nnn, a, N, width, method] 
calc.params_check(params)
''' ----------------------------------------------- '''


''' --------definition of wave vectors (1D & 2D)-------- '''
bound_1D =  np.pi / params[3] #dimension of the 1D bound (linear case)
bound_2D = (4 * np.pi) / params[3] #dimension of the 2D bound (hexagonal case)
tr_to_sq_factor = 2/np.sqrt(3) #normalization factor needed for triangular FBZ for plotting

k_vec = np.linspace(-bound_1D, bound_1D , params[4]) # wave vector (1D case)

kx_vec = np.linspace(-bound_2D, bound_2D, params[4]) # wave vector x component (2D case)
ky_vec = np.linspace(-bound_2D * tr_to_sq_factor, bound_2D * tr_to_sq_factor, params[4]) # wave vector y component (2D case)

hexagon = calc.hexagonal_contour(kx_vec, ky_vec, bound_2D)
kx_grid, ky_grid = np.meshgrid(kx_vec[hexagon], ky_vec[hexagon]) # mesh grid of the wave vector (2D case)
''' ---------------------------------------------------- '''

''' --------calculation and plotting of energy bands and DOS-------- '''
params[0] = 1 #1D NN case
energy_values = calc.TB_1D(params, k_vec)
dos_range, dos_values = calc.DOS_1D(params, energy_values)

plot.Energy_and_DOS_1D_plotter(params, k_vec, energy_values, dos_range, dos_values)


params[0] = 2 #1D NNN case
energy_values = calc.TB_1D(params, k_vec)
dos_range, dos_values = calc.DOS_1D(params, energy_values)

plot.Energy_and_DOS_1D_plotter(params, k_vec, energy_values, dos_range, dos_values)


params[0] = 3 #2D NN case
energy_values = calc.TB_2D(params, kx_grid, ky_grid)
dos_range, dos_values = calc.DOS_2D(params, energy_values)

plot.Energy_and_DOS_2D_plotter(params, kx_grid, ky_grid, energy_values, dos_range, dos_values)
plot.color_map_plotter(params, kx_grid, ky_grid, energy_values)


params[0] = 4 #2D NNN case
energy_values = calc.TB_2D(params, kx_grid, ky_grid)
dos_range, dos_values = calc.DOS_2D(params, energy_values)

plot.Energy_and_DOS_2D_plotter(params, kx_grid, ky_grid, energy_values, dos_range, dos_values)
plot.color_map_plotter(params, kx_grid, ky_grid, energy_values)
''' ---------------------------------------------------------------- '''

print(f"Code executed successfully in {time() - tempo:.2f} seconds!")