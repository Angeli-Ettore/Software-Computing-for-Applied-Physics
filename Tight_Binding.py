# Band structure and Density Of States calculation for a triangular lattice in different cases of dimensionality and neighbor interaction

import numpy as np
import functions as func
import matplotlib.pyplot as plt



'''
Definition of working parameters
'''
N = 300 #number of lattice sites
t1 = 1.0 # nearest neighbor hopping parameter
t2 = 0.1 * t1 # next-nearest neighbor hopping parameter
a = 5.0 #lattice constant
eta = 0.01 # broading parameter of the gaussian that approximate delta function in DOS calculation
bound = (-8,4) #energy values for DOS calculation and plotting

'''
Initialization of the wave vectors (1D & 2D)
'''
k_vec = np.linspace(-np.pi / a, np.pi / a , N) # wave vector k (1D case)
kx_vec = np.linspace(-4 * np.pi / (3 * a), 4 * np.pi / (3 * a), N) # x component of the wave vector k (2D case)
ky_vec = np.linspace(-4 * np.pi / (3 * a), 4 * np.pi / (3 * a), N) # y component of the wave vector k (2D case)
kx_grid, ky_grid = np.meshgrid(kx_vec, ky_vec) # mesh grid for definition of the 2D energy array


'''
Initialization of the energy bands
'''
energies_1D_nn = func.TB_1D_nn(k_vec, t1, a) # 1D nearest neighbors case
energies_1D_nnn = func.TB_1D_nnn(k_vec, t1, t2, a) # 1D next-nearest neighbors case
energies_2D_nn = func.TB_2D_nn(kx_grid, ky_grid, t1, a) # 2D nearest neighbors case
energies_2D_nnn = func.TB_2D_nnn(kx_grid, ky_grid, t1, t2, a) # 2D next-nearest neighbors case



'''
Initialization of the density of states (and range for plotting)
'''
range_1D_nn, DOS_1D_nn_values = func.DOS_1D_nn(t1, a, k_vec, eta, bound) # 1D nearest neighbors case
range_1D_nnn, DOS_1D_nnn_values = func.DOS_1D_nnn(t1, t2, a, k_vec, eta, bound) # 1D next-nearest neighbors case
range_2D_nn, DOS_2D_nn_values = func.DOS_2D_nn(t1, a, kx_grid, ky_grid, eta, bound) # 2D nearest neighbors case
range_2D_nnn, DOS_2D_nnn_values = func.DOS_2D_nnn(t1, t2, a, kx_grid, ky_grid, eta, bound) # 2D next-nearest neighbors case




'''
Plotting energy bands (1D)
'''
fig, ax = plt.subplots(figsize=(8, 6)) # figure & axis definition
ax.plot(k_vec, energies_1D_nn, label=r'nearest neighbors 1D', color='b')  #plot of nn 1D band
ax.plot(k_vec, energies_1D_nnn, label=r'next-nearest neighbors 1D', color='r') #plot of nnn 1D band

ax.set_xlabel(r'Wave Vector $k$')
ax.set_ylabel(r'Energy $\epsilon(k)$')
ax.set_xlim(-np.pi/a, np.pi/a) # setting graph limits
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
xticks = [-np.pi/a, -np.pi/(2*a), 0, np.pi/(2*a), np.pi/a]  #ticks at ±π/a and ±π/2a
xtick_labels = [r'$-\pi/a$', r'$-\pi/2a$', '0', r'$+\pi/2a$', r'$+\pi/a$']
ax.tick_params(axis='x', direction='in') # ticks set inside the graph
ax.tick_params(axis='y', direction='in')
ax.set_xticks(xticks)  # Set the ticks and labels on the x-axis
ax.set_xticklabels(xtick_labels)
ax.set_title('Energy Bands for 1D Case')
ax.legend()
plt.tight_layout()
plt.show()


#Add plots of energy bands 2D cases (NN & NNN) for high symmetry paths!!!

'''
Plotting color map of energy bands for 2D case
'''
func.color_map(kx_grid, ky_grid, energies_2D_nn, 'NN') #Color map nearest neighbors
func.color_map(kx_grid, ky_grid, energies_2D_nnn, 'NNN') #Color map next-nearest neighbors


'''
Plotting color gradient of energy bands for 2D case
'''
func.dos_plotter(range_1D_nn, DOS_1D_nn_values, "DOS for 1D NN") #DOS 1D nearest neighbors

func.dos_plotter(range_1D_nnn, DOS_1D_nnn_values, "DOS for 1D NNN") #DOS 1D next-nearest neighbors

func.dos_plotter(range_2D_nn, DOS_2D_nn_values, "DOS for 2D NN") #DOS 2D nearest neighbors

func.dos_plotter(range_2D_nnn, DOS_2D_nnn_values,"DOS for 2D NNN") #DOS 2D next-nearest neighbors

