import plots as plot
import calculations as calc
import configparser
from time import time

''' --------definition of working parameters-------- '''
config = configparser.ConfigParser()
config.read('parameters.ini') # load data from the configuration file 'parameters.ini'

tnn_p = float(config['parameters']['tnn']) # nearest neighbor hopping parameter
ratio_p = float(config['parameters']['ratio']) # ratio between next-nearest neighbor and nearest neighbor hopping parameters
tnnn_p = tnn_p * ratio_p # next-nearest neighbor hopping parameter
a_p = float(config['parameters']['a']) # lattice constant
N_p = int(config['parameters']['N']) # number of lattice sites
w_p = float(config['parameters']['w']) # parameter of the gaussian\lorentzian that approximate delta function in DOS calculation
size_p = int(config['parameters']['size'])# number of plotted hexagonal unit cells
method_p = config['parameters']['method'] # method for approximating Dirac's delta in DOS calculation (use only gaussian or lorentzian)
tempo = time() # time at which the calculation starts
''' ------------------------------------------------ '''



''' --------crafting and checking of params-------- '''
params = [tnn_p, tnnn_p, a_p, N_p, w_p, size_p, method_p] 
calc.params_check(params)
''' ----------------------------------------------- '''



''' --------definition of wave vectors (1D & 2D)-------- '''
k, kx_grid, ky_grid = calc.wave_vectors_builder(a_p, N_p, size_p)
''' ---------------------------------------------------- '''



''' --------calculation and plotting of energy bands and DOS-------- '''
filename_1D_nn = "Energy Band and Density of States (1D nn)"
energy_values = calc.TB_1D_nn(tnn_p, a_p, k)
dos_range, dos_values = calc.DOS_1D(tnn_p, N_p, w_p, method_p, energy_values)
plot.Energy_and_DOS_1D_plotter(filename_1D_nn, a_p, k, energy_values, dos_range, dos_values)


filename_1D_nnn = "Energy Band and Density of States (1D nnn)"
energy_values = calc.TB_1D_nnn(tnn_p, tnnn_p, a_p, k)
dos_range, dos_values = calc.DOS_1D(tnn_p, N_p, w_p, method_p, energy_values)
plot.Energy_and_DOS_1D_plotter(filename_1D_nnn, a_p, k, energy_values, dos_range, dos_values)


filename_2D_nn = "Energy Band and Density of States (2D nn)"
energy_mesh = calc.TB_2D_nn(tnn_p, a_p, kx_grid, ky_grid)
dos_range, dos_values = calc.DOS_2D(tnn_p, N_p, w_p, method_p, energy_mesh)
plot.Energy_and_DOS_2D_plotter(filename_2D_nn, tnn_p, tnnn_p, a_p, kx_grid, ky_grid, energy_mesh, dos_range, dos_values)
plot.color_map_plotter("Color Map of the Energy Band (2D nn)", a_p, kx_grid, ky_grid, energy_mesh)


filename_2D_nnn = "Energy Band and Density of States (2D nnn)"
energy_mesh = calc.TB_2D_nnn(tnn_p, tnnn_p, a_p, kx_grid, ky_grid)
dos_range, dos_values = calc.DOS_2D(tnn_p, N_p, w_p, method_p, energy_mesh)
plot.Energy_and_DOS_2D_plotter(filename_2D_nnn, tnn_p, tnnn_p, a_p, kx_grid, ky_grid, energy_mesh, dos_range, dos_values)
plot.color_map_plotter("Color Map of the Energy Band (2D nnn)", a_p, kx_grid, ky_grid, energy_mesh)
''' ---------------------------------------------------------------- '''



print(f"Code executed successfully in {time() - tempo:.2f} seconds!")