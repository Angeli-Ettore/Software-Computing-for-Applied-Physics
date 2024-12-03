import matplotlib.pyplot as plt
import numpy as np
import calculations as calc
import os

# Set up global plot settings using rcParams
plt.rcParams.update({
    'lines.color': 'blue',           # Set default line color
    'lines.linewidth': 2,            # Set default line width
    'axes.labelsize': 12,            # Font size for axis labels
    'axes.titlesize': 14,            # Font size for figure titles
    'axes.grid': True,               # Enable grid by default
    'xtick.direction': 'in',         # X-tick direction inside
    'ytick.direction': 'in',         # Y-tick direction inside
    'figure.figsize': (10, 5),       # Default figure size
    'savefig.bbox': 'tight',         # Save figures with tight layout
})

def saving_pictures(filename):
    '''
    save a generated graph as pdf file with filename corresponding to its title.
    Input:
        filename : name of the generated file, title of the graph
    Output:
        none (plots the graphs)
    Notes:
        creates a folder 'Images' in the same directory as the code, in which it saves the image.
    '''

    # Define the directory where the images will be saved
    current_directory = os.path.dirname(os.path.abspath(__file__))
    images_directory = os.path.join(current_directory, "Images")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    
    # Save the plot using the 'label' variable
    save_path = os.path.join(images_directory, f"{filename}.pdf")
    plt.savefig(save_path)  # Save the plot with tight layout    

    return

def Energy_and_DOS_1D_plotter(filename, a, k, energy_values, dos_range, dos_values):
    '''
    plots energy band and DOS in the 1D tight-binding model.
    Input:
        filename: name to give to the generated file, should be indicative of the dimensionality and case.
        a: lattice constant.
        k : array of the wave vector (1D).
        energy_values : array of the energy band.
        dos_range : array of the energy range for which the DOS is plotted.
        dos_values : array of the normalized DOS values for the corresponding energy range.
    Output:
        none (plots the graphs)
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(k, energy_values)
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")

    ax1.set_xlim(-np.pi/a*1.1, np.pi/a*1.1) # setting x limits
    ax1.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits
    
    xticks = [-np.pi/a, -np.pi/(2*a), 0, np.pi/(2*a), np.pi/a]  #ticks at ±π/a and ±π/2a
    xtick_labels = [r'$-\pi/a$', r'$-\pi/2a$', '0', r'$+\pi/2a$', r'$+\pi/a$']
        
    ax1.set_xticks(xticks)  # Set the ticks and labels on the x-axis
    ax1.set_xticklabels(xtick_labels)
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red')
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right
    ax2.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits
        
    fig.suptitle(filename, fontsize=14)
    saving_pictures(filename)

    plt.show()
    return 

def Energy_and_DOS_2D_plotter(filename, tnn, tnnn, a, kx, ky, energy_values, dos_range, dos_values):
    '''
    plots energy band and DOS in the 2D tight-binding model.
    Input:
        filename: name to give to the generated file, should be indicative of the dimensionality and case.
        tnn: hopping parameter for nearest neighbors.
        tnnn: hopping parameter for next-nearest neighbors.
        a: lattice constant.
        kx: array of the x component of the wave vector (2D).
        ky: array of the y component of the wave vector (2D).
        energy_values : array of the energy band.
        dos_range : array of the energy range for which the DOS is plotted.
        dos_values : array of the normalized DOS values for the corresponding energy range.
    Output:
        none (plots the graphs)
    '''
    N = len(kx)
    kx_path, ky_path = calc.high_symmetry_path(a, N)
    
    if filename == "Energy Band and Density of States (2D nn)":
        energy_path_values = calc.TB_2D_nn(tnn, a, kx_path, ky_path)
    elif filename == "Energy Band and Density of States (2D nnn)":
        energy_path_values = calc.TB_2D_nnn(tnn, tnnn, a, kx_path, ky_path)
        
    path_label =['Γ', 'K', 'M', 'Γ']
    path = np.linspace(0, 3*N, 3*N)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(path, energy_path_values)    
    ax1.set_xticks([0, N, 2*N, 3*N], path_label)
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")    
    ax1.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red')
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right    
    ax2.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits
    
    fig.suptitle(filename, fontsize=14)
    saving_pictures(filename)

    plt.show()
    return 


def color_map_plotter(filename, a, kx, ky, energy_values):
    '''
    plots color map of the energy band in the tight-binding model (2D) with FBZ (white line).
    Input:
        filename: name to give to the generated file, should be indicative of the dimensionality and case.
        a: lattice constant.
        kx: array of the x component of the wave vector (2D).
        ky: array of the y component of the wave vector (2D).
        energies_values : 2D array of the energy band across the (kx,ky) grid.
    Output:
        none (plots the color map)
    '''
    plt.figure(figsize=(10, 7))
    plt.contourf(kx, ky, energy_values, levels=50, cmap="turbo")  # 2D nearest neighbor case
    plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
    plt.xlabel("Wave Vector $k_x$")
    plt.ylabel("Wave Vector $k_y$")
    
    #plotting the FBZ as a white line    
    fbz = calc.hexagonal_contour(a, 0.33333, kx, ky)
    plt.contour(kx, ky, fbz, colors='white')

    plt.title(filename)
    saving_pictures(filename)
    
    plt.show()
    return 