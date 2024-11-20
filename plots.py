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
        none
    Notes:
        creates a folder 'Images' in th esame directory as the code, in which downloading the images.
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

def Energy_and_DOS_1D_plotter(params, k, energy_values, dos_range, dos_values):
    '''
    plots energy band and DOS in the tight-binding model (1D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[3]: lattice constant 'a'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.        
        k : array of the wave vector (1D)     
    Output:
        none (plots the graphs)
    Raises:
        ValueError
            if params[0] is not 1 or 2.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(k, energy_values)
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")

    ax1.set_xlim(-np.pi/params[3]*1.1, np.pi/params[3]*1.1) # setting x limits
    ax1.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits
    
    xticks = [-np.pi/params[3], -np.pi/(2*params[3]), 0, np.pi/(2*params[3]), np.pi/params[3]]  #ticks at ±π/a and ±π/2a
    xtick_labels = [r'$-\pi/a$', r'$-\pi/2a$', '0', r'$+\pi/2a$', r'$+\pi/a$']
        
    ax1.set_xticks(xticks)  # Set the ticks and labels on the x-axis
    ax1.set_xticklabels(xtick_labels)
    
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red')
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right
    ax2.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits
    
    if params[0] == 1:
        case = "NN"
    elif params[0] == 2:
        case = "NNN"
    else:
        raise ValueError("Error: when calling Energy_and_DOS_1D_plotter(), the case value is invalid.")
    
    title = f"Energy Band and Density of States (1D {case} {params[6]})"
    fig.suptitle(title, fontsize=14)
    saving_pictures(title)

    plt.show()
    return 

def Energy_and_DOS_2D_plotter(params, kx, ky, energy_values, dos_range, dos_values):
    '''
    plots energy band and DOS in the tight-binding model (2D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[3]: lattice constant 'a'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.        
        kx: array of the x component of the wave vector (2D)
        ky: array of the y component of the wave vector (2D)
    Output:
        none (plots the graphs)
    Raises:
        ValueError
            if params[0] is not 3 or 4.
    '''
     
    kx_path, ky_path = calc.high_symmetry_path(params)
    energy_path_values = calc.TB_2D(params, kx_path, ky_path)
    path_label =['Γ', 'K', 'M', 'Γ']
    path = np.linspace(0, 3*params[4], 3*params[4])
        
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(path, energy_path_values)    
    ax1.set_xticks([0, params[4], 2*params[4], 3*params[4]], path_label)
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")    
    ax1.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red')
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right    
    ax2.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits
    
    if params[0] == 3:
        case = "NN"
    elif params[0] == 4:
        case = "NNN"
    else:
        raise ValueError("Error: when calling Energy_and_DOS_2D_plotter(), the case value is invalid.")
    
    title = f"Energy Band and Density of States (2D {case} {params[6]})"
    fig.suptitle(title, fontsize=14)
    saving_pictures(title)

    plt.show()
    return 


def color_map_plotter(params, kx, ky, energy_values):
    '''
    plots color map of the energy band in the tight-binding model (2D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[3]: lattice constant 'a'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.        
        kx: array of the x component of the wave vector (2D)
        ky: array of the y component of the wave vector (2D)
        energies_values : 2D array of the energy band across the (kx,ky) grid 
    Output:
        none (plots the color map)
    Raises:
        ValueError
            if params[0] is not 3 or 4.
    '''
    if params[0] == 3:
        case = "NN"
    elif params[0] == 4:
        case = "NNN"
    else:
        raise ValueError("Error: when calling color_map_plotter(), the case value is invalid.")
    plt.figure(figsize=(10, 7))
    plt.contourf(kx, ky, energy_values, levels=50, cmap="turbo")  # 2D nearest neighbor case
    plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
    plt.xlabel("Wave Vector $k_x$")
    plt.ylabel("Wave Vector $k_y$")
    
    #plotting the FBZ as a black line    
    fbz = calc.hexagonal_contour(params, kx, ky, (4*np.pi)/(3*params[3]))
    plt.contour(kx, ky, fbz, colors='white')

    title = f"Energy Map (2D {case})"
    plt.title(title)
    saving_pictures(title)
    
    plt.show()
    return 