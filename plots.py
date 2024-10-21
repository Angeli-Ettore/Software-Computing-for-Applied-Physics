import matplotlib.pyplot as plt
import numpy as np
import functions as calc
import os


# Set up global plot settings using rcParams
plt.rcParams.update({
    'lines.color': 'blue',             # Set default line color
    'lines.linewidth': 2,            # Set default line width
    'axes.labelsize': 12,            # Font size for axis labels
    'axes.titlesize': 14,            # Font size for figure titles
    'axes.grid': True,               # Enable grid by default
    'xtick.direction': 'in',         # X-tick direction inside
    'ytick.direction': 'in',         # Y-tick direction inside
    'figure.figsize': (10, 5),       # Default figure size
    'savefig.bbox': 'tight',         # Save figures with tight layout
})



def Energy_and_DOS_1D_plotter(params, k):
    """
    Plots the energy band and density of states (DOS) for a one-dimensional (1D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters required for the tight-binding calculation:
        - params[0]: An integer representing the case (1 for NN, 2 for NNN).
        - params[3]: Lattice constant 'a'.
        - params[8]: A description of the case being used (e.g., "NN" or "NNN").
        
    k : array-like
        Wave vector values for the 1D system.
        
    Returns:
    --------
    None
        The function displays the plot but does not return any value.
        
    Notes:
    ------
    - This function raises a ValueError if `params[0]` is not 1 or 2.
    - The plot includes the energy band and the density of states for the given parameters.
    """
    energy_values, energy_label = calc.TB_1D(params, k)
    dos_range, dos_values, dos_label = calc.DOS_1D(params, energy_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(k, energy_values, label = energy_label)
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")

    ax1.set_xlim(-np.pi/params[3]*1.1, np.pi/params[3]*1.1) # setting x limits
    ax1.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits
    
    xticks = [-np.pi/params[3], -np.pi/(2*params[3]), 0, np.pi/(2*params[3]), np.pi/params[3]]  #ticks at ±π/a and ±π/2a
    xtick_labels = [r'$-\pi/a$', r'$-\pi/2a$', '0', r'$+\pi/2a$', r'$+\pi/a$']
        
    ax1.set_xticks(xticks)  # Set the ticks and labels on the x-axis
    ax1.set_xticklabels(xtick_labels)
    ax1.legend()
    
    
    # Plot on the second subplot
    ax2.plot(dos_values*1000, dos_range, color = 'red', label = dos_label)
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right

    ax2.set_ylim(min(energy_values)-1, max(energy_values)+1) # setting y limits

    ax2.legend()
    
    if params[0] == 1:
        case = "NN"
    elif params[0] == 2:
        case = "NNN"
    else:
        raise ValueError("Error: when calling Energy_and_DOS_1D_plotter(), the case value is invalid.")
    
    fig.suptitle(f"Energy Band & Density of States (1D {case}: {params[8]})", fontsize=14)

    # Define the directory where the images will be saved
    current_directory = os.path.dirname(os.path.abspath(__file__))
    images_directory = os.path.join(current_directory, "Images")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    
    # Save the plot using the 'label' variable
    filename = f"EnergyBand_DOS_1D_{case}_{params[8]}.jpg"  # Create the filename using the label variable
    save_path = os.path.join(images_directory, filename)
    plt.savefig(save_path)  # Save the plot with tight layout    

    plt.show()
    return 

def Energy_and_DOS_2D_plotter(params, kx, ky):
    """
    Plots the energy band along a high-symmetry path and the density of states (DOS) for a two-dimensional (2D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters required for the tight-binding calculation:
        - params[0]: An integer representing the case (3 for NN, 4 for NNN).
        - params[4]: Path length used for high-symmetry points.
        - params[8]: A description of the case being used (e.g., "NN" or "NNN").
        
    kx : array-like
        Wave vector values in the x-direction for the 2D system.
        
    ky : array-like
        Wave vector values in the y-direction for the 2D system.
        
    Returns:
    --------
    None
        The function displays the plot but does not return any value.
        
    Notes:
    ------
    - The function automatically sets high-symmetry points ('Γ', 'M', 'K', 'Γ') for the path.
    - This function raises a ValueError if `params[0]` is not 3 or 4.
    - The plot includes the energy band along the high-symmetry path and the density of states for the given parameters.
    """
     
    kx_path, ky_path = calc.high_symmetry_path(params)
    energy_path_values, _ = calc.TB_2D(params, kx_path, ky_path)
    path_label =['Γ', 'M', 'K', 'Γ']
    path = np.linspace(0, 3*params[4], 3*params[4])
    
    energy_values, energy_label = calc.TB_2D(params, kx, ky)
    dos_range, dos_values, dos_label = calc.DOS_2D(params, energy_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(path, energy_path_values, label = energy_label)    
    ax1.set_xticks([0, params[4], 2*params[4], 3*params[4]], path_label)

    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")
    
    ax1.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits

    ax1.legend()
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red', label = dos_label)
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right
    
    ax2.set_ylim(min(energy_path_values)-1, max(energy_path_values)+1) # setting y limits

    ax2.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
    
    if params[0] == 3:
        case = "NN"
    elif params[0] == 4:
        case = "NNN"
    else:
        raise ValueError("Error: when calling Energy_and_DOS_2D_plotter(), the case value is invalid.")
    
    fig.suptitle(f"Energy Band & Density of States (2D {case}: {params[8]})", fontsize=14)
        
    # Define the directory where the images will be saved
    current_directory = os.path.dirname(os.path.abspath(__file__))
    images_directory = os.path.join(current_directory, "Images")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    
    # Save the plot using the 'label' variable
    filename = f"EnergyBand_DOS_2D_{case}_{params[8]}.jpg"  # Create the filename using the label variable
    save_path = os.path.join(images_directory, filename)
    plt.savefig(save_path)  # Save the plot with tight layout    


    plt.show()
    return 


def color_map_plotter(params, kx, ky):
    """
    Plots a contour map for the energy bands of a two-dimensional (2D) tight-binding model based on wave vectors kx and ky.
    
    Parameters:
    -----------
    params : list
        A list of parameters required for the tight-binding calculation:
        - params[0]: An integer representing the case (3 for NN, 4 for NNN).
        
    kx : array-like
        Wave vector values in the x-direction (2D grid).
        
    ky : array-like
        Wave vector values in the y-direction (2D grid).
        
    Returns:
    --------
    None
        The function displays the contour plot but does not return any value.
        
    Notes:
    ------
    - The color map uses a "viridis" colormap to represent the energy bands visually.
    - This function raises a ValueError if `params[0]` is not 3 or 4.
    - The plot includes a color bar that represents energy values for each (kx, ky) pair.
    """
    energy_values, energy_label = calc.TB_2D(params, kx, ky)
    if params[0] == 3:
        case = "NN"
    elif params[0] == 4:
        case = "NNN"
    else:
        raise ValueError("Error: when calling color_map_plotter(), the case value is invalid.")

    plt.contourf(ky, kx, energy_values, cmap="viridis")  # 2D nearest neighbor case
    plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
    plt.xlabel("Wave Vector $k_x$")
    plt.ylabel("Wave Vector $k_y$")
    
    title = f"Energy Band for 2D {case}"
    plt.title(title)

    # Define the directory where the images will be saved
    current_directory = os.path.dirname(os.path.abspath(__file__))
    images_directory = os.path.join(current_directory, "Images")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    
    # Save the plot using the 'label' variable
    filename = f"{title}.jpg"  # Create the filename using the label variable
    save_path = os.path.join(images_directory, filename)
    plt.savefig(save_path)  # Save the plot with tight layout    

    
    plt.show()
    return 