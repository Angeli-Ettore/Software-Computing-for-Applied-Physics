import matplotlib.pyplot as plt
import numpy as np
import functions as calc

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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(k, energy_values, label = energy_label, color = 'red')
    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")

    ax1.set_xlim(-np.pi/params[3], np.pi/params[3]) # setting graph limits
    xticks = [-np.pi/params[3], -np.pi/(2*params[3]), 0, np.pi/(2*params[3]), np.pi/params[3]]  #ticks at ±π/a and ±π/2a
    xtick_labels = [r'$-\pi/a$', r'$-\pi/2a$', '0', r'$+\pi/2a$', r'$+\pi/a$']
    ax1.tick_params(axis='x', direction='in') # ticks set inside the graph
    ax1.tick_params(axis='y', direction='in')
        
    ax1.set_xticks(xticks)  # Set the ticks and labels on the x-axis
    ax1.set_xticklabels(xtick_labels)

    ax1.grid(True)
    ax1.legend()
    
    # Plot on the second subplot
    ax2.plot(dos_values*1000, dos_range, color = 'red', label = dos_label)
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right
    ax2.yaxis.set_label_position('right')  # Move the label to the right
    ax2.set_ylabel(r"Energy $\epsilon(k)$")  # Set the label and add some padding for clarity
    ax2.tick_params(axis='x', direction='in') # ticks set inside the graph
    ax2.tick_params(axis='y', direction='in')
    ax2.grid(True)
    ax2.legend()
    
    if params[0]==1:
        fig.suptitle(f"Energy Band & Density of States in 1D (NN case: {params[8]})", fontsize=14)
    elif params[0]==2:
        fig.suptitle(f"Energy Band & Density of States in 1D (NNN case: {params[8]})", fontsize=14)
    else:
        raise ValueError("Error: when calling Energy_and_DOS_1D_plotter(), the case value is invalid.")
        
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0})
    
    # Plot on the first subplot
    ax1.plot(path, energy_path_values, color = 'red', label = energy_label)    
    ax1.set_xticks([0, params[4], 2*params[4], 3*params[4]], path_label)

    ax1.set_xlabel(r"Wave Vector $k$")
    ax1.set_ylabel(r"Energy $\epsilon(k)$")
    ax1.tick_params(axis='x', direction='in') # ticks set inside the graph
    ax1.tick_params(axis='y', direction='in')
    ax1.grid(True)
    ax1.legend()
    
    # Plot on the second subplot
    ax2.plot(dos_values, dos_range, color = 'red', label = dos_label)
    ax2.set_xlabel("Density of States")
    ax2.yaxis.set_ticks_position('right')  # Move ticks to the right
    ax2.yaxis.set_label_position('right')  # Move the label to the right
    ax2.set_ylabel(r"Energy $\epsilon(k)$")  # Set the label and add some padding for clarity
    ax2.tick_params(axis='x', direction='in') # ticks set inside the graph
    ax2.tick_params(axis='y', direction='in')
    ax2.grid(True)
    ax2.legend()
    
    if params[0]==3:
        fig.suptitle(f"Energy Band & Density of States in 2D (NN case: {params[8]})", fontsize=14)
    elif params[0]==4:
        fig.suptitle(f"Energy Band & Density of States in 2D (NNN case: {params[8]})", fontsize=14)
    else:
        raise ValueError("Error: when calling Energy_and_DOS_2D_plotter(), the case value is invalid.")
        
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
    plt.title(f"Energy Band for 2D {case}")
    plt.show()
    return 