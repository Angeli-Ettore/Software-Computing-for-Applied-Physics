import numpy as np 
import matplotlib.pyplot as plt


def TB_1D_nn(k, t1, a):
    """
    Computes the energy dispersion for a 1D tight-binding model with only nearest neighbors.

    Parameters:
    k (array): Wave vector values (1D array).
    t1 (float): Nearest neighbor hopping parameter.
    a (float): Lattice constant.

    Returns:
    array: Energy values corresponding to each wave vector in k.
    """
    return -2 * t1 * np.cos(k * a)

def TB_1D_nnn(k, t1, t2, a):
    """
    Computes the energy dispersion for a 1D tight-binding model including both nearest 
    and next-nearest neighbors.

    Parameters:
    k (array): Wave vector values (1D array).
    t1 (float): Nearest neighbor hopping parameter.
    t2 (float): Next-nearest neighbor hopping parameter.
    a (float): Lattice constant.

    Returns:
    array: Energy values corresponding to each wave vector in k.
    """
    return TB_1D_nn(k, t1, a) - 2 * t2 * np.cos(k * a)

def TB_2D_nn(kx, ky, t1, a):
    """
    Computes the energy dispersion for a 2D tight-binding model with only nearest neighbors.

    Parameters:
    kx (array): Wave vector values in the x-direction (2D grid).
    ky (array): Wave vector values in the y-direction (2D grid).
    t1 (float): Nearest neighbor hopping parameter.
    a (float): Lattice constant.

    Returns:
    array: Energy values corresponding to each (kx, ky) pair in the grid.
    """
    return -2 * t1 * (np.cos(kx * a) + 2 * np.cos(kx * a / 2) * np.cos(ky * a * np.sqrt(3) / 2))

def TB_2D_nnn(kx, ky, t1, t2, a):
    """
    Computes the energy dispersion for a 2D tight-binding model including both nearest 
    and next-nearest neighbors.

    Parameters:
    kx (array): Wave vector values in the x-direction (2D grid).
    ky (array): Wave vector values in the y-direction (2D grid).
    t1 (float): Nearest neighbor hopping parameter.
    t2 (float): Next-nearest neighbor hopping parameter.
    a (float): Lattice constant.

    Returns:
    array: Energy values corresponding to each (kx, ky) pair in the grid.
    """
    return TB_2D_nn(kx, ky, t1, a) - 2 * t2 * (np.cos(ky * a * np.sqrt(3)) + 2 * np.cos(ky * a * np.sqrt(3) / 2) * np.cos(ky * a * 3 / 2))


def color_map_plotter(kx, ky, energies, case):
    """
    Plots a color map for the energy bands based on wave vectors kx and ky.

    Parameters:
    kx (array): Wave vector values in the x-direction (2D grid).
    ky (array): Wave vector values in the y-direction (2D grid).
    energies (array): Energy values for each (kx, ky) pair.
    case (str): Description of the case being plotted ("NN" or "NNN").

    Returns:
    None
    """
    plt.contourf(ky, kx, energies, cmap="viridis")  # 2D nearest neighbor case
    plt.colorbar(label="Energy $\epsilon(k_x, k_y)$")
    plt.xlabel("Wave Vector $k_x$")
    plt.ylabel("Wave Vector $k_y$")
    plt.title(f"Energy Band for 2D {case}")
    plt.show()
    return 

def dos_plotter(x, y, params, label):
    """
    Plots the Density of States (DOS) based on energy values and the corresponding DOS values.
    
    The function automatically determines the plotting method (Gaussian or Lorentzian) 
    based on the parameter settings in `params`. It also sets up the plot labels, titles, 
    and displays the graph.

    Parameters:
    x (array): An array of energy values over which the DOS is calculated.
    y (array): An array of DOS values corresponding to each energy value in `x`.
    params (list): A list of parameters used for DOS calculation.
        - params[8] specifies the method for DOS calculation:
            * 1: Gaussian method
            * 2: Lorentzian method
    label (str): A string label for the plot, typically describing the method or case.

    Raises:
    ValueError: If `params[8]` is not 1 or 2, indicating an invalid method value.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=label)
    plt.xlabel("Energy")
    plt.ylabel("Density of States")
    
    if params[8] == 1:
        plt.title(f"{label}: Gaussian method")
    elif params[8] == 2:
        plt.title(f"{label}: Lorentzian method")
    else:
        raise ValueError("Error: when calling dos_plotter(), the method value is invalid. Must be 1 (Gaussian) or 2 (Lorentzian).")
    
    plt.grid(True)
    plt.legend()
    plt.show()


def omni_DOS(params, k, k_x, k_y):
    """
    Computes the Density of States (DOS) based on the selected tight-binding model and 
    energy range using either Gaussian or Lorentzian functions for the Dirac delta approximation.

    Parameters:
    params (list): List of parameters as follows:
        - params[0] (int): Case identifier (1=1Dnn, 2=1Dnnn, 3=2Dnn, 4=2Dnnn).
        - params[1] (float): Nearest neighbor hopping parameter (t1).
        - params[2] (float): Next-nearest neighbor hopping parameter (t2).
        - params[3] (float): Lattice constant (a).
        - params[4] (int): Number of lattice points (N).
        - params[5] (float): Width parameter (eta for Gaussian, gamma for Lorentzian).
        - params[6] (float): Lower limit of energy range.
        - params[7] (float): Upper limit of energy range.
        - params[8] (int): Method for Dirac delta approximation (1=Gaussian, 2=Lorentzian).
    k (array): Wave vector values for 1D cases.
    k_x (array): Wave vector values in the x-direction (2D cases).
    k_y (array): Wave vector values in the y-direction (2D cases).

    Raises:
    ValueError:  when calling omni_DOS(), the case value is invalid. Must be between 1 and 4.")

    Returns:
    tuple: A tuple containing:
        - rng (array): Energy range values.
        - dos (array): Computed DOS values corresponding to the energy range.
    """
    rng = np.linspace(params[6], params[7], params[4])  # Energy values for which the DOS is calculated
    dos = np.zeros(params[4])

    if params[0] == 1:  # nearest neighbor case (1D)
        energies = TB_1D_nn(k, params[1], params[3])
    
    elif params[0] == 2:  # next-nearest neighbor case (1D)
        energies = TB_1D_nnn(k, params[1], params[2], params[3])             

    elif params[0] == 3:  # nearest neighbor case (2D)
        energies = TB_2D_nn(k_x, k_y, params[1], params[3]).flatten()

    elif params[0] == 4:  # next-nearest neighbor case (2D)
        energies = TB_2D_nnn(k_x, k_y, params[1], params[2], params[3]).flatten()
        
    else:
        raise ValueError("Error: when calling omni_DOS(), the case value is invalid. Must be between 1 and 4.")

    for i, E in enumerate(rng):
        if params[8] == 1:  # Gaussian case
            weights = np.exp(-((E - energies) ** 2) / (params[5] ** 2)) / (np.sqrt(np.pi) * params[5])
        
        elif params[8] == 2:  # Lorentzian case
            weights = (params[5] / np.pi) / ((E - energies) ** 2 + params[5] ** 2)
        
        else:
            raise ValueError("Error: Invalid approximation method specified. Choose either 1 (gaussian) or 2(lorentzian).")

        dos[i] = np.sum(weights) / (params[4] ** 2)

    return rng, dos
