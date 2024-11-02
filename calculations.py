import numpy as np

def params_check(params):
    if not params: # check if the list is empty
        raise ValueError("Parameter list is empty. Please initialize the sequence of paramters.")
    
    if params[1] <= 0: #check if tnn is positive
        raise ValueError("Nearest neighbor hopping parameter is invalid. Please insert a positive value.")
    
    if params[2] <= 0 or params[2] > params[1]: #check if tnnn is positive and less than tnn
        raise ValueError("Next-nearest neighbor hopping parameter is invalid. Please insert a positive value, which is less than the nearest neighbor hopping parameter.")

    if params[3] <= 0: # check if a is positive
        raise ValueError("Lattice parameter is invalid. Please insert a positive value.")

    if not isinstance(params[4], int): # check if N is an integer
        raise ValueError("Number of lattice points is invalid. Please insert a positive integer value.")

    if params[4] <= 0: # check if N is positive
        raise ValueError("Number of lattice points is invalid. Please insert a positive integer value.")

    if params[5] >= 1 or params[5] <= 0: # check if the width of the gaussian/lorentzian is valid
        raise ValueError("Width of the Gaussian/Lorentzian is invalid. Please insert a positive value which is less than 1.")

    if params[6] != "gaussian" and params[6] != "lorentzian": #check if the method string is valid 
        raise ValueError("Method for DOS calculation is invalid. Please insert either 'gaussian' or 'lorentzian'.")

    print("List of inserted parameters is valid. The calculation will start now!")
    return


def hexagonal_contour(params, kx, ky, bound):
    # Define the boundary lines for the hexagon with required size 
    if bound >= 0:
        contour1 = ky <= -np.sqrt(3) * (kx - bound)
        contour2 = ky >= -np.sqrt(3) * (kx + bound)
        contour3 = ky >= np.sqrt(3) * (kx - bound)
        contour4 = ky <= np.sqrt(3) * (kx + bound)
        contour5 = np.abs(ky) <= bound
        
        # Define the hexagon
        hexagon = contour1 & contour2 & contour3 & contour4 & contour5
    else:
        raise ValueError("Error when calling function hexagonal_contour(). Please insert a positive value for the bound.")

    return hexagon


def high_symmetry_path(params):
    """
    Generates the high-symmetry path in the Brillouin zone for a 2D tight-binding model.

    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[3]: Lattice constant 'a'.
        - params[4]: Number of points in each segment of the high-symmetry path.
    
    Returns:
    --------
    kx_path : ndarray
        The x-components of the wave vector along the high-symmetry path.
    
    ky_path : ndarray
        The y-components of the wave vector along the high-symmetry path.

    Notes:
    ------
    The path follows the sequence Γ -> K -> M -> Γ in the Brillouin zone.
    """
    G = [0, 0]
    M = [np.pi / params[3], np.pi / (np.sqrt(3) * params[3])]
    K = [4 * np.pi / (3 * params[3]), 0]

    # Γ -> K
    kx_segment1 = np.linspace(G[0], K[0], params[4])
    ky_segment1 = np.linspace(G[1], K[1], params[4])
    # K -> M
    kx_segment2 = np.linspace(K[0], M[0], params[4])
    ky_segment2 = np.linspace(K[1], M[1], params[4])
    # M -> Γ
    kx_segment3 = np.linspace(M[0], G[0], params[4])
    ky_segment3 = np.linspace(M[1], G[1], params[4])
    
    kx_path = np.concatenate([kx_segment1, kx_segment2, kx_segment3])
    ky_path = np.concatenate([ky_segment1, ky_segment2, ky_segment3])
    return kx_path, ky_path


def TB_1D(params, k):
    """
    Calculates the energy band for a one-dimensional (1D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (1 for NN, 2 for NNN).
        - params[1]: The hopping parameter for nearest neighbors.
        - params[2]: The hopping parameter for next-nearest neighbors (used when params[0] == 2).
        - params[3]: Lattice constant 'a'.
        
    k : array-like
        Wave vector values for the 1D system.
        
    Returns:
    --------
    energy_values : ndarray
        The energy values corresponding to each wave vector.

    Raises:
    -------
    ValueError
        If params[0] is not 1 or 2.
    """
    energy_values = np.zeros_like(k)
    if params[0]==1:
        energy_values = -2 * params[1] * np.cos(k * params[3])
    elif params[0]==2:
        energy_values = -2 * params[1] * np.cos(k * params[3]) - 2 * params[2] * np.cos(k * params[3])
    else:
        raise ValueError("Error: when calling TB_1D(), the case value is invalid.")
    return energy_values


def TB_2D(params, kx, ky):
    """
    Calculates the energy band for a two-dimensional (2D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (3 for NN, 4 for NNN).
        - params[1]: The hopping parameter for nearest neighbors.
        - params[2]: The hopping parameter for next-nearest neighbors (used when params[0] == 4).
        - params[3]: Lattice constant 'a'.
        
    kx : array-like
        Wave vector values in the x-direction.
        
    ky : array-like
        Wave vector values in the y-direction.
        
    Returns:
    --------
    energy_values : ndarray
        The energy values corresponding to each (kx, ky) pair.
        
    Raises:
    -------
    ValueError
        If params[0] is not 3 or 4.
    """
    energy_values = np.zeros_like(kx)
    if params[0]==3:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2))
    elif params[0]==4:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2)) - 2 * params[2] * (np.cos(ky * params[3] * np.sqrt(3)) + 2 * np.cos(ky * params[3] * np.sqrt(3) / 2) * np.cos(kx * params[3] * 3 / 2))
    else:
        raise ValueError("Error: when calling TB_2D(), the case value is invalid.")
    return energy_values
    


def DOS_1D(params, energy_values):
    """
    Calculates the density of states (DOS) for a one-dimensional (1D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (1 for NN, 2 for NNN).
        - params[4]: The number of points for the DOS calculation.
        - params[5]: Broadening factor (used for Gaussian or Lorentzian smoothing).
        - params[6]: String specifying the broadening method ("gaussian" or "lorentzian").
        
    energies : array-like
        Energy values for which the DOS will be calculated.
        
    Returns:
    --------
    dos_range : ndarray
        The energy range for which the DOS is calculated.
        
    dos_values : ndarray
        The DOS values corresponding to the energy range.

    Raises:
    -------
    ValueError
        If params[6] is not "gaussian" or "lorentzian".
    """
    dos_range = np.linspace(min(energy_values)*1.1, max(energy_values)*1.1, params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    

    for i, E in enumerate(dos_range):
        if params[6] == "gaussian":  # Gaussian approximation
            weights = np.exp(-((E - energy_values) ** 2) / (params[5] ** 2)) / (np.sqrt(np.pi) * params[5])
        elif params[6] == "lorentzian":  # Lorentzian approximation
            weights = (params[5] / np.pi) / ((E - energy_values) ** 2 + params[5] ** 2)
        else:
            raise ValueError("Error: Invalid approximation method specified. Choose either gaussian or lorentzian.")
        dos_values[i] = np.sum(weights) / (params[4] ** 2)

    # Normalization of DOS using trapezoidal integration
    total_area = np.trapz(dos_values, dos_range)
    dos_values /= total_area

    return dos_range, dos_values


def DOS_2D(params, energy_mesh):
    """
    Calculates the density of states (DOS) for a two-dimensional (2D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (3 for NN, 4 for NNN).
        - params[4]: The number of points for the DOS calculation.
        - params[5]: Broadening factor (used for Gaussian or Lorentzian smoothing).
        - params[6]: String specifying the broadening method ("gaussian" or "lorentzian").
        
    energies_mesh : ndarray
        A 2D array representing the energy values across the (kx, ky) grid.
        
    Returns:
    --------
    dos_range : ndarray
        The energy range for which the DOS is calculated.
        
    dos_values : ndarray
        The DOS values corresponding to the energy range.
        
    Raises:
    -------
    ValueError
        If params[6] is not "gaussian" or "lorentzian".
    """
    energy_values = energy_mesh.flatten()
    dos_range = np.linspace(min(energy_values)*1.1,max(energy_values)*1.1, params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    

    for i, E in enumerate(dos_range):
        if params[6] == "gaussian":  # Gaussian case
            weights = np.exp(-((E - energy_values) ** 2) / (params[5] ** 2)) / (np.sqrt(np.pi) * params[5])
        elif params[6] == "lorentzian":  # Lorentzian case
            weights = (params[5] / np.pi) / ((E - energy_values) ** 2 + params[5] ** 2)        
        else:
            raise ValueError("Error: Invalid approximation method specified. Choose either gaussian or lorentzian.")
        dos_values[i] = np.sum(weights) / (params[4] ** 2)

    # Normalization of DOS using trapezoidal integration
    total_area = np.trapz(dos_values, dos_range)
    dos_values /= total_area

    return dos_range, dos_values