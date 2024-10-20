import numpy as np 

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
    The path follows the sequence Γ -> M -> K -> Γ in the Brillouin zone.
    """
    G = [0, 0]
    M = [np.pi / params[3], np.pi / (np.sqrt(3) * params[3])]
    K = [4 * np.pi / (3 * params[3]), 0]

    # Γ -> M
    kx_segment1 = np.linspace(G[0], M[0], params[4])
    ky_segment1 = np.linspace(G[1], M[1], params[4])
    # M -> K
    kx_segment2 = np.linspace(M[0], K[0], params[4])
    ky_segment2 = np.linspace(M[1], K[1], params[4])
    # K -> Γ
    kx_segment3 = np.linspace(K[0], G[0], params[4])
    ky_segment3 = np.linspace(K[1], G[1], params[4])
    
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
        
    energy_label : str
        A string label indicating the type of energy band (NN or NNN).

    Raises:
    -------
    ValueError
        If params[0] is not 1 or 2.
    """
    energy_values = np.zeros_like(k)
    if params[0]==1:
        energy_values = -2 * params[1] * np.cos(k * params[3])
        energy_label = "Energy band for 1D NN"
    elif params[0]==2:
        energy_values = -2 * params[1] * np.cos(k * params[3]) - 2 * params[2] * np.cos(k * params[3])
        energy_label = "Energy band for 1D NNN"
    else:
        raise ValueError("Error: when calling TB_1D(), the case value is invalid.")
    return energy_values, energy_label


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
        
    energy_label : str
        A string label indicating the type of energy band (NN or NNN).

    Raises:
    -------
    ValueError
        If params[0] is not 3 or 4.
    """
    energy_values = np.zeros_like(kx)
    if params[0]==3:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2))
        energy_label = "Energy band for 2D NN"
    elif params[0]==4:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2)) - 2 * params[2] * (np.cos(ky * params[3] * np.sqrt(3)) + 2 * np.cos(ky * params[3] * np.sqrt(3) / 2) * np.cos(ky * params[3] * 3 / 2))
        energy_label = "Energy band for 2D NNN"
    else:
        raise ValueError("Error: when calling TB_2D(), the case value is invalid.")
    return energy_values, energy_label
    

def DOS_1D(params, energies):
    """
    Calculates the density of states (DOS) for a one-dimensional (1D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (1 for NN, 2 for NNN).
        - params[4]: The number of points for the DOS calculation.
        - params[5]: Broadening factor (used for Gaussian or Lorentzian smoothing).
        - params[6]: Minimum energy value for the DOS calculation range.
        - params[7]: Maximum energy value for the DOS calculation range.
        - params[8]: String specifying the broadening method ("gaussian" or "lorentzian").
        
    energies : array-like
        Energy values for which the DOS will be calculated.
        
    Returns:
    --------
    dos_range : ndarray
        The energy range for which the DOS is calculated.
        
    dos_values : ndarray
        The DOS values corresponding to the energy range.
        
    dos_label : str
        A string label indicating the type of DOS (NN or NNN).

    Raises:
    -------
    ValueError
        If params[0] is not 1 or 2.
        If params[8] is not "gaussian" or "lorentzian".
    """
    dos_range = np.linspace(params[6], params[7], params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    dos_label = "miao"
    
    if params[0] == 1:  # nearest neighbor case (1D)
        dos_label = "DOS for 1D NN"
    elif params[0] == 2:  # next-nearest neighbor case (1D)
        dos_label = "DOS for 1D NNN"
    else:
        raise ValueError("Error: when calling DOS_1D(), the case value is invalid.")

    for i, E in enumerate(dos_range):
        if params[8] == "gaussian":  # Gaussian approximation
            weights = np.exp(-((E - energies) ** 2) / (params[5] ** 2)) / (np.sqrt(np.pi) * params[5])
        elif params[8] == "lorentzian":  # Lorentzian approximation
            weights = (params[5] / np.pi) / ((E - energies) ** 2 + params[5] ** 2)
        else:
            raise ValueError("Error: Invalid approximation method specified. Choose either gaussian or lorentzian.")
        dos_values[i] = np.sum(weights) / (params[4] ** 2)

    return dos_range, dos_values, dos_label


def DOS_2D(params, energies_mesh):
    """
    Calculates the density of states (DOS) for a two-dimensional (2D) tight-binding model.
    
    Parameters:
    -----------
    params : list
        A list of parameters where:
        - params[0]: An integer representing the case (3 for NN, 4 for NNN).
        - params[4]: The number of points for the DOS calculation.
        - params[5]: Broadening factor (used for Gaussian or Lorentzian smoothing).
        - params[6]: Minimum energy value for the DOS calculation range.
        - params[7]: Maximum energy value for the DOS calculation range.
        - params[8]: String specifying the broadening method ("gaussian" or "lorentzian").
        
    energies_mesh : ndarray
        A 2D array representing the energy values across the (kx, ky) grid.
        
    Returns:
    --------
    dos_range : ndarray
        The energy range for which the DOS is calculated.
        
    dos_values : ndarray
        The DOS values corresponding to the energy range.
        
    dos_label : str
        A string label indicating the type of DOS (NN or NNN).

    Raises:
    -------
    ValueError
        If params[0] is not 3 or 4.
        If params[8] is not "gaussian" or "lorentzian".
    """
    dos_range = np.linspace(params[6], params[7], params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    dos_label = "miao"    
    energies = energies_mesh.flatten()
    
    if params[0] == 3:  # nearest neighbor case (2D)
        dos_label = "DOS for 2D NN"
    elif params[0] == 4:  # next-nearest neighbor case (2D)
        dos_label = "DOS for 2D NNN"
    else:
        raise ValueError("Error: when calling DOS_2D(), the case value is invalid.")

    for i, E in enumerate(dos_range):
        if params[8] == "gaussian":  # Gaussian case
            weights = np.exp(-((E - energies) ** 2) / (params[5] ** 2)) / (np.sqrt(np.pi) * params[5])
        elif params[8] == "lorentzian":  # Lorentzian case
            weights = (params[5] / np.pi) / ((E - energies) ** 2 + params[5] ** 2)        
        else:
            raise ValueError("Error: Invalid approximation method specified. Choose either gaussian or lorentzian.")
        dos_values[i] = np.sum(weights) / (params[4] ** 2)

    return dos_range, dos_values, dos_label
