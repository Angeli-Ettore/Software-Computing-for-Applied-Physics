import numpy as np

def params_check(params):
    '''
    check the validity of each inserted paramter.
    Input:
        params : list
            - params[1]: hopping parameter for nearest neighbors 'tnn'.
            - params[2]: hopping parameter for next-nearest neighbors 'tnnn'.
            - params[3]: lattice constant 'a'.
            - params[4]: number of points 'N'.
            - params[5]: width of the function approximating a Dirac's delta 'w'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.
    Output:
        none
    Raises:
        ValueError
            if params is empty
            if tnn < 0
            if tnnn < 0 or tnnn > tnn
            if a <= 0
            if N is not an integer or N < 99
            if w >= 1 or w <=0
            if method is not gaussian or lorentzian
    Notes:
        prints the list of inserted parameters
    '''
    if not params: # check if the list is empty
        raise ValueError("Parameter list is empty. Please initialize the sequence of paramters.")
    
    if params[1] < 0: #check if 'tnn' is positive
        raise ValueError("Nearest neighbor hopping parameter is invalid. Please insert a positive value.")
    
    if params[2] < 0 or params[2] > params[1]: #check if 'tnnn' is positive and less or equal to tnn
        raise ValueError("Next-nearest neighbor hopping parameter is invalid. Please insert a positive value, which is less than the nearest neighbor hopping parameter.")

    if params[3] <= 0: # check if 'a' is positive
        raise ValueError("Lattice parameter is invalid. Please insert a positive value.")

    if not isinstance(params[4], int): # check if 'N' is an integer
        raise ValueError("Number of lattice points is invalid. Please insert a positive integer value.")

    if params[4] <= 99: # check if 'N' is positive and greater than 99
        raise ValueError("Number of lattice points is invalid. Please insert a positive integer higher than 99.")

    if params[5] >= 1 or params[5] <= 0: # check if the width of the gaussian/lorentzian is valid
        raise ValueError("Width of the Gaussian/Lorentzian is invalid. Please insert a positive value which is less than 1.")

    if params[6] != "gaussian" and params[6] != "lorentzian": #check if the method string is valid 
        raise ValueError("Method for DOS calculation is invalid. Please insert either 'gaussian' or 'lorentzian'.")

    print("List of inserted parameters is valid:\n"
        f"tnn = {params[1]}\n"
        f"tnnn = {params[2]}\n"
        f"a = {params[3]}\n"
        f"N = {params[4]}\n"
        f"width = {params[5]}\n"
        f"method = {params[6]}\n"
        "the calculation will start now..."
    ) 
    return


def hexagonal_contour(kx, ky, bound):
    '''
    define the boundary lines for the hexagon with required size.
    Input:
        params : list
            - bound: positive float which defines the height of the hexagon side.
        kx: array of the x component of the wave vector (2D)
        ky: array of the y component of the wave vector (2D)
    Output:
        hexagon: hexagonal contour of kx and ky
    Raises:
        ValueError
            if bound < 0
    '''
    if bound >= 0:
        contour1 = ky <= -np.sqrt(3) * (kx - bound)
        contour2 = ky >= -np.sqrt(3) * (kx + bound)
        contour3 = ky >= np.sqrt(3) * (kx - bound)
        contour4 = ky <= np.sqrt(3) * (kx + bound)
        contour5 = np.abs(ky) <= bound
        
        hexagon = contour1 & contour2 & contour3 & contour4 & contour5 # Define the hexagon
    else:
        raise ValueError("Error when calling function hexagonal_contour(). Please insert a positive value for the bound.")
    return hexagon


def high_symmetry_path(params):
    '''
    generates the high-symmetry path Γ -> K -> M -> Γ in the First Brillouin Zone.
    Input:
        params : list
            - params[3]: lattice constant 'a'.
            - params[4]: number of points 'N'.
    Output:
        kx_path : array of the x-components of the wave vector along the high-symmetry path.
        ky_path : array of the y-components of the wave vector along the high-symmetry path.
    '''
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
    '''
    calculates the energy band in the tight-binding model (1D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[1]: hopping parameter for nearest neighbors 'tnn'.
            - params[2]: hopping parameter for next-nearest neighbors 'tnnn'.
            - params[3]: lattice constant 'a'.
        k : array of the wave vector (1D)
    Output:
        energy_values : array of the energy band 
    Raises:
        ValueError
            if params[0] is not 1 or 2.
    '''
    energy_values = np.zeros_like(k)
    if params[0]==1:
        energy_values = -2 * params[1] * np.cos(k * params[3])
    elif params[0]==2:
        energy_values = - 2 * params[1] * np.cos(k * params[3]) - 2 * params[2] * np.cos(2 * k * params[3])
    else:
        raise ValueError("Error: when calling TB_1D(), the case value is invalid.")
    return energy_values


def TB_2D(params, kx, ky):
    '''
    calculates the energy band in the tight-binding model (2D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (3=nn, 4=nnn).
            - params[1]: hopping parameter for nearest neighbors 'tnn'.
            - params[2]: hopping parameter for next-nearest neighbors 'tnnn'.
            - params[3]: lattice constant 'a'.
        kx: array of the x component of the wave vector (2D)
        ky: array of the y component of the wave vector (2D)
    Output:
        energy_values : array of the energy band 
    Raises:
        ValueError
            if params[0] is not 3 or 4.
    '''
    energy_values = np.zeros_like(kx)
    if params[0]==3:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2))
    elif params[0]==4:
        energy_values = -2 * params[1] * (np.cos(kx * params[3]) + 2 * np.cos(kx * params[3] / 2) * np.cos(ky * params[3] * np.sqrt(3) / 2)) - 2 * params[2] * (np.cos(ky * params[3] * np.sqrt(3)) + 2 * np.cos(ky * params[3] * np.sqrt(3) / 2) * np.cos(kx * params[3] * 3 / 2))
    else:
        raise ValueError("Error: when calling TB_2D(), the case value is invalid.")
    return energy_values
    


def DOS_1D(params, energy_values):
    '''
    calculates the density of states in the tight-binding model (1D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[4]: number of points 'N'.
            - params[5]: width of the function approximating a Dirac's delta 'w'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.
        energies : array of the energy band 
    Output:
        dos_range : array of the energy range for which the DOS is calculated.
        dos_values : array of the normalized DOS values for the corresponding energy range
    Raises:
        ValueError
            If params[6] is not "gaussian" or "lorentzian".
    Notes:
        if the energy band is zero everywhere, the DOS is manually set to zero without normalization.
    '''
    dos_range = np.linspace(min(energy_values)*1.1, max(energy_values)*1.1, params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    
    if params[1] != 0:
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
    else:
       print("0 nearest neighbor hopping parameter define 0 Density of States.")     
    return dos_range, dos_values


def DOS_2D(params, energy_mesh):
    '''
    calculates the density of states in the tight-binding model (2D).
    Input:
        params : list
            - params[0]: case for the 1D & 2D calculation (1=nn, 2=nnn).
            - params[4]: number of points 'N'.
            - params[5]: width of the function approximating a Dirac's delta 'w'.
            - params[6]: string specifying the approximation method 'gaussian' or 'lorentzian'.
        energies_mesh : 2D array of the energy band across the (kx,ky) grid 
    Output:
        dos_range : array of the energy range for which the DOS is calculated.
        dos_values : array of the normalized DOS values for the corresponding energy range
    Raises:
        ValueError
            If params[6] is not "gaussian" or "lorentzian".
    Notes:
        if the energy band is zero everywhere, the DOS is manually set to zero without normalization.
    '''
    energy_values = energy_mesh.flatten()
    dos_range = np.linspace(min(energy_values)*1.1,max(energy_values)*1.1, params[4])  # Energy values for which the DOS is calculated
    dos_values = np.zeros(params[4])
    
    if params[1] != 0:
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
    else:
       print("0 nearest neighbor hopping parameter define 0 Density of States.")     

    return dos_range, dos_values