import calculations as calc
import numpy as np
import pytest

''' --------testing check_params-------- '''
def test_params_check_empty():
    '''
    test that params_check() raises a ValueError if the parameter list is empty.
    '''
    with pytest.raises(ValueError, match="Parameter list is empty."):
        calc.params_check([])

def test_params_check_negative_tnn(): 
    '''
    test that params_check() raises a ValueError if nearest neighbor hopping 'tnn' is not positive.
    '''
    params = [-1, 0.5, 1.0, 100, 0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Nearest neighbor hopping parameter is invalid."):
        calc.params_check(params)

def test_params_check_negative_tnnn():
    '''
    test that params_check() raises a ValueError if next-nearest neighbor hopping 'tnnn' is not positive.
    '''
    params = [1.0, -0.5, 1.0, 100, 0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Next-nearest neighbor hopping parameter is invalid."):
        calc.params_check(params)

def test_params_check_negative_lattice_constant():
    '''
    test that params_check() raises a ValueError if lattice constant 'a' is not positive.
    '''
    params = [1.0, 0.5, -1.0, 100, 0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Lattice parameter is invalid."):
        calc.params_check(params)

def test_params_check_non_integer_lattice_points():
    '''
    test that params_check() raises a ValueError if number of lattice points 'N' is not an integer.
    '''
    params = [1.0, 0.5, 1.0, 100.5, 0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params)

def test_params_check_invalid_lattice_points():
    '''
    test that params_check() raises a ValueError if the number of lattice points 'N' is below 100 or above 5000.
    '''
    params_low_N = [1.0, 0.5, 1.0, 99, 0.1, 1, 'gaussian']
    params_high_N = [1.0, 0.5, 1.0, 5001, 0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params_low_N)
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params_high_N)

def test_params_check_invalid_width():
    '''
    test that params_check() raises a ValueError if the width parameter 'w' is negative or higher than 1.
    '''
    params_width_high = [1.0, 0.5, 1.0, 100, 1.1, 1, 'gaussian']
    params_width_negative = [1.0, 0.5, 1.0, 100, -0.1, 1, 'gaussian']
    with pytest.raises(ValueError, match="Width of the Gaussian/Lorentzian is invalid."):
        calc.params_check(params_width_high)
    with pytest.raises(ValueError, match="Width of the Gaussian/Lorentzian is invalid."):
        calc.params_check(params_width_negative)

def test_params_check_negative_size():
    '''
    test that params_check() raises a ValueError if the parameter 'size' is negative.
    '''
    params = [1.0, 0.5, 1.0, 100, 0.1, -1, 'gaussian']
    with pytest.raises(ValueError, match="Size of the color map is invalid."):
        calc.params_check(params)

def test_params_check_invalid_method():
    '''
    test that params_check() raises a ValueError if the method for DOS calculation is not 'gaussian' or 'lorentzian'.  
    '''
    params = [1.0, 0.5, 1.0, 100, 0.1, 1, 'invalid method']
    with pytest.raises(ValueError, match="Method for DOS calculation is invalid."):
        calc.params_check(params)

def test_params_check_valid_params():
    '''
    test that params_check() does not raise an error for valid parameters
    '''
    params = [1.0, 0.5, 1.0, 100, 0.1, 1, 'gaussian'] # No exception should be raised for valid parameters
    calc.params_check(params)
''' ------------------------------------ '''



''' --------testing TD_1D-------- '''
def test_TB_1D_nn_symmetry():
    '''
    test that TB_1D_nn() defines a symmetric energy band by comparing it to the one generated with -k.
    '''
    tnn = 1.0
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band_nn = calc.TB_1D_nn(tnn, a, k)
    symmetric_band_nn = calc.TB_1D_nn(tnn, a, -k)
    
    assert np.allclose(band_nn, symmetric_band_nn, rtol=1e-5), "Energy band (1D nn) is not symmetric."

def test_TB_1D_nnn_symmetry():
    '''
    test that TB_1D_nnn() defines a symmetric energy band by comparing it to the one generated with -k.
    '''
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band_nnn = calc.TB_1D_nnn(tnn, tnnn, a, k)
    symmetric_band_nnn = calc.TB_1D_nnn(tnn, tnnn, a, -k)
    
    assert np.allclose(band_nnn, symmetric_band_nnn, rtol=1e-5), "Energy band (1D nnn) is not symmetric."

def test_TB_1D_nn_boundaries():
    '''
    test that TD_1D_nn() defines an energy band which is contained in the expected boundaries.
    '''
    tnn = 1.0
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    result = calc.TB_1D_nn(tnn, a, k)

    assert np.all(result >= -2 * tnn), f"Energy band values are below expected minimum: -2*{tnn}"
    assert np.all(result <= 2 * tnn), f"Energy band values exceed expected maximum: 2*{tnn}"

def test_TB_1D_nnn_boundaries_low_interaction():
    '''
    test that TD_1D_nnn() defines an energy band which is contained in the expected boundaries.
    if tnnn < 0.25*tnn, the absolute maxima of the band are at k=±π/a
    '''
    tnn = 1.0
    tnnn_low = 0.24 * tnn
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    result_low = calc.TB_1D_nnn(tnn, tnnn_low, a, k)

    assert np.all(result_low >= - 2 * tnn - 2 * tnnn_low), f"Energy band values are below expected minimum: -2*{tnn}"
    assert np.all(result_low <= 2 * tnn + 2 * tnnn_low), f"Energy band values exceed expected maximum: 2*{tnn}"

def test_TB_1D_nnn_boundaries_high_interaction():
    '''
    test that TD_1D_nnn() defines an energy band which is contained in the expected boundaries.
    if tnnn > 0.25*tnn, the absolute maxima of the band are shifted towards k=±π/2a
    '''
    tnn = 1.0
    tnnn_high = 0.26 * tnn
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    result_high = calc.TB_1D_nnn(tnn, tnnn_high, a, k)

    assert np.all(result_high >= - 2 * tnn - 2 * tnnn_high), f"Energy band values are below expected minimum: -2*{tnn}"
    assert np.all(result_high <= 2 * tnn + 2 * tnnn_high), f"Energy band values exceed expected maximum: 2*{tnn}"

def test_TB_1D_nn_zero_tnn():
    '''
    test that TD_1D_nn() defines a flat energy band when the hopping parameter 'tnn' is set to 0.
    '''
    tnn = 0.0
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)

    result = calc.TB_1D_nn(tnn, a, k)

    # energy values must be all zero
    expected = np.zeros_like(k)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_1D_nnn_zero_tnn():
    '''
    test that TD_1D_nnn() defines a flat energy band when the hopping parameter 'tnn' is set to 0.
    '''
    tnn = 0.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)

    result = calc.TB_1D_nnn(tnn, tnnn, a, k)

    # energy values must be all zero
    expected = np.zeros_like(k)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_1D_zero_tnnn():
    '''
    test that, if next-neighbors hopping parameter 'tnnn' is set to 0, TB_1D_nn() and TB_1D_nnn() generate the same energy band.
    '''
    tnn = 0.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    k = np.linspace(-np.pi/a, np.pi/a , N)
    
    result_nn = calc.TB_1D_nn(tnn, a, k)
    result_nnn = calc.TB_1D_nnn(tnn, tnnn, a, k)
    
    assert np.allclose(result_nn, result_nnn, rtol=1e-5), "Nnn case should collapse on nn case when tnnn is zero."
''' ----------------------------- '''



''' --------testing TD_2D-------- '''
def test_TB_2D_nn_symmetry():
    '''
    test that TB_2D_nn() defines a symmetric energy band by comparing it to the one generated with -k.
    '''
    tnn = 1.0
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band_nn = calc.TB_2D_nn(tnn, a, kx, ky)
    symmetric_band_nn = calc.TB_2D_nn(tnn, a, -kx, -ky)

    assert np.allclose(band_nn, symmetric_band_nn, rtol=1e-5), "Energy band (2D nn) is not symmetric."

def test_TB_2D_nnn_symmetry():
    '''
    test that TB_2D_nnn() defines a symmetric energy band by comparing it to the one generated with -k.
    '''
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band_nnn = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)
    symmetric_band_nnn = calc.TB_2D_nnn(tnn, tnnn, a, -kx, -ky)

    assert np.allclose(band_nnn, symmetric_band_nnn, rtol=1e-5), "Energy band (2D nnn) is not symmetric."

def test_TB_2D_nn_boundaries():
    '''
    test that TD_2D_nn() defines an energy band which is contained in the expected boundaries.
    '''
    tnn = 1.0
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    result = calc.TB_2D_nn(tnn, a, kx, ky)

    assert np.all(result >= -6 * tnn), f"Energy band (2D nn) values are below expected minimum: -6*{tnn}"
    assert np.all(result <= 3 * tnn), f"Energy band (2D nn) values exceed expected maximum: 3*{tnn}"

def test_TB_2D_nnn_boundaries_low_interaction():
    '''
    test that TD_2D_nn() defines an energy band which is contained in the expected boundaries.
    if tnnn < 0.125*tnn, the absolute maximum of the band is at the K point
    '''
    tnn = 1.0
    tnnn_low = 0.124 * tnn
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    result_low = calc.TB_2D_nnn(tnn, tnnn_low, a, kx, ky)

    assert np.all(result_low >= - 6 * tnn - 6 * tnnn_low), f"Energy band (2D nnn) values are below expected minimum: -6*{tnn}-6*{tnnn_low}"
    assert np.all(result_low <= 3 * tnn - 6 * tnnn_low), f"Energy band (2D nnn) values exceed expected maximum: 3*{tnn}-6*{tnnn_low}"

def test_TB_2D_nnn_boundaries_high_interaction():
    '''
    test that TD_2D_nn() defines an energy band which is contained in the expected boundaries.
    if tnnn > 0.125*tnn, the absolute maximum of the band is at the M point
    '''
    tnn = 1.0
    tnnn_high = 0.126 * tnn
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    result_high = calc.TB_2D_nnn(tnn, tnnn_high, a, kx, ky)

    assert np.all(result_high >= - 6 * tnn - 6 * tnnn_high), f"Energy band (2D nnn) values are below expected minimum: -6*{tnn}-6*{tnnn_high}"
    assert np.all(result_high <= 2 * tnn + 2 * tnnn_high), f"Energy band (2D nnn) values exceed expected maximum: 2*{tnn}+2*{tnnn_high}"

def test_TB_2D_nn_zero_tnn():
    '''
    test that TD_2D_nn() defines a flat energy band when the hopping parameter 'tnn' is set to 0.
    '''
    tnn = 0.0
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))

    result = calc.TB_2D_nn(tnn, a, kx, ky)

    # energy values must be all zero
    expected = np.zeros_like(kx)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_2D_nnn_zero_tnn():
    '''
    test that TD_2D_nnn() defines a flat energy band when the hopping parameter 'tnn' is set to 0.
    '''
    tnn = 0.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))

    result = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)

    # energy values must be all zero
    expected = np.zeros_like(kx)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_2D_zero_tnnn():
    '''
    test that, if next-neighbors hopping parameter 'tnnn' is set to 0, TB_2D_nn() and TB_2D_nnn() generate the same energy band.
    '''
    tnn = 1.0
    tnnn = 0.0
    a = 1.0
    N = 100
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    
    result_nn = calc.TB_2D_nn(tnn, a, kx, ky)
    result_nnn = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)
    
    assert np.allclose(result_nn, result_nnn, rtol=1e-5), "Nnn case should collapse on nn case when tnnn is zero."
''' ----------------------------- '''



''' --------testing DOS_1D-------- '''
def test_DOS_1D_nn_length():
    '''
    test that in the nn case DOS_1D() outputs, dos_range and dos_values, have the expected length, corresponding to the number of lattice points N.
    '''  
    tnn = 1.0
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band = calc.TB_1D_nn(tnn, a, k)
    dos_range, dos_values = calc.DOS_1D(tnn, N, w, method, band)
    
    assert len(dos_range) == N, "DOS range length does not match expected (1D nn)."
    assert len(dos_values) == N, "DOS values length does not match expected (1D nn)."

def test_DOS_1D_nnn_length():
    '''
    test that in the nnn case DOS_1D() outputs, dos_range and dos_values, have the expected length, corresponding to the number of lattice points N.
    '''  
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band = calc.TB_1D_nnn(tnn, tnnn, a, k)
    dos_range, dos_values = calc.DOS_1D(tnn, N, w, method, band)
    
    assert len(dos_range) == N, "DOS range length does not match expected (1D nnn)."
    assert len(dos_values) == N, "DOS values length does not match expected (1D nnn)."

def test_DOS_1D_normalization():
    '''
    test that DOS_1D() defines a Density of States which is normalized to 1.
    '''
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band = calc.TB_1D_nnn(tnn, tnnn, a, k)
    dos_range, dos_values = calc.DOS_1D(tnn, N, w, method, band)
    
    assert np.isclose(np.trapz(dos_values, dos_range), 1.0, rtol=1e-3), "DOS normalization failed."
    
def test_DOS_1D_zero_tnn():
    '''
    test that DOS_1D() defines a Density of States which is 0 everywhere if tnn is set to 0.
    '''
    tnn = 0.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    k = np.linspace(-np.pi/a, np.pi/a , N)
    band = calc.TB_1D_nnn(tnn, tnnn, a, k)
    dos_range, dos_values = calc.DOS_1D(tnn, N, w, method, band)
    
    expected = np.zeros_like(k)
    assert np.allclose(dos_values, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."
''' ----------------------------- '''


''' --------testing DOS_2D-------- '''
def test_DOS_2D_nn_length():
    '''
    test that in the nn case DOS_2D() outputs, dos_range and dos_values, have the expected length, corresponding to the number of lattice points N.
    '''  
    tnn = 1.0
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band = calc.TB_2D_nn(tnn, a, kx, ky)
    dos_range, dos_values = calc.DOS_2D(tnn, N, w, method, band)
    
    assert len(dos_range) == N, "DOS range length does not match expected (2D nn)."
    assert len(dos_values) == N, "DOS values length does not match expected (2D nn)."

def test_DOS_2D_nnn_length():
    '''
    test that in the nnn case DOS_2D() outputs, dos_range and dos_values, have the expected length, corresponding to the number of lattice points N.
    '''  
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)
    dos_range, dos_values = calc.DOS_2D(tnn, N, w, method, band)
    
    assert len(dos_range) == N, "DOS range length does not match expected (2D nnn)."
    assert len(dos_values) == N, "DOS values length does not match expected (2D nnn)."
    
def test_DOS_2D_normalization():
    '''
    test that DOS_2D() defines a Density of States which is normalized to 1.
    '''
    tnn = 1.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)
    dos_range, dos_values = calc.DOS_2D(tnn, N, w, method, band)
    
    assert np.isclose(np.trapz(dos_values, dos_range), 1.0, rtol=1e-3), "DOS normalization failed."
    
def test_DOS_2D_zero_tnn():
    '''
    test that DOS_2D() defines a Density of States which is 0 everywhere if tnn is set to 0.
    '''
    tnn = 0.0
    tnnn = 0.5 * tnn
    a = 1.0
    N = 100
    w = 0.01
    method = 'lorentzian'
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    band = calc.TB_2D_nnn(tnn, tnnn, a, kx, ky)
    dos_range, dos_values = calc.DOS_2D(tnn, N, w, method, band)
    
    expected = np.zeros_like(kx)
    assert np.allclose(dos_values, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."
''' ----------------------------- '''




''' --------testing gaussian_weights-------- '''
def test_gaussian_weights_empty_band():
    '''
    test that gaussian_weights() returns an empty array if the energy band is empty
    '''
    w = 0.01
    E = 1.0
    E_k = np.array([])
    expected = np.array([])
    assert np.array_equal(calc.gaussian_weights(w, E, E_k), expected), "Function defines non-empty weigth array when it's given empty energy band"

def test_gaussian_weights_zero_band():
    '''
    test that gaussian_weights() returns the expected array of weights if the energy band is all equal to zero
    '''
    N = 100
    w = 0.01
    E = 1.0
    E_k = np.zeros(N)
    result = calc.gaussian_weights(w, E, E_k)
    expected = np.full(N, np.exp(-(E ** 2) / (w ** 2)))
    
    assert np.array_equal(result, expected), "Function defines wrong weigth array when it's given zero-like energy band"
''' ---------------------------------------- '''

''' --------testing lorentzian_weights-------- '''
def test_lorentzian_weights_empty_band():
    '''
    test that lorentzian_weights() returns an empty array if the energy band is empty
    '''
    w = 0.01
    E = 1.0
    E_k = np.array([])
    expected = np.array([])
    assert np.array_equal(calc.lorentzian_weights(w, E, E_k), expected), "Function defines non-empty weigth array when it's given empty energy band"
    
def test_lorentzian_weights_zero_band():
    '''
    test that lorentzian_weights() returns the expected array of weights if the energy band is all equal to zero
    '''
    N = 100
    w = 0.01
    E = 1.0
    E_k = np.zeros(N)
    result = calc.lorentzian_weights(w, E, E_k)
    expected = np.full(N, 1 / (E ** 2 + w ** 2))
    
    assert np.array_equal(result, expected), "Function defines wrong weigth array when it's given zero-like energy band"
''' ---------------------------------------- '''



''' --------testing hexagonal_contour-------- '''
def test_hexagonal_contour_size_error():
    '''
    test that hexagonal_contour() correctly raise the ValueError if the input 'size' is lower than 0    
    '''
    a = 1.0
    N = 100
    size = -1
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    
    with pytest.raises(ValueError, match="Error when calling function hexagonal_contour()."):
        calc.hexagonal_contour(a, size, kx, ky)
        
def test_hexagonal_contour_boundaries():
    '''
    test that when applying hexagonal_contour() on the 2D wave vector components, the required hexagonal boundaries are respected.
    '''
    a = 1.0
    N = 100
    size = 1
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N),
        np.linspace(-(4*np.pi)/a, (4*np.pi)/a, N))
    hexagon = calc.hexagonal_contour(a, size, kx, ky)    
    kx_grid, ky_grid = np.meshgrid(kx[hexagon], ky[hexagon])
    
    assert np.all(ky_grid >= -size*(4 * np.pi)/a), f"2D wave vector x component is below expected minimum: -{size}*(4)π/a"
    assert np.all(ky_grid <= size*(4 * np.pi)/a), f"2D wave vector x component is above expected maximum: {size}*(4)π/a"
''' ----------------------------------------- '''


''' --------testing wave_vectors_builder-------- '''
def test_wave_vectors_builder_boundaries():
    '''
    test that wave_vectors_builder defines wave vectors with the required length   
    '''
    a = 1.0
    N = 100
    size = 1
    k, kx_grid, ky_grid = calc.wave_vectors_builder(a, N, size)

    assert np.all(k >= -np.pi/a), "1D wave vector values are below expected minimum: -π/a"
    assert np.all(k <= np.pi/a), "1D wave vector values are above expected maximum: π/a"
    
    assert np.all(kx_grid >= -size*(4*np.pi)/a), f"2D wave vector x component values are below expected minimum: -{size}*4*π/a"
    assert np.all(kx_grid <= size*(4*np.pi)/a), f"1D wave vector k values are above expected maximum: {size}*4*π/a"

    assert np.all(ky_grid >= -size*(8*np.pi)/(a*np.sqrt(3))), f"2D wave vector x component values are below expected minimum: -{size}*(8*π)/(a*sqrt(3))"
    assert np.all(ky_grid <= size*(8*np.pi)/(a*np.sqrt(3))), f"2D wave vector y component values are above expected maximum: {size}*(8*π)/(a*sqrt(3))"
''' ----------------------------------------- '''


''' --------testing high_symmetry_path-------- '''
def test_high_symmetry_path():
    '''
    test that high_symmetry_path() generates the required path by checking the 4 high symmetry points (Γ - K - M - Γ) located at the known positions    
    '''
    a = 1.0
    N = 100
    kx_path, ky_path = calc.high_symmetry_path(a, N)

    assert kx_path[0] == 0, "Gamma point has wrong x coordinate when calling high_symmetry_path()"
    assert ky_path[0] == 0, "Gamma point has wrong x coordinate when calling high_symmetry_path()"

    assert kx_path[N-1] == 4 * np.pi / (3 * a), "K point has wrong x coordinate when calling high_symmetry_path()"
    assert ky_path[N-1] == 0, "K point has wrong x coordinate when calling high_symmetry_path()"

    assert kx_path[2*N-1] == np.pi / a, "M point has wrong x coordinate when calling high_symmetry_path()"
    assert ky_path[2*N-1] == np.pi / (np.sqrt(3) * a), "M point has wrong x coordinate when calling high_symmetry_path()"

    assert kx_path[3*N-1] == 0, "Gamma point has wrong x coordinate when calling high_symmetry_path()"
    assert ky_path[3*N-1] == 0, "Gamma point has wrong x coordinate when calling high_symmetry_path()"
''' ----------------------------------------- '''

