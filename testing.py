import calculations as calc
import numpy as np
from hypothesis import strategies as st
from hypothesis import given
import pytest

''' --------testing check_params-------- '''
def test_params_check_empty(): # test that params_check() raises a ValueError if the parameter list is empty.
    with pytest.raises(ValueError, match="Parameter list is empty."):
        calc.params_check([])

def test_params_check_negative_tnn(): # test that params_check() raises a ValueError if nearest neighbor hopping (params[1]) is not positive.
    params = [1, -1, 0.5, 1.0, 100, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Nearest neighbor hopping parameter is invalid."):
        calc.params_check(params)

def test_params_check_invalid_tnnn(): # test that params_check() raises a ValueError if next-nearest neighbor hopping (params[2]) is not positive or exceeds nearest neighbor hopping.
    params_tnnn_negative = [1, 1.0, -0.5, 1.0, 100, 0.1, "gaussian"]
    params_tnnn_exceeds_tnn = [1, 1.0, 2.0, 1.0, 100, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Next-nearest neighbor hopping parameter is invalid."):
        calc.params_check(params_tnnn_negative)
    with pytest.raises(ValueError, match="Next-nearest neighbor hopping parameter is invalid."):
        calc.params_check(params_tnnn_exceeds_tnn)

def test_params_check_invalid_lattice_constant(): # test that params_check() raises a ValueError if lattice constant (params[3]) is not positive.
    params = [1, 1.0, 0.5, -1.0, 100, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Lattice parameter is invalid."):
        calc.params_check(params)

def test_params_check_non_integer_lattice_points(): # test that params_check() raises a ValueError if the number of lattice points (params[4]) is not an integer.
    params = [1, 1.0, 0.5, 1.0, 100.5, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params)

def test_params_check_negative_lattice_points(): # test that params_check() raises a ValueError if the number of lattice points (params[4]) is not positive.
    params = [1, 1.0, 0.5, 1.0, -10, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params)

def test_params_check_low_lattice_points(): # test that params_check() raises a ValueError if the number of lattice points (params[4]) is smaller than 100.
    params = [1, 1.0, 0.5, 1.0, 99, 0.1, "gaussian"]
    with pytest.raises(ValueError, match="Number of lattice points is invalid."):
        calc.params_check(params)

def test_params_check_invalid_width(): # test that params_check() raises a ValueError if the width parameter (params[5]) is not in the (0, 1) range.
    params_width_high = [1, 1.0, 0.5, 1.0, 100, 1.1, "gaussian"]
    params_width_negative = [1, 1.0, 0.5, 1.0, 100, -0.1, "gaussian"]
    with pytest.raises(ValueError, match="Width of the Gaussian/Lorentzian is invalid."):
        calc.params_check(params_width_high)
    with pytest.raises(ValueError, match="Width of the Gaussian/Lorentzian is invalid."):
        calc.params_check(params_width_negative)

def test_params_check_invalid_method(): # test that params_check() raises a ValueError if the method for DOS calculation (params[6]) is not 'gaussian' or 'lorentzian'.
    params = [1, 1.0, 0.5, 1.0, 100, 0.1, "invalid_method"]
    with pytest.raises(ValueError, match="Method for DOS calculation is invalid."):
        calc.params_check(params)

def test_params_check_valid_params(): # test that params_check() does not raise an error for valid parameters
    params = [1, 1.0, 0.5, 1.0, 100, 0.1, "gaussian"]
    # No exception should be raised for valid parameters
    calc.params_check(params)
''' ------------------------------------ '''



''' --------testing TD_1D-------- '''
@given(params=st.tuples(
            st.integers(min_value=1, max_value=2),                    # case (1 or 2)
            st.floats(min_value=0.1, max_value=10).map(lambda x: round(x, 4)),  # tnn with 4 decimal places
            st.floats(min_value=0, max_value=1).map(lambda x: round(x, 4)),     # tnnn/tnn (can be 0) with 4 decimal places
            st.floats(min_value=0.1, max_value=10).map(lambda x: round(x, 3)),  # a with 3 decimal places
            st.integers(min_value=100, max_value=800),               # N
            st.floats(min_value=0.001, max_value=1.0).map(lambda x: round(x, 3)), # width with 3 decimal places
            st.sampled_from(["gaussian", "lorentzian"])))             # method
def test_TB_1D(params):
    k = np.linspace(-np.pi/params[3], np.pi/params[3] , params[4])
    params = list(params)  # convert tuple to list to allow modification
    tnnn = params[2] * params[1] # tnnn goes now from 0 to tnn
    params[2]=tnnn
    result = calc.TB_1D(params, k)
    
    # output shape
    assert result.shape == k.shape

    # output finiteness
    assert np.all(np.isfinite(result))

    # expected energy bands
    if params[0] == 1:
        expected = -2 * params[1] * np.cos(k * params[3])
    elif params[0] == 2:
        expected = -2 * params[1] * np.cos(k * params[3]) - 2 * params[2] * np.cos(k * params[3])
    assert np.allclose(result, expected, rtol=1e-5), "Energy band does not match expected values."

    # symmetry check: E(k) == E(-k)
    assert np.allclose(result, calc.TB_1D(params, -k), rtol=1e-5), "Energy band is not symmetric."

    # bounds check for nn & nnn
    if params[0] == 1:
        assert np.all(result >= -2 * params[1]), "Energy band values are below expected minimum."
        assert np.all(result <= 2 * params[1]), "Energy band values exceed expected maximum."
    elif params[0] == 2:
        assert np.all(result >= -2 * params[1]- 2 * params[2]), "Energy band values are below expected minimum."
        assert np.all(result <= 2 * params[1] + 2* params[2]), "Energy band values exceed expected maximum."
''' ----------------------------- '''


def test_TB_1D_zero_tnn():
    params = [1, 0.0, 0.1, 5.0, 100, 0.01, "lorentzian"]
    k = np.linspace(-np.pi/params[3], np.pi/params[3] , params[4])

    result = calc.TB_1D(params, k)

    # energy values must be all zero
    expected = np.zeros_like(k)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_1D_zero_tnnn():
    params_nn = [1, 1.0, 0.0, 5.0, 100, 0.01, "lorentzian"] #nn case
    params_nnn = [2, 1.0, 0.0, 5.0, 100, 0.01, "lorentzian"] #nnn case
    k = np.linspace(-np.pi/params_nn[3], np.pi/params_nn[3] , params_nn[4])
    
    result_nn = calc.TB_1D(params_nn, k)
    result_nnn = calc.TB_1D(params_nnn, k)
    
    assert np.allclose(result_nn, result_nnn, rtol=1e-5), "Nnn case should collapse on nn case when tnnn is zero."



''' --------testing TD_2D-------- '''
@given(params=st.tuples(
            st.integers(min_value=3, max_value=4),                    # case (3 for NN, 4 for NNN)
            st.floats(min_value=0.1, max_value=10).map(lambda x: round(x, 4)),  # tnn with 4 decimal places
            st.floats(min_value=0, max_value=1).map(lambda x: round(x, 4)),     # tnnn/tnn (can be 0) with 4 decimal places
            st.floats(min_value=0.1, max_value=10).map(lambda x: round(x, 3)),  # a with 3 decimal places
            st.integers(min_value=100, max_value=800),               # N
            st.floats(min_value=0.001, max_value=1.0).map(lambda x: round(x, 3)), # width with 3 decimal places
            st.sampled_from(["gaussian", "lorentzian"])))             # method
def test_TB_2D(params):
    kx_start = np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4])
    ky_start = np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4])
    hexagon = calc.hexagonal_contour(params, kx_start, ky_start, (4*np.pi)/params[3])
    kx, ky = np.meshgrid(kx_start[hexagon],ky_start[hexagon])
    
    
    params = list(params)  # convert tuple to list to allow modification
    tnnn = params[2] * params[1] # tnnn goes now from 0 to tnn
    params[2]=tnnn
    result = calc.TB_2D(params, kx, ky)

    # output shape
    assert result.shape == kx.shape, "Output shape does not match kx array shape."

    # output finiteness
    assert np.all(np.isfinite(result)), "Energy band contains non-finite values."

    # expected energy bands
    if params[0] == 3:
        expected = -2*params[1]*(np.cos(kx*params[3])+2*np.cos(kx*params[3]/2)*np.cos(ky*params[3]*np.sqrt(3)/2))
    elif params[0] == 4:
        expected = -2*params[1]*(np.cos(kx*params[3])+2*np.cos(kx*params[3]/2)*np.cos(ky*params[3]*np.sqrt(3)/2)) - 2*params[2]*(np.cos(ky*params[3]*np.sqrt(3))+2*np.cos(ky*params[3]*np.sqrt(3)/2)*np.cos(kx*params[3]*3/2))
    assert np.allclose(result, expected, rtol=1e-5), "Energy band does not match expected values."

    # symmetry check: E(kx, ky) == E(-kx, -ky)
    assert np.allclose(result, calc.TB_2D(params, -kx, -ky), rtol=1e-5), "Energy band is not symmetric."


def test_TB_2D_nnn_bounds():
    params = [4, 1.0, 0.126, 5.0, 500, 0.01, "lorentzian"]
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4]),
        np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4]))

    #M point has higher energy than K point
    result = calc.TB_2D(params, kx, ky)
    min_energy = - 6 * (params[1] + params[2])
    max_energy = 2 * (params[1] + params[2])

    assert np.all(result >= min_energy), "Energy band values are below expected minimum.(K<M)"
    assert np.all(result <= max_energy), "Energy band values exceed expected maximum.(K<M)"


    #K point has higher energy than M point
    params[2]=0.124
    result = calc.TB_2D(params, kx, ky)
    min_energy = - 6 * (params[1] + params[2])
    max_energy = 3 * (params[1] - 2 * params[2])

    assert np.all(result >= min_energy), "Energy band values are below expected minimum.(K>M)"
    assert np.all(result <= max_energy), "Energy band values exceed expected maximum.(K>M)"

def test_TB_2D_zero_tnn():
    params = [3, 0.0, 0.1, 5.0, 100, 0.01, "lorentzian"]
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4]),
        np.linspace(-(4*np.pi)/params[3], (4*np.pi)/params[3], params[4]))

    result = calc.TB_2D(params, kx, ky)

    # energy values must be all zero
    expected = np.zeros_like(kx)
    assert np.allclose(result, expected, rtol=1e-5), "Energy band is not zero when tnn is zero."

def test_TB_2D_zero_tnnn():
    params_nn = [3, 1.0, 0.0, 5.0, 100, 0.01, "lorentzian"]  # NN case
    params_nnn = [4, 1.0, 0.0, 5.0, 100, 0.01, "lorentzian"]  # NNN case
    kx, ky = np.meshgrid(
        np.linspace(-(4*np.pi)/params_nn[3], (4*np.pi)/params_nn[3], params_nn[4]),
        np.linspace(-(4*np.pi)/params_nn[3], (4*np.pi)/params_nn[3], params_nn[4]))

    result_nn = calc.TB_2D(params_nn, kx, ky)
    result_nnn = calc.TB_2D(params_nnn, kx, ky)

    assert np.allclose(result_nn, result_nnn, rtol=1e-5), ("nnn case should collapse to nn case when tnnn is zero.")
''' ----------------------------- '''
