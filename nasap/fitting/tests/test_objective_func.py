import numpy as np
import numpy.typing as npt
import pytest

from nasap.fitting import make_objective_func_from_ode_rhs
from nasap.fitting.sample_data import get_a_to_b_sample


def test():
    sample = get_a_to_b_sample()
    
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, sample.y, sample.t[0], sample.y0)
    
    np.testing.assert_allclose(
        objective_func(*sample.params), 0.0, atol=1e-3)


def test_ydata_with_nan():
    sample = get_a_to_b_sample()
    ydata = sample.y.copy()

    # Pre-test
    ydata += 0.1
    # RSS should be 0.1^2 * ydata.size
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, ydata, sample.t[0], sample.y0)
    expected = 0.1**2 * ydata.size
    np.testing.assert_allclose(
        objective_func(*sample.params), expected)
    
    # Test
    ydata[0][0] = np.nan
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, ydata, sample.t[0], sample.y0)
    expected -= 0.1**2  # Subtract the contribution of the first element
    np.testing.assert_allclose(
        objective_func(*sample.params), expected)


def test_unpack_args():
    # Imaginary simulating function with minimum at (0, 0)
    def sim_func(t, y0, k1, k2):  # Receives unpacked arguments
        return np.zeros((len(t), len(y0))) + k1**2 + k2**2
    
    # Imaginary data
    tdata = np.array([0.0, 1.0, 2.0])
    ydata = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    y0 = np.array([0.0, 0.0])
    
    # By specifying unpack_args=True, the array of the fitting parameters,
    # i.e., np.array([k1, k2]), will be unpacked into k1 and k2
    # before passing them to sim_func
    objective_func = make_objective_func_from_simulating_func(
        tdata, ydata, sim_func, y0, unpack_args=True)

    result = differential_evolution(objective_func, [(-1, 1), (-1, 1)])

    assert result.success
    assert np.allclose(result.x, [0, 0])


def test_not_unpack_args():
    # Imaginary simulating function with minimum at (0, 0)
    def sim_func(t, y0, ks):  # Receives packed arguments
        k1 = ks[0]
        k2 = ks[1]
        return np.zeros((len(t), len(y0))) + k1**2 + k2**2
    
    # Imaginary data
    tdata = np.array([0.0, 1.0, 2.0])
    ydata = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    y0 = np.array([0.0, 0.0])
    
    # By specifying unpack_args=False, the array of the fitting parameters,
    # i.e., np.array([k1, k2]), will be passed as a single argument ks
    # to sim_func
    objective_func = make_objective_func_from_simulating_func(
        tdata, ydata, sim_func, y0, unpack_args=False)  # Do not unpack arguments

    result = differential_evolution(objective_func, [(-1, 1), (-1, 1)])

    assert result.success
    assert np.allclose(result.x, [0, 0])


if __name__ == '__main__':
    pytest.main(['-v', __file__])
