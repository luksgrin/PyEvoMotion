"""Smoke tests for PyEvoMotion.PyEvoMotionBase (the Rust port).

Run after `maturin develop --release` from the project root.
"""
import numpy as np
import pytest

from PyEvoMotion import PyEvoMotionBase


def test_count_prefixes():
    assert PyEvoMotionBase.count_prefixes("s", ["s_1_A", "i_2_GT", "s_3_C"]) == 2
    assert PyEvoMotionBase.count_prefixes("i", []) == 0


def test_mutation_length_modification():
    assert PyEvoMotionBase.mutation_length_modification("s_1_A") == 0
    assert PyEvoMotionBase.mutation_length_modification("i_5_GTC") == 3
    assert PyEvoMotionBase.mutation_length_modification("d_8_AAAA") == -4
    with pytest.raises(ValueError):
        PyEvoMotionBase.mutation_length_modification("x_1_A")


def test_weighting_function():
    n = np.array([0.0, 15.0, 30.0, 60.0])
    w = np.tanh(2 * n / 30)
    out = np.asarray(PyEvoMotionBase._weighting_function(n))
    np.testing.assert_allclose(out, w, rtol=1e-12)


def test_linear_regression_basic():
    x = np.arange(20, dtype=np.float64)
    y = 3.0 * x + 5.0 + np.zeros_like(x)
    res = PyEvoMotionBase.linear_regression(x, y)
    assert res["expression"] == "mx + b"
    assert res["parameters"]["m"] == pytest.approx(3.0, rel=1e-9)
    assert res["parameters"]["b"] == pytest.approx(5.0, rel=1e-9)
    assert res["r2"] == pytest.approx(1.0, abs=1e-12)
    # Callable model
    yhat = np.asarray(res["model"](x))
    np.testing.assert_allclose(yhat, y, rtol=1e-12)


def test_linear_regression_no_intercept():
    x = np.arange(20, dtype=np.float64)
    y = 2.5 * x
    res = PyEvoMotionBase.linear_regression(x, y, fit_intercept=False)
    assert res["expression"] == "mx"
    assert "b" not in res["parameters"]
    assert res["parameters"]["m"] == pytest.approx(2.5, rel=1e-9)


def test_power_law_fit_basic():
    x = np.linspace(1, 30, 30)
    y = 2.0 * np.power(x, 1.7)
    res = PyEvoMotionBase.power_law_fit(x, y)
    assert res["expression"] == "d*x^alpha"
    assert res["parameters"]["d"] == pytest.approx(2.0, rel=1e-3)
    assert res["parameters"]["alpha"] == pytest.approx(1.7, rel=1e-3)
    assert res["r2"] == pytest.approx(1.0, abs=1e-3)


def test_inheritance_subclass():
    class Custom(PyEvoMotionBase):
        @classmethod
        def linear_regression(cls, x, y, **kw):
            r = super().linear_regression(x, y, **kw)
            r["from_subclass"] = True
            return r

    x = np.arange(10, dtype=np.float64)
    y = 2 * x + 1
    res = Custom.linear_regression(x, y)
    assert res["from_subclass"] is True
    assert res["parameters"]["m"] == pytest.approx(2.0, rel=1e-9)


def test_compute_confidence_intervals():
    params = {"m": 1.0, "b": 0.5}
    se = {"m": 0.1, "b": 0.05}
    ci = PyEvoMotionBase._compute_confidence_intervals(params, se, 18, 0.95)
    # t_{0.975, 18} ≈ 2.10092
    lo, hi = ci["m"]
    assert lo == pytest.approx(1.0 - 2.10092 * 0.1, rel=1e-3)
    assert hi == pytest.approx(1.0 + 2.10092 * 0.1, rel=1e-3)


def test_remove_nan():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([10.0, 20.0, np.nan, 40.0])
    z = np.array([100.0, 200.0, 300.0, 400.0])
    xc, yc, zc = PyEvoMotionBase._remove_nan(x, y, z)
    np.testing.assert_array_equal(np.asarray(xc), [1.0, 4.0])
    np.testing.assert_array_equal(np.asarray(yc), [10.0, 40.0])
    np.testing.assert_array_equal(np.asarray(zc), [100.0, 400.0])
