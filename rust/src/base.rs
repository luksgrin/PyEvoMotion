use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{storage::Owned, Dyn, Matrix, Vector2, U2};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple, PyType};

const N0_DEFAULT: f64 = 30.0;

// ─────────────────────────── helpers ───────────────────────────

fn weighting_fn(n: &Array1<f64>, n0: f64) -> Array1<f64> {
    n.mapv(|v| (2.0 * v / n0).tanh())
}

fn t_quantile(df: f64, level: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let alpha = 1.0 - level;
    let t = StudentsT::new(0.0, 1.0, df).expect("t-dist");
    t.inverse_cdf(1.0 - alpha / 2.0)
}

fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};
    if d1 <= 0.0 || d2 <= 0.0 || !x.is_finite() {
        return f64::NAN;
    }
    let f = FisherSnedecor::new(d1, d2).expect("F-dist");
    1.0 - f.cdf(x)
}

/// numpy-style flat copy from any 1D/2D readonly array passed from Python.
fn read_1d_or_2d(any: &Bound<'_, PyAny>) -> PyResult<Array1<f64>> {
    if let Ok(a) = any.extract::<PyReadonlyArray2<f64>>() {
        let v = a.as_array();
        return Ok(Array1::from_iter(v.iter().copied()));
    }
    if let Ok(a) = any.extract::<PyReadonlyArray1<f64>>() {
        return Ok(a.as_array().to_owned());
    }
    // Fallback: use numpy.asarray then reshape(-1)
    let py = any.py();
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("asarray", (any,))?;
    let arr = arr.call_method1("astype", ("float64",))?;
    let arr = arr.call_method1("reshape", ((-1i64,),))?;
    let pa: PyReadonlyArray1<f64> = arr.extract()?;
    Ok(pa.as_array().to_owned())
}

fn dict_set_param_ci<'py>(
    _py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    name: &str,
    _value: f64,
    lo: f64,
    hi: f64,
) -> PyResult<()> {
    dict.set_item(name, (lo, hi))
}

fn r2_score(y: &Array1<f64>, y_pred: &Array1<f64>, w: Option<&Array1<f64>>) -> f64 {
    let mean = match w {
        Some(w) => {
            let sw = w.sum();
            if sw == 0.0 { y.mean().unwrap_or(0.0) } else { (y * w).sum() / sw }
        }
        None => y.mean().unwrap_or(0.0),
    };
    let (ss_res, ss_tot) = match w {
        Some(w) => {
            let res: f64 = y
                .iter()
                .zip(y_pred.iter())
                .zip(w.iter())
                .map(|((yi, pi), wi)| wi * (yi - pi).powi(2))
                .sum();
            let tot: f64 = y
                .iter()
                .zip(w.iter())
                .map(|(yi, wi)| wi * (yi - mean).powi(2))
                .sum();
            (res, tot)
        }
        None => {
            let res: f64 = y
                .iter()
                .zip(y_pred.iter())
                .map(|(yi, pi)| (yi - pi).powi(2))
                .sum();
            let tot: f64 = y.iter().map(|yi| (yi - mean).powi(2)).sum();
            (res, tot)
        }
    };
    if ss_tot == 0.0 { 0.0 } else { 1.0 - ss_res / ss_tot }
}

// ─────────────────── callables returned in result dicts ───────────────────

#[pyclass(name = "_LinearCallable", module = "PyEvoMotion")]
pub struct LinearCallable {
    #[pyo3(get)]
    pub m: f64,
    #[pyo3(get)]
    pub b: Option<f64>,
}

#[pymethods]
impl LinearCallable {
    fn __call__<'py>(&self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();
        let np = py.import_bound("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;
        let scaled = arr.call_method1("__mul__", (self.m,))?;
        match self.b {
            Some(b) => scaled.call_method1("__add__", (b,)),
            None => Ok(scaled),
        }
    }

    fn __repr__(&self) -> String {
        match self.b {
            Some(b) => format!("LinearCallable(m={}, b={})", self.m, b),
            None => format!("LinearCallable(m={})", self.m),
        }
    }
}

#[pyclass(name = "_PowerLawCallable", module = "PyEvoMotion")]
pub struct PowerLawCallable {
    #[pyo3(get)]
    pub d: f64,
    #[pyo3(get)]
    pub alpha: f64,
}

#[pymethods]
impl PowerLawCallable {
    fn __call__<'py>(&self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();
        let np = py.import_bound("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;
        let powered = np.call_method1("power", (arr, self.alpha))?;
        powered.call_method1("__mul__", (self.d,))
    }

    fn __repr__(&self) -> String {
        format!("PowerLawCallable(d={}, alpha={})", self.d, self.alpha)
    }
}

// ─────────────────── LM problem for power-law fit ───────────────────
//
// Parameterise a as exp(t) so a is always positive (mirrors scipy's
// `bounds=([1e-10, -inf], [inf, inf])` lower-bound on the coefficient).

struct PowerLawProblem {
    x: Array1<f64>,
    y: Array1<f64>,
    sigma: Option<Array1<f64>>,
    p: Vector2<f64>, // [t_a, b], where a = exp(t_a)
}

impl LeastSquaresProblem<f64, Dyn, U2> for PowerLawProblem {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2>;

    fn set_params(&mut self, p: &Vector2<f64>) {
        self.p = *p;
    }

    fn params(&self) -> Vector2<f64> {
        self.p
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, nalgebra::U1, Self::ResidualStorage>> {
        let a = self.p[0].exp();
        let b = self.p[1];
        let n = self.x.len();
        let mut r = nalgebra::DVector::<f64>::zeros(n);
        for i in 0..n {
            let xi = self.x[i];
            let pred = if xi > 0.0 || b.fract() == 0.0 {
                a * xi.powf(b)
            } else {
                f64::NAN
            };
            let mut ri = self.y[i] - pred;
            if let Some(s) = &self.sigma {
                ri /= s[i].max(1e-300);
            }
            if !ri.is_finite() {
                ri = 0.0;
            }
            r[i] = ri;
        }
        Some(r)
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let a = self.p[0].exp();
        let b = self.p[1];
        let n = self.x.len();
        let mut j = nalgebra::OMatrix::<f64, Dyn, U2>::zeros(n);
        for i in 0..n {
            let xi = self.x[i];
            let xb = if xi > 0.0 { xi.powf(b) } else { 0.0 };
            // d residual / d t_a = -d (a*x^b) / d t_a = -a*x^b
            // d residual / d b   = -a*x^b * ln(x)
            let scale = match &self.sigma {
                Some(s) => 1.0 / s[i].max(1e-300),
                None => 1.0,
            };
            let mut dt_a = -a * xb * scale;
            let mut db = if xi > 0.0 { -a * xb * xi.ln() * scale } else { 0.0 };
            if !dt_a.is_finite() { dt_a = 0.0; }
            if !db.is_finite() { db = 0.0; }
            j[(i, 0)] = dt_a;
            j[(i, 1)] = db;
        }
        Some(j)
    }
}

// ─────────────────────── PyEvoMotionBase ───────────────────────

#[pyclass(subclass, name = "PyEvoMotionBase", module = "PyEvoMotion")]
pub struct PyEvoMotionBase;

#[pymethods]
impl PyEvoMotionBase {
    // Accept and ignore any args/kwargs so subclasses (which may forward
    // their own __init__ args via super().__init__) construct cleanly,
    // matching the no-op behavior of the original Python mixin.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    // count_prefixes ----------------------------------------------------
    #[staticmethod]
    fn count_prefixes(prefix: &str, mutations: Vec<String>) -> usize {
        mutations.iter().filter(|m| m.starts_with(prefix)).count()
    }

    // mutation_length_modification --------------------------------------
    #[staticmethod]
    fn mutation_length_modification(mutation: &str) -> PyResult<i64> {
        if mutation.starts_with('s') {
            return Ok(0);
        }
        let last = mutation.rsplit('_').next().unwrap_or("");
        let len = last.chars().count() as i64;
        if mutation.starts_with('i') {
            return Ok(len);
        }
        if mutation.starts_with('d') {
            return Ok(-len);
        }
        Err(PyValueError::new_err(format!(
            "Mutation not recognized: {}",
            mutation
        )))
    }

    // date_grouper (calls pandas) ---------------------------------------
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn date_grouper<'py>(
        py: Python<'py>,
        df: Bound<'py, PyAny>,
        DT: &str,
        origin: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import_bound("pandas")?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("key", "date")?;
        kwargs.set_item("axis", 0)?;
        kwargs.set_item("freq", DT)?;
        kwargs.set_item("origin", origin)?;
        let grouper = pd.call_method("Grouper", (), Some(&kwargs))?;
        df.call_method1("groupby", (grouper,))
    }

    // _invoke_method (Python passthrough) -------------------------------
    #[staticmethod]
    #[pyo3(signature = (instance, method, *args, **kwargs))]
    fn _invoke_method<'py>(
        instance: Bound<'py, PyAny>,
        method: &str,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match instance.getattr(method) {
            Ok(callable) => callable.call(args, kwargs),
            Err(_) => {
                let msg = format!("Method {} not found in {}", method, instance);
                println!("{}", msg);
                Ok(instance.py().None().into_bound(instance.py()))
            }
        }
    }

    // _remove_nan -------------------------------------------------------
    // Equivalent to pd.DataFrame({x,y,z}).dropna() and reshape(-1,1).
    #[staticmethod]
    pub fn _remove_nan<'py>(
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        z: Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let xa = read_1d_or_2d(&x)?;
        let ya = read_1d_or_2d(&y)?;
        let za = read_1d_or_2d(&z)?;
        let n = xa.len().min(ya.len()).min(za.len());
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        let mut zv = Vec::with_capacity(n);
        for i in 0..n {
            if xa[i].is_finite() && ya[i].is_finite() && za[i].is_finite() {
                xv.push(xa[i]);
                yv.push(ya[i]);
                zv.push(za[i]);
            }
        }
        // The Python version reshapes to (n, 1); we return 1D arrays. Callers in
        // base.py only use them via element access / iteration, which works.
        Ok((
            Array1::from(xv).into_pyarray_bound(py),
            Array1::from(yv).into_pyarray_bound(py),
            Array1::from(zv).into_pyarray_bound(py),
        ))
    }

    // _weighting_function -----------------------------------------------
    #[staticmethod]
    #[pyo3(signature = (n, n_0=N0_DEFAULT))]
    fn _weighting_function<'py>(
        py: Python<'py>,
        n: Bound<'py, PyAny>,
        n_0: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let arr = read_1d_or_2d(&n)?;
        Ok(weighting_fn(&arr, n_0).into_pyarray_bound(py))
    }

    // _compute_confidence_intervals -------------------------------------
    #[staticmethod]
    #[pyo3(signature = (parameters, standard_errors, degrees_of_freedom, confidence_level=0.95))]
    fn _compute_confidence_intervals<'py>(
        py: Python<'py>,
        parameters: Bound<'py, PyDict>,
        standard_errors: Bound<'py, PyDict>,
        degrees_of_freedom: i64,
        confidence_level: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let tval = t_quantile(degrees_of_freedom as f64, confidence_level);
        let out = PyDict::new_bound(py);
        for (k, v) in parameters.iter() {
            let name: String = k.extract()?;
            let p: f64 = v.extract()?;
            let se: f64 = standard_errors.get_item(&name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing SE for {}", name)))?
                .extract()?;
            let m = tval * se;
            dict_set_param_ci(py, &out, &name, p, p - m, p + m)?;
        }
        Ok(out)
    }

    // _power_law --------------------------------------------------------
    #[staticmethod]
    fn _power_law<'py>(
        x: Bound<'py, PyAny>,
        a: f64,
        b: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();
        let np = py.import_bound("numpy")?;
        let powered = np.call_method1("power", (x, b))?;
        powered.call_method1("__mul__", (a,))
    }

    // linear_regression -------------------------------------------------
    #[classmethod]
    #[pyo3(signature = (x, y, weights=None, fit_intercept=true, confidence_level=0.95))]
    pub fn linear_regression<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        weights: Option<Bound<'py, PyAny>>,
        fit_intercept: bool,
        confidence_level: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = cls;
        let xa = read_1d_or_2d(&x)?;
        let ya = read_1d_or_2d(&y)?;
        if xa.len() != ya.len() {
            return Err(PyValueError::new_err("x and y must have same length"));
        }
        let n = xa.len();
        let w_eff: Option<Array1<f64>> = match &weights {
            Some(w) => Some(weighting_fn(&read_1d_or_2d(w)?, N0_DEFAULT)),
            None => None,
        };
        let (m, b_opt, mse, sxx, x_mean) = wls_fit(&xa, &ya, w_eff.as_ref(), fit_intercept);

        let df = (n as i64) - if fit_intercept { 2 } else { 1 };
        let se_m = (mse / sxx.max(f64::MIN_POSITIVE)).sqrt();

        let model_dict = PyDict::new_bound(py);
        let params = PyDict::new_bound(py);
        params.set_item("m", m)?;
        if let Some(b) = b_opt {
            params.set_item("b", b)?;
        }

        let ci = PyDict::new_bound(py);
        let tval = t_quantile(df.max(1) as f64, confidence_level);
        ci.set_item("m", (m - tval * se_m, m + tval * se_m))?;
        if let Some(b) = b_opt {
            let se_b = (mse * (1.0 / n as f64 + x_mean.powi(2) / sxx.max(f64::MIN_POSITIVE))).sqrt();
            ci.set_item("b", (b - tval * se_b, b + tval * se_b))?;
        }

        let callable = Py::new(py, LinearCallable { m, b: b_opt })?;
        model_dict.set_item("model", callable)?;
        model_dict.set_item("parameters", params)?;
        model_dict.set_item("confidence_intervals", ci)?;
        model_dict.set_item(
            "expression",
            if fit_intercept { "mx + b" } else { "mx" },
        )?;
        model_dict.set_item("confidence_level", confidence_level)?;

        // r2
        let mut yp = Array1::<f64>::zeros(n);
        for i in 0..n {
            yp[i] = m * xa[i] + b_opt.unwrap_or(0.0);
        }
        let r2 = r2_score(&ya, &yp, w_eff.as_ref());
        model_dict.set_item("r2", r2)?;
        Ok(model_dict)
    }

    // power_law_fit -----------------------------------------------------
    #[classmethod]
    #[pyo3(signature = (x, y, weights=None, confidence_level=0.95))]
    fn power_law_fit<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        weights: Option<Bound<'py, PyAny>>,
        confidence_level: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = cls;
        let xa = read_1d_or_2d(&x)?;
        let ya = read_1d_or_2d(&y)?;
        let n = xa.len();
        let w_eff: Option<Array1<f64>> = match &weights {
            Some(w) => Some(weighting_fn(&read_1d_or_2d(w)?, N0_DEFAULT)),
            None => None,
        };
        // sigma = 1/sqrt(w)
        let sigma: Option<Array1<f64>> = w_eff
            .as_ref()
            .map(|w| w.mapv(|v| 1.0 / v.max(1e-300).sqrt()));

        // Initial guess via log-log linear regression on positive entries.
        let (mut a0, mut b0) = (1.0_f64, 1.0_f64);
        let mut xl: Vec<f64> = Vec::new();
        let mut yl: Vec<f64> = Vec::new();
        for i in 0..n {
            if xa[i] > 0.0 && ya[i] > 0.0 {
                xl.push(xa[i].ln());
                yl.push(ya[i].ln());
            }
        }
        if xl.len() > 1 {
            let mean_x = xl.iter().sum::<f64>() / xl.len() as f64;
            let mean_y = yl.iter().sum::<f64>() / yl.len() as f64;
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..xl.len() {
                num += (xl[i] - mean_x) * (yl[i] - mean_y);
                den += (xl[i] - mean_x).powi(2);
            }
            if den > 0.0 {
                let slope = num / den;
                let intercept = mean_y - slope * mean_x;
                b0 = slope;
                a0 = intercept.exp();
            }
        }
        let (mut popt_a, mut popt_b, mut cov_a, mut cov_b, mut converged) =
            run_lm(&xa, &ya, sigma.as_ref(), a0, b0);
        if !converged {
            // Retry with default initial point, like the Python version.
            let r = run_lm(&xa, &ya, sigma.as_ref(), 1.0, 1.0);
            popt_a = r.0;
            popt_b = r.1;
            cov_a = r.2;
            cov_b = r.3;
            converged = r.4;
        }
        if !converged {
            // Match the Python "could not converge" branch.
            popt_a = 0.0;
            popt_b = 0.0;
            cov_a = f64::INFINITY;
            cov_b = f64::INFINITY;
        }

        // CIs
        let df_free = (n as i64) - 2;
        let tval = t_quantile(df_free.max(1) as f64, confidence_level);
        let se_a = cov_a.sqrt();
        let se_b = cov_b.sqrt();

        let model_dict = PyDict::new_bound(py);
        let params = PyDict::new_bound(py);
        params.set_item("d", popt_a)?;
        params.set_item("alpha", popt_b)?;
        let ci = PyDict::new_bound(py);
        ci.set_item("d", (popt_a - tval * se_a, popt_a + tval * se_a))?;
        ci.set_item("alpha", (popt_b - tval * se_b, popt_b + tval * se_b))?;

        let callable = Py::new(
            py,
            PowerLawCallable {
                d: popt_a,
                alpha: popt_b,
            },
        )?;
        model_dict.set_item("model", callable)?;
        model_dict.set_item("parameters", params)?;
        model_dict.set_item("confidence_intervals", ci)?;
        model_dict.set_item("expression", "d*x^alpha")?;
        model_dict.set_item("confidence_level", confidence_level)?;

        // r2
        let mut yp = Array1::<f64>::zeros(n);
        for i in 0..n {
            yp[i] = popt_a * xa[i].powf(popt_b);
        }
        let r2 = r2_score(&ya, &yp, w_eff.as_ref());
        model_dict.set_item("r2", r2)?;
        Ok(model_dict)
    }

    // F_test ------------------------------------------------------------
    #[classmethod]
    #[pyo3(signature = (model1, model2, data, weights=None))]
    fn F_test<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        model1: Bound<'py, PyDict>,
        model2: Bound<'py, PyDict>,
        data: Bound<'py, PyAny>,
        weights: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(f64, f64)> {
        let _ = cls;
        let _ = py;
        let (rss1, rss2, p1, p2, n) = compute_rss(&model1, &model2, &data, weights.as_ref())?;
        if p1 >= p2 || n <= p2 {
            return Ok((f64::NAN, f64::NAN));
        }
        let f = ((rss1 - rss2) / (p2 - p1) as f64) / (rss2 / (n - p2) as f64);
        let p = f_sf(f, (p2 - p1) as f64, (n - p2) as f64);
        Ok((f, p))
    }

    // AIC ---------------------------------------------------------------
    #[classmethod]
    #[pyo3(signature = (model1, model2, data, weights=None))]
    fn AIC<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        model1: Bound<'py, PyDict>,
        model2: Bound<'py, PyDict>,
        data: Bound<'py, PyAny>,
        weights: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let _ = cls;
        let _ = py;
        let (mut rss1, mut rss2, k1, k2, n) =
            compute_rss(&model1, &model2, &data, weights.as_ref())?;
        if rss1 == 0.0 {
            rss1 = 1e-10;
        }
        if rss2 == 0.0 {
            rss2 = 1e-10;
        }
        let denom1 = (n as i64) - (k1 as i64) - 1;
        let denom2 = (n as i64) - (k2 as i64) - 1;
        let const_term = (n as f64) * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        let aicc1 = if denom1 <= 0 {
            f64::INFINITY
        } else {
            const_term
                + (n as f64) * (rss1 / n as f64).ln()
                + 2.0 * k1 as f64
                + (2.0 * k1 as f64 * (k1 as f64 + 1.0)) / denom1 as f64
        };
        let aicc2 = if denom2 <= 0 {
            f64::INFINITY
        } else {
            const_term
                + (n as f64) * (rss2 / n as f64).ln()
                + 2.0 * k2 as f64
                + (2.0 * k2 as f64 * (k2 as f64 + 1.0)) / denom2 as f64
        };
        let min = aicc1.min(aicc2);
        let d1 = aicc1 - min;
        let d2 = aicc2 - min;
        let r1 = if d1.is_finite() { (-0.5 * d1).exp() } else { 0.0 };
        let r2 = if d2.is_finite() { (-0.5 * d2).exp() } else { 0.0 };
        let denom = if r1 + r2 > 0.0 { r1 + r2 } else { 1.0 };
        let w1 = r1 / denom;
        let w2 = r2 / denom;
        Ok((aicc1, aicc2, d1, d2, w1, w2))
    }

    // adjust_model ------------------------------------------------------
    #[classmethod]
    #[pyo3(signature = (x, y, name=None, weights=None, confidence_level=0.95))]
    pub fn adjust_model<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        x: Bound<'py, PyAny>,
        y: Bound<'py, PyAny>,
        name: Option<&str>,
        weights: Option<Bound<'py, PyAny>>,
        confidence_level: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Run _remove_nan via cls (classmethod dispatch lets subclasses override).
        let zero = if let Some(w) = &weights {
            w.clone()
        } else {
            let np = py.import_bound("numpy")?;
            np.call_method1("zeros_like", (&x,))?
        };
        let cleaned = cls.call_method1("_remove_nan", (x.clone(), y.clone(), zero))?;
        let cleaned: (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>) = cleaned.extract()?;
        let (xc, yc, wc) = cleaned;
        let xlen: usize = xc.call_method0("__len__")?.extract()?;
        let ylen: usize = yc.call_method0("__len__")?.extract()?;
        if xlen <= 1 || ylen <= 1 {
            let empty_df = py.import_bound("pandas")?.getattr("DataFrame")?.call0()?;
            let msg = format!(
                "Dataset length after filtering is: x: {} elements; y: {} elements. Perhaps NaN appeared for certain entries.",
                xlen, ylen
            );
            cls.call_method1("_check_dataset_is_not_empty", (empty_df, msg))?;
        }

        // Linear (no intercept) and power law fits, dispatched via cls so subclasses can override.
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("weights", wc.clone())?;
        kwargs.set_item("fit_intercept", false)?;
        kwargs.set_item("confidence_level", confidence_level)?;
        let model1: Bound<'py, PyDict> = cls
            .call_method("linear_regression", (xc.clone(), yc.clone()), Some(&kwargs))?
            .extract()?;
        let kwargs2 = PyDict::new_bound(py);
        kwargs2.set_item("weights", wc.clone())?;
        kwargs2.set_item("confidence_level", confidence_level)?;
        let model2: Bound<'py, PyDict> = cls
            .call_method("power_law_fit", (xc.clone(), yc.clone()), Some(&kwargs2))?
            .extract()?;

        let (aic1, aic2, d1, d2, w1, w2): (f64, f64, f64, f64, f64, f64) = cls
            .call_method1("AIC", (model1.clone(), model2.clone(), yc.clone(), wc.clone()))?
            .extract()?;

        let selected_name = if aic1 <= aic2 { "linear" } else { "power_law" };

        let model1_with = clone_dict(py, &model1)?;
        model1_with.set_item("AIC", aic1)?;
        model1_with.set_item("delta_AIC", d1)?;
        model1_with.set_item("akaike_weight", w1)?;
        model1_with.set_item("confidence_level", confidence_level)?;

        let model2_with = clone_dict(py, &model2)?;
        model2_with.set_item("AIC", aic2)?;
        model2_with.set_item("delta_AIC", d2)?;
        model2_with.set_item("akaike_weight", w2)?;
        model2_with.set_item("confidence_level", confidence_level)?;

        let result = PyDict::new_bound(py);
        let selected = if aic1 <= aic2 {
            model1.clone()
        } else {
            model2.clone()
        };
        result.set_item("selected_model", selected)?;
        result.set_item("linear_model", model1_with)?;
        result.set_item("power_law_model", model2_with)?;
        let sel = PyDict::new_bound(py);
        sel.set_item("selected", selected_name)?;
        sel.set_item("linear_AIC", aic1)?;
        sel.set_item("power_law_AIC", aic2)?;
        sel.set_item("delta_AIC_linear", d1)?;
        sel.set_item("delta_AIC_power_law", d2)?;
        sel.set_item("akaike_weight_linear", w1)?;
        sel.set_item("akaike_weight_power_law", w2)?;
        result.set_item("model_selection", sel)?;

        if let Some(name) = name {
            let wrap = PyDict::new_bound(py);
            wrap.set_item(name, result)?;
            Ok(wrap)
        } else {
            Ok(result)
        }
    }

    // plot_single_data_and_model (matplotlib pass-through) --------------
    #[staticmethod]
    #[pyo3(signature = (data_x, data_y, data_ylabel, model, model_label, data_xlabel_units, ax, dt_ratio, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn plot_single_data_and_model<'py>(
        py: Python<'py>,
        data_x: Bound<'py, PyAny>,
        data_y: Bound<'py, PyAny>,
        data_ylabel: &str,
        model: Bound<'py, PyAny>,
        model_label: &str,
        data_xlabel_units: &str,
        ax: Bound<'py, PyAny>,
        dt_ratio: f64,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let line_kwargs = PyDict::new_bound(py);
        line_kwargs.set_item("linestyle", py.None())?;
        line_kwargs.set_item("color", "#1f77b4")?;
        let point_kwargs = PyDict::new_bound(py);
        point_kwargs.set_item("color", "#1f77b4")?;
        if let Some(kw) = kwargs {
            for (k, v) in kw.iter() {
                let key: String = k.extract()?;
                if let Some((flag, kk)) = key.split_once('_') {
                    if flag == "line" && line_kwargs.contains(kk)? {
                        line_kwargs.set_item(kk, &v)?;
                    }
                    if flag == "point" && point_kwargs.contains(kk)? {
                        point_kwargs.set_item(kk, &v)?;
                    }
                }
            }
        }
        let np = py.import_bound("numpy")?;
        let xnum = data_x.call_method0("to_numpy")?;
        let xnum = xnum.call_method1("__mul__", (dt_ratio,))?;
        ax.call_method("scatter", (&xnum, &data_y), Some(&point_kwargs))?;
        let ymodel = model.call1((xnum.clone(),))?;
        let line_kwargs2 = PyDict::new_bound(py);
        for (k, v) in line_kwargs.iter() {
            line_kwargs2.set_item(k, v)?;
        }
        line_kwargs2.set_item("label", model_label)?;
        ax.call_method("plot", (&xnum, ymodel), Some(&line_kwargs2))?;
        ax.call_method1("set_ylabel", (data_ylabel,))?;
        ax.call_method1("set_xlabel", (format!("time ({})", data_xlabel_units),))?;
        ax.call_method0("legend")?;
        let _ = np;
        Ok(())
    }

    // _check_dataset_is_not_empty ---------------------------------------
    #[staticmethod]
    fn _check_dataset_is_not_empty<'py>(df: Bound<'py, PyAny>, msg: &str) -> PyResult<()> {
        let empty: bool = df.getattr("empty")?.extract()?;
        if empty {
            return Err(PyValueError::new_err(format!(
                "The dataset is (almost) empty at this point of the analysis.\n{}",
                msg
            )));
        }
        Ok(())
    }

    // _get_time_ratio ---------------------------------------------------
    #[staticmethod]
    #[pyo3(signature = (dt, reference="7D"))]
    fn _get_time_ratio<'py>(
        py: Python<'py>,
        dt: Bound<'py, PyAny>,
        reference: &str,
    ) -> PyResult<f64> {
        let pd = py.import_bound("pandas")?;
        let a = pd.call_method1("Timedelta", (dt,))?;
        let b = pd.call_method1("Timedelta", (reference,))?;
        a.call_method1("__truediv__", (b,))?.extract()
    }

    // _verify_dt --------------------------------------------------------
    #[classmethod]
    fn _verify_dt<'py>(cls: &Bound<'py, PyType>, dt: Bound<'py, PyAny>) -> PyResult<()> {
        let one_day = pyo3::types::PyString::new_bound(dt.py(), "1D");
        let r: f64 = cls
            .call_method1("_get_time_ratio", (dt.clone(), one_day))?
            .extract()?;
        if r <= 1.0 {
            return Err(PyValueError::new_err(format!(
                "Time window must be greater than 1 day. Got {}",
                dt.str()?.to_string_lossy()
            )));
        }
        Ok(())
    }
}

// ─────────────── helpers used by the impl above ───────────────

fn wls_fit(
    x: &Array1<f64>,
    y: &Array1<f64>,
    w: Option<&Array1<f64>>,
    fit_intercept: bool,
) -> (f64, Option<f64>, f64, f64, f64) {
    let n = x.len() as f64;
    let (sw, swx, swy, swxx, swxy) = match w {
        Some(w) => {
            let sw: f64 = w.sum();
            let swx: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi).sum();
            let swy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| wi * yi).sum();
            let swxx: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi * xi).sum();
            let swxy: f64 = x
                .iter()
                .zip(y.iter())
                .zip(w.iter())
                .map(|((xi, yi), wi)| wi * xi * yi)
                .sum();
            (sw, swx, swy, swxx, swxy)
        }
        None => {
            let sx: f64 = x.sum();
            let sy: f64 = y.sum();
            let sxx: f64 = x.iter().map(|v| v * v).sum();
            let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            (n, sx, sy, sxx, sxy)
        }
    };
    // For SE / sxx_centered the Python original uses the *unweighted* mean of x
    // and the *unweighted* sxx, even when weights are present. Match that.
    let x_mean_unw = x.mean().unwrap_or(0.0);
    let sxx_centered_unw: f64 = x.iter().map(|v| (v - x_mean_unw).powi(2)).sum();

    if fit_intercept {
        let det = sw * swxx - swx * swx;
        let m = if det != 0.0 { (sw * swxy - swx * swy) / det } else { 0.0 };
        let b = if sw > 0.0 { (swy - m * swx) / sw } else { 0.0 };
        let mut rss = 0.0;
        for i in 0..x.len() {
            let r = y[i] - (m * x[i] + b);
            let wi = w.map(|ww| ww[i]).unwrap_or(1.0);
            rss += wi * r * r;
        }
        let dofs = if w.is_some() { sw - 2.0 } else { n - 2.0 };
        let mse = if dofs > 0.0 { rss / dofs } else { 0.0 };
        (m, Some(b), mse, sxx_centered_unw.max(f64::MIN_POSITIVE), x_mean_unw)
    } else {
        let m = if swxx > 0.0 { swxy / swxx } else { 0.0 };
        let mut rss = 0.0;
        for i in 0..x.len() {
            let r = y[i] - m * x[i];
            let wi = w.map(|ww| ww[i]).unwrap_or(1.0);
            rss += wi * r * r;
        }
        let dofs = if w.is_some() { sw - 1.0 } else { n - 1.0 };
        let mse = if dofs > 0.0 { rss / dofs } else { 0.0 };
        (m, None, mse, sxx_centered_unw.max(f64::MIN_POSITIVE), x_mean_unw)
    }
}

fn run_lm(
    x: &Array1<f64>,
    y: &Array1<f64>,
    sigma: Option<&Array1<f64>>,
    a0: f64,
    b0: f64,
) -> (f64, f64, f64, f64, bool) {
    let p0 = Vector2::new(a0.max(1e-12).ln(), b0);
    let problem = PowerLawProblem {
        x: x.clone(),
        y: y.clone(),
        sigma: sigma.cloned(),
        p: p0,
    };
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);
    let p = problem.params();
    let a = p[0].exp();
    let b = p[1];
    let converged = report.termination.was_successful();

    // Approximate covariance via Jacobian^T * J inversion at the optimum.
    let (cov_a, cov_b) = if converged {
        let j = match LeastSquaresProblem::jacobian(&problem) {
            Some(j) => j,
            None => return (a, b, f64::INFINITY, f64::INFINITY, false),
        };
        let r = match LeastSquaresProblem::residuals(&problem) {
            Some(r) => r,
            None => return (a, b, f64::INFINITY, f64::INFINITY, false),
        };
        let n = x.len();
        let dof = (n as i64 - 2).max(1) as f64;
        let rss: f64 = r.iter().map(|v| v * v).sum();
        let sigma2 = rss / dof;
        let jtj = j.transpose() * j;
        let cov = match jtj.try_inverse() {
            Some(m) => m * sigma2,
            None => return (a, b, f64::INFINITY, f64::INFINITY, false),
        };
        // delta method: a = exp(t_a) → da/dt_a = a, so var(a) = a^2 * cov[0,0]
        let var_t_a = cov[(0, 0)];
        let var_b = cov[(1, 1)];
        (a * a * var_t_a, var_b)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    (a, b, cov_a, cov_b, converged)
}

fn compute_rss<'py>(
    model1: &Bound<'py, PyDict>,
    model2: &Bound<'py, PyDict>,
    data: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
) -> PyResult<(f64, f64, usize, usize, usize)> {
    let py = data.py();
    let arr = read_1d_or_2d(data)?;
    let n = arr.len();
    let p1: usize = model1
        .get_item("parameters")?
        .ok_or_else(|| PyValueError::new_err("model1 missing parameters"))?
        .call_method0("__len__")?
        .extract()?;
    let p2: usize = model2
        .get_item("parameters")?
        .ok_or_else(|| PyValueError::new_err("model2 missing parameters"))?
        .call_method0("__len__")?
        .extract()?;

    let w_eff: Array1<f64> = match weights {
        Some(w) => weighting_fn(&read_1d_or_2d(w)?, N0_DEFAULT),
        None => Array1::ones(n),
    };

    let m1 = model1
        .get_item("model")?
        .ok_or_else(|| PyValueError::new_err("model1 missing model callable"))?;
    let m2 = model2
        .get_item("model")?
        .ok_or_else(|| PyValueError::new_err("model2 missing model callable"))?;

    let xs = (0..n).map(|i| i as f64).collect::<Vec<_>>();
    let xs_arr = Array1::from(xs.clone()).into_pyarray_bound(py);

    let y1 = m1.call1((xs_arr.clone(),))?;
    let y2 = m2.call1((xs_arr,))?;
    let y1 = read_1d_or_2d(&y1)?;
    let y2 = read_1d_or_2d(&y2)?;

    let mut rss1 = 0.0;
    let mut rss2 = 0.0;
    for i in 0..n {
        let r1 = arr[i] - y1[i];
        let r2 = arr[i] - y2[i];
        if r1.is_finite() && r2.is_finite() {
            rss1 += w_eff[i] * r1 * r1;
            rss2 += w_eff[i] * r2 * r2;
        }
    }
    Ok((rss1, rss2, p1, p2, n))
}

fn clone_dict<'py>(py: Python<'py>, src: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let dst = PyDict::new_bound(py);
    for (k, v) in src.iter() {
        dst.set_item(k, v)?;
    }
    Ok(dst)
}
