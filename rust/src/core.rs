use std::os::raw::c_int;

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple, PyType};

use crate::base::{self, PyEvoMotionBase};
use crate::parser::PyEvoMotionParser;

const MUTATION_TYPES: &[&str] = &["substitutions", "indels"];

// ─────────────────────── helpers ───────────────────────

fn mutation_type_switch(mutation_kind: &str) -> PyResult<Vec<&'static str>> {
    match mutation_kind {
        "all" => Ok(vec!["substitutions", "indels", "mutations"]),
        "total" => Ok(vec!["mutations"]),
        "substitutions" => Ok(vec!["substitutions"]),
        "indels" => Ok(vec!["indels"]),
        other => Err(PyValueError::new_err(format!(
            "Mutation kind \"{}\" not recognized. It has to be one of all, total, substitutions, indels",
            other
        ))),
    }
}

/// Apply scaling correction in place: divides slope/coefficient by dt_ratio
/// (and dt_ratio^alpha for the power-law term), updating both the
/// `parameters` dict and `confidence_intervals` if present, and rebuilds
/// the `model` callable so it reflects the scaled parameters.
fn apply_scaling_correction<'py>(
    py: Python<'py>,
    dt_ratio: f64,
    model: &Bound<'py, PyDict>,
) -> PyResult<()> {
    let expr: String = model
        .get_item("expression")?
        .ok_or_else(|| PyValueError::new_err("model dict missing 'expression'"))?
        .extract()?;

    let params: Bound<'py, PyDict> = model
        .get_item("parameters")?
        .ok_or_else(|| PyValueError::new_err("model dict missing 'parameters'"))?
        .extract()?;
    let cis: Option<Bound<'py, PyDict>> = model
        .get_item("confidence_intervals")?
        .map(|v| v.extract())
        .transpose()?;

    match expr.as_str() {
        "mx + b" | "mx" => {
            let m: f64 = params.get_item("m")?.unwrap().extract()?;
            let m_scaled = m / dt_ratio;
            params.set_item("m", m_scaled)?;
            let b_opt: Option<f64> = if expr == "mx + b" {
                Some(params.get_item("b")?.unwrap().extract()?)
            } else {
                None
            };
            let callable = Py::new(
                py,
                base::LinearCallable {
                    m: m_scaled,
                    b: b_opt,
                },
            )?;
            model.set_item("model", callable)?;
            if let Some(ci) = cis {
                if let Some(t) = ci.get_item("m")? {
                    let (lo, hi): (f64, f64) = t.extract()?;
                    ci.set_item("m", (lo / dt_ratio, hi / dt_ratio))?;
                }
            }
        }
        "d*x^alpha" => {
            let d: f64 = params.get_item("d")?.unwrap().extract()?;
            let alpha: f64 = params.get_item("alpha")?.unwrap().extract()?;
            let d_scaled = d / dt_ratio.powf(alpha);
            params.set_item("d", d_scaled)?;
            let callable = Py::new(
                py,
                base::PowerLawCallable {
                    d: d_scaled,
                    alpha,
                },
            )?;
            model.set_item("model", callable)?;
            if let Some(ci) = cis {
                if let Some(t) = ci.get_item("d")? {
                    let (lo, hi): (f64, f64) = t.extract()?;
                    let scale = dt_ratio.powf(alpha);
                    ci.set_item("d", (lo / scale, hi / scale))?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn pdict_subset<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyDict>,
    keys: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    let dst = PyDict::new_bound(py);
    for k in keys {
        if let Some(v) = src.get_item(k)? {
            dst.set_item(k, v)?;
        }
    }
    Ok(dst)
}

// ─────────────────────── pyclass ───────────────────────

// _PyEvoMotionCore now extends PyEvoMotionParser (which extends
// PyEvoMotionBase), collapsing the former Python multi-inheritance
// (PyEvoMotion(_PyEvoMotionCore, PyEvoMotionParser)) into a single
// inheritance chain so the public PyEvoMotion class can live in Rust.
// The MRO is unchanged: PyEvoMotion → Core → Parser → Base.
#[pyclass(subclass, extends = PyEvoMotionParser, name = "_PyEvoMotionCore", module = "PyEvoMotion")]
pub struct PyEvoMotionCore;

#[pymethods]
impl PyEvoMotionCore {
    // Cooperative no-op constructor: builds the full layout chain so a
    // bare _PyEvoMotionCore(...) still instantiates cleanly.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyEvoMotionBase)
            .add_subclass(PyEvoMotionParser)
            .add_subclass(Self)
    }

    // _mutation_type_switch ---------------------------------------------
    #[classmethod]
    fn _mutation_type_switch<'py>(
        _cls: &Bound<'py, PyType>,
        mutation_kind: &str,
    ) -> PyResult<Vec<String>> {
        Ok(mutation_type_switch(mutation_kind)?
            .into_iter()
            .map(String::from)
            .collect())
    }

    // count_mutation_types ----------------------------------------------
    fn count_mutation_types<'py>(slf: &Bound<'py, Self>) -> PyResult<()> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;
        let data = slf.getattr("data")?;
        let mut_instr_series = data.call_method1("__getitem__", ("mutation instructions",))?;
        let mut_instr: Vec<Vec<String>> = mut_instr_series.call_method0("tolist")?.extract()?;

        for ty in MUTATION_TYPES.iter().chain(["insertions", "deletions"].iter()) {
            let prefix = &ty[..1];
            let counts: Vec<usize> = mut_instr
                .iter()
                .map(|m| m.iter().filter(|s| s.starts_with(prefix)).count())
                .collect();
            let series = pd.call_method1("Series", (counts,))?;
            data.call_method1("__setitem__", (format!("number of {}", ty), series))?;
        }

        // indels = ins + del (overrides the prefix-based count above, which
        // would only have caught entries actually starting with 'i').
        let ins = data.call_method1("__getitem__", ("number of insertions",))?;
        let del_ = data.call_method1("__getitem__", ("number of deletions",))?;
        let combined = ins.call_method1("__add__", (del_,))?;
        data.call_method1("__setitem__", ("number of indels", combined))?;

        // mutations = len(mutation instructions)
        let lens: Vec<usize> = mut_instr.iter().map(|m| m.len()).collect();
        let series = pd.call_method1("Series", (lens,))?;
        data.call_method1("__setitem__", ("number of mutations", series))?;

        Ok(())
    }

    // get_lengths -------------------------------------------------------
    fn get_lengths<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;
        let data = slf.getattr("data")?;
        let mut_instr_series = data.call_method1("__getitem__", ("mutation instructions",))?;
        let mut_instr: Vec<Vec<String>> = mut_instr_series.call_method0("tolist")?.extract()?;
        let reference = slf.getattr("reference")?;
        let ref_seq = reference.getattr("seq")?;
        let ref_len: i64 = ref_seq.call_method0("__len__")?.extract()?;

        let lengths: Vec<i64> = mut_instr
            .iter()
            .map(|muts| {
                muts.iter()
                    .map(|m| {
                        if m.starts_with('s') {
                            0i64
                        } else if let Some(last) = m.rsplit('_').next() {
                            let n = last.chars().count() as i64;
                            if m.starts_with('i') {
                                n
                            } else if m.starts_with('d') {
                                -n
                            } else {
                                0
                            }
                        } else {
                            0
                        }
                    })
                    .sum::<i64>()
                    + ref_len
            })
            .collect();

        Ok(pd.call_method1("Series", (lengths,))?)
    }

    // length_filter (preserves the no-op bug from the Python original) ---
    #[pyo3(signature = (length, how="gt"))]
    fn length_filter<'py>(slf: &Bound<'py, Self>, length: i64, how: &str) -> PyResult<()> {
        let lengths = slf.call_method0("get_lengths")?;
        let _mask = match how {
            "gt" => lengths.call_method1("__gt__", (length,))?,
            "lt" => lengths.call_method1("__lt__", (length,))?,
            "eq" => lengths.call_method1("__eq__", (length,))?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Filter \"{}\" not recognized",
                    other
                )))
            }
        };
        let data = slf.getattr("data")?;
        let _ = data.call_method1("__getitem__", (_mask,))?;
        let kw = PyDict::new_bound(slf.py());
        kw.set_item("drop", true)?;
        kw.set_item("inplace", true)?;
        data.call_method("reset_index", (), Some(&kw))?;
        Ok(())
    }

    // n_filter (also preserves the no-op bug) ---------------------------
    #[pyo3(signature = (threshold=0.01, how="lt"))]
    fn n_filter<'py>(slf: &Bound<'py, Self>, threshold: f64, how: &str) -> PyResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PyValueError::new_err("Threshold must be between 0 and 1"));
        }
        let py = slf.py();
        let data = slf.getattr("data")?;
        let n_count = data.call_method1("__getitem__", ("N count",))?;
        let lengths = slf.call_method0("get_lengths")?;
        let n_freq = n_count.call_method1("__truediv__", (lengths,))?;
        let _mask = match how {
            "gt" => n_freq.call_method1("__gt__", (threshold,))?,
            "lt" => n_freq.call_method1("__lt__", (threshold,))?,
            "eq" => n_freq.call_method1("__eq__", (threshold,))?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Filter \"{}\" not recognized",
                    other
                )))
            }
        };
        let _ = data.call_method1("__getitem__", (_mask,))?;
        let kw = PyDict::new_bound(py);
        kw.set_item("drop", true)?;
        kw.set_item("inplace", true)?;
        data.call_method("reset_index", (), Some(&kw))?;
        Ok(())
    }

    // compute_stats -----------------------------------------------------
    #[pyo3(signature = (DT, origin, mutation_kind="all"))]
    #[allow(non_snake_case)]
    fn compute_stats<'py>(
        slf: &Bound<'py, Self>,
        DT: &str,
        origin: Bound<'py, PyAny>,
        mutation_kind: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;
        let data_orig = slf.getattr("data")?;
        let mut data = data_orig.call_method0("copy")?;

        // Edge case: duplicate first row if alone on origin.
        let iloc = data.getattr("iloc")?;
        let first_row = iloc.call_method1("__getitem__", (0i64,))?;
        let first_date = first_row.call_method1("__getitem__", ("date",))?;
        let same_origin: bool = first_date.eq(&origin)?;
        let dates_eq_origin = data
            .call_method1("__getitem__", ("date",))?
            .call_method1("__eq__", (&origin,))?;
        let count_at_origin: i64 = data
            .call_method1("__getitem__", (dates_eq_origin,))?
            .call_method0("__len__")?
            .extract()?;
        if same_origin && count_at_origin == 1 {
            let dup = pd.call_method1("DataFrame", (PyList::new_bound(py, [first_row]),))?;
            let kw = PyDict::new_bound(py);
            kw.set_item("ignore_index", true)?;
            data = pd.call_method(
                "concat",
                (PyList::new_bound(py, [data.clone(), dup]),),
                Some(&kw),
            )?;
            let kw = PyDict::new_bound(py);
            kw.set_item("by", "date")?;
            kw.set_item("inplace", true)?;
            data.call_method("sort_values", (), Some(&kw))?;
            let kw = PyDict::new_bound(py);
            kw.set_item("drop", true)?;
            kw.set_item("inplace", true)?;
            data.call_method("reset_index", (), Some(&kw))?;
        }

        // Group, then keep only weeks with >=2 observations, then re-group.
        let grouped = base::PyEvoMotionBase::date_grouper(py, data, DT, origin.clone())?;
        let filter_lambda = py.eval_bound("lambda x: len(x) >= 2", None, None)?;
        let filtered = grouped.call_method1("filter", (filter_lambda,))?;
        let len_filtered: i64 = filtered.call_method0("__len__")?.extract()?;
        if len_filtered == 0 {
            return Err(PyValueError::new_err(
                "No groups with at least 2 observations. Consider widening the time interval.",
            ));
        }
        let grouped = base::PyEvoMotionBase::date_grouper(py, filtered, DT, origin)?;

        let kinds = mutation_type_switch(mutation_kind)?;
        let levels: Vec<String> = kinds.iter().map(|k| format!("number of {}", k)).collect();

        let mut frames: Vec<Bound<'py, PyAny>> = Vec::new();
        for method in &["mean", "var", "size"] {
            let stat: Bound<'py, PyAny> = if *method == "size" {
                let s = grouped.call_method0("size")?;
                pd.call_method1("DataFrame", (s,))?
            } else {
                let g_levels = grouped.call_method1("__getitem__", (levels.clone(),))?;
                let agg = g_levels.call_method0(*method)?;
                pd.call_method1("DataFrame", (agg,))?
            };
            let rename_lambda = if *method == "size" {
                py.eval_bound("lambda col: 'size'", None, None)?
            } else {
                let code = format!("lambda col: '{} ' + col", method);
                py.eval_bound(&code, None, None)?
            };
            let kw = PyDict::new_bound(py);
            kw.set_item("columns", rename_lambda)?;
            let renamed = stat.call_method("rename", (), Some(&kw))?;
            frames.push(renamed);
        }
        let kw = PyDict::new_bound(py);
        kw.set_item("axis", 1)?;
        let combined = pd.call_method("concat", (PyList::new_bound(py, frames),), Some(&kw))?;
        let kw = PyDict::new_bound(py);
        kw.set_item("level", PyList::new_bound(py, ["date"]))?;
        Ok(combined.call_method("reset_index", (), Some(&kw))?)
    }

    // analysis ----------------------------------------------------------
    //
    // Math (linear_regression, adjust_model, _remove_nan) is invoked
    // directly via the Rust associated functions on PyEvoMotionBase, not
    // through Python attribute lookup — that's the "use base directly
    // from Rust" aspect.
    #[pyo3(signature = (length, show=false, mutation_kind="all", export_plots_filename=None, confidence_level=0.95))]
    fn analysis<'py>(
        slf: &Bound<'py, Self>,
        length: i64,
        show: bool,
        mutation_kind: &str,
        export_plots_filename: Option<&str>,
        confidence_level: f64,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;

        // Apply N-count and length filters (these are on `self`/subclass,
        // so route via Python attribute lookup to honor any overrides).
        slf.call_method0("n_filter")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("length", length)?;
        slf.call_method("length_filter", (), Some(&kw))?;

        // Compute stats.
        let dt: String = slf.getattr("dt")?.extract()?;
        let origin = slf.getattr("origin")?;
        let stats = slf.call_method1("compute_stats", (&dt, origin, mutation_kind))?;
        let weights = stats.call_method1("__getitem__", ("size",))?;

        let regs = PyDict::new_bound(py);

        let columns: Vec<String> = stats
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        let n_cols = columns.len();
        if n_cols < 2 {
            return Err(PyValueError::new_err(
                "compute_stats returned no usable columns",
            ));
        }
        let middle_cols = &columns[1..n_cols - 1];

        let cls_base = py.get_type_bound::<PyEvoMotionBase>();

        for col in middle_cols {
            let col_data = stats.call_method1("__getitem__", (col,))?;
            let stats_index = stats.getattr("index")?;

            if col.starts_with("mean") {
                let cleaned = base::PyEvoMotionBase::_remove_nan(
                    py,
                    stats_index,
                    col_data,
                    weights.clone(),
                )?;
                let (xc, yc, wc) = cleaned;
                let result = base::PyEvoMotionBase::linear_regression(
                    &cls_base,
                    py,
                    xc.into_any(),
                    yc.into_any(),
                    Some(wc.into_any()),
                    true,
                    confidence_level,
                )?;
                regs.set_item(format!("{} model", col), result)?;
            } else if col.starts_with("var") {
                let var_min = col_data.call_method0("min")?;
                let var_scaled = col_data.call_method1("__sub__", (var_min,))?;
                let weights_np = weights
                    .call_method0("to_numpy")?
                    .call_method0("flatten")?;
                let model_name = format!("scaled {} model", col);
                let adjust_out = base::PyEvoMotionBase::adjust_model(
                    &cls_base,
                    py,
                    stats_index,
                    var_scaled,
                    Some(&model_name),
                    Some(weights_np),
                    confidence_level,
                )?;
                let inner: Bound<'py, PyDict> = adjust_out
                    .get_item(&model_name)?
                    .ok_or_else(|| PyValueError::new_err("adjust_model missing inner key"))?
                    .extract()?;
                let selected = inner
                    .get_item("selected_model")?
                    .ok_or_else(|| PyValueError::new_err("adjust_model missing selected_model"))?;
                regs.set_item(&model_name, selected)?;
                regs.set_item(format!("{}_full_results", model_name), inner)?;
            }
        }

        // Apply scaling correction to the top-level (non-_full_results) entries.
        let dt_ratio: f64 = slf.getattr("dt_ratio")?.extract()?;
        let keys: Vec<String> = regs.keys().extract()?;
        for k in &keys {
            if k.ends_with("_full_results") {
                continue;
            }
            let v = regs.get_item(k)?.unwrap();
            let v_dict: Bound<'py, PyDict> = v.extract()?;
            apply_scaling_correction(py, dt_ratio, &v_dict)?;
        }
        for k in &keys {
            if !k.ends_with("_full_results") {
                continue;
            }
            let v_dict: Bound<'py, PyDict> = regs.get_item(k)?.unwrap().extract()?;
            for sub in &["selected_model", "linear_model", "power_law_model"] {
                if let Some(m) = v_dict.get_item(sub)? {
                    let m_dict: Bound<'py, PyDict> = m.extract()?;
                    apply_scaling_correction(py, dt_ratio, &m_dict)?;
                }
            }
        }

        // _sets: distinct mutation-type tags (everything after the first word
        // of each middle column name).
        let mut sets: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for c in middle_cols {
            let mut parts = c.split_whitespace();
            parts.next();
            sets.insert(parts.collect::<Vec<_>>().join(" "));
        }

        // stats["dt_idx"] = (stats["date"] - stats["date"].min()) / pd.Timedelta("7D")
        let stats_date = stats.call_method1("__getitem__", ("date",))?;
        let date_min = stats_date.call_method0("min")?;
        let date_diff = stats_date.call_method1("__sub__", (date_min,))?;
        let week_td = pd.call_method1("Timedelta", ("7D",))?;
        let dt_idx = date_diff.call_method1("__truediv__", (week_td,))?;
        stats.call_method1("__setitem__", ("dt_idx", dt_idx))?;

        if show || export_plots_filename.is_some() {
            let cls_self = slf.getattr("__class__")?;
            let pdf_handle = if let Some(filename) = export_plots_filename {
                let pdf_module = py.import_bound("matplotlib.backends.backend_pdf")?;
                Some(pdf_module.call_method1("PdfPages", (format!("{}.pdf", filename),))?)
            } else {
                None
            };

            for ts in &sets {
                let cols = vec![
                    "date".to_string(),
                    "dt_idx".to_string(),
                    format!("mean {}", ts),
                    format!("var {}", ts),
                ];
                let stats_subset = stats.call_method1("__getitem__", (cols,))?;
                let regs_subset = pdict_subset(
                    py,
                    &regs,
                    &[
                        format!("mean {} model", ts),
                        format!("scaled var {} model", ts),
                    ],
                )?;
                if show {
                    cls_self.call_method1(
                        "plot_results",
                        (stats_subset.clone(), regs_subset.clone(), "wk", dt_ratio),
                    )?;
                }
                if let Some(ref pdf) = pdf_handle {
                    cls_self.call_method1(
                        "export_plot_results",
                        (stats_subset, regs_subset, "wk", dt_ratio, pdf),
                    )?;
                }
            }
            if let Some(pdf) = pdf_handle {
                pdf.call_method0("close")?;
            }
        }

        Ok((stats, regs))
    }

    // _apply_scaling_correction_to_model --------------------------------
    fn _apply_scaling_correction_to_model<'py>(
        slf: &Bound<'py, Self>,
        model: Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let py = slf.py();
        let dt_ratio: f64 = slf.getattr("dt_ratio")?.extract()?;
        apply_scaling_correction(py, dt_ratio, &model)
    }

    // plot_results ------------------------------------------------------
    #[classmethod]
    fn plot_results<'py>(
        cls: &Bound<'py, PyType>,
        stats: Bound<'py, PyAny>,
        regs: Bound<'py, PyDict>,
        data_xlabel_units: &str,
        dt_ratio: f64,
    ) -> PyResult<()> {
        let py = stats.py();
        let plt = py.import_bound("matplotlib.pyplot")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("figsize", (10i64, 10i64))?;
        let fig_axes = plt.call_method("subplots", (3i64, 1i64), Some(&kw))?;
        let ax = fig_axes.get_item(1)?;

        let columns: Vec<String> = stats
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        let mean_col = &columns[2];
        let var_col = &columns[3];

        let pick_model = |prefix: &str| -> PyResult<Bound<'py, PyDict>> {
            for (k, v) in regs.iter() {
                let ks: String = k.extract()?;
                if ks.starts_with(prefix) {
                    return v.extract();
                }
            }
            Err(PyValueError::new_err(format!(
                "regs missing entry starting with '{}'",
                prefix
            )))
        };

        let mean_model = pick_model("mean")?;
        let var_model = pick_model("scaled var")?;
        let mean_data = stats.call_method1("__getitem__", (mean_col,))?;
        let var_data = stats.call_method1("__getitem__", (var_col,))?;

        let kw_mean = PyDict::new_bound(py);
        kw_mean.set_item("dt_ratio", dt_ratio)?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                mean_data.clone(),
                mean_data.getattr("name")?,
                mean_model.get_item("model")?.unwrap(),
                format!(
                    "$r^2$: {:.2}",
                    mean_model.get_item("r2")?.unwrap().extract::<f64>()?
                ),
                data_xlabel_units,
                ax.get_item(0)?,
            ),
            Some(&kw_mean),
        )?;

        let kw_var = PyDict::new_bound(py);
        kw_var.set_item("dt_ratio", dt_ratio)?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                var_data.clone(),
                var_data.getattr("name")?,
                var_model.get_item("model")?.unwrap(),
                format!(
                    "$r^2$: {:.2}",
                    var_model.get_item("r2")?.unwrap().extract::<f64>()?
                ),
                data_xlabel_units,
                ax.get_item(1)?,
            ),
            Some(&kw_var),
        )?;

        let mean_name: String = mean_data.getattr("name")?.extract()?;
        let label = mean_name
            .split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join(" ");
        let dispersion = mean_data.call_method1("__truediv__", (var_data,))?;
        let const_one = py.eval_bound("lambda x: [1]*len(x)", None, None)?;
        let kw_disp = PyDict::new_bound(py);
        kw_disp.set_item("dt_ratio", dt_ratio)?;
        kw_disp.set_item("line_linestyle", "--")?;
        kw_disp.set_item("line_color", "black")?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                dispersion,
                format!("dispersion index of {}", label),
                const_one,
                "Poissonian regime",
                data_xlabel_units,
                ax.get_item(2)?,
            ),
            Some(&kw_disp),
        )?;

        plt.call_method0("tight_layout")?;
        plt.call_method0("show")?;
        Ok(())
    }

    // export_plot_results -----------------------------------------------
    #[classmethod]
    #[pyo3(signature = (stats, regs, data_xlabel_units, dt_ratio, output_ptr=None))]
    fn export_plot_results<'py>(
        cls: &Bound<'py, PyType>,
        stats: Bound<'py, PyAny>,
        regs: Bound<'py, PyDict>,
        data_xlabel_units: &str,
        dt_ratio: f64,
        output_ptr: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let py = stats.py();
        let plt = py.import_bound("matplotlib.pyplot")?;
        let pdf = match output_ptr {
            Some(p) => p,
            None => {
                let pdf_module = py.import_bound("matplotlib.backends.backend_pdf")?;
                pdf_module.call_method1("PdfPages", ("output_plots.pdf",))?
            }
        };

        let columns: Vec<String> = stats
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        let mean_col = &columns[2];
        let var_col = &columns[3];

        let pick_model = |prefix: &str| -> PyResult<Bound<'py, PyDict>> {
            for (k, v) in regs.iter() {
                let ks: String = k.extract()?;
                if ks.starts_with(prefix) {
                    return v.extract();
                }
            }
            Err(PyValueError::new_err(format!(
                "regs missing entry starting with '{}'",
                prefix
            )))
        };

        let mean_model = pick_model("mean")?;
        let var_model = pick_model("scaled var")?;
        let mean_data = stats.call_method1("__getitem__", (mean_col,))?;
        let var_data = stats.call_method1("__getitem__", (var_col,))?;
        let mean_name: String = mean_data.getattr("name")?.extract()?;
        let label = mean_name
            .split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join(" ");

        // Mean panel
        plt.call_method0("figure")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("dt_ratio", dt_ratio)?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                mean_data.clone(),
                mean_data.getattr("name")?,
                mean_model.get_item("model")?.unwrap(),
                format!(
                    "$r^2$: {:.2}",
                    mean_model.get_item("r2")?.unwrap().extract::<f64>()?
                ),
                data_xlabel_units,
                plt.call_method0("gca")?,
            ),
            Some(&kw),
        )?;
        plt.call_method1("title", (mean_data.getattr("name")?,))?;
        pdf.call_method0("savefig")?;
        plt.call_method0("close")?;

        // Variance panel: shifted model = base_model(x) + variance.min()
        plt.call_method0("figure")?;
        let var_min = var_data.call_method0("min")?;
        let model_callable = var_model.get_item("model")?.unwrap();
        // Bind `model`/`vmin` as the lambda's globals, not eval-locals: a
        // lambda resolves its free variables against __globals__ when it is
        // later called, so eval-locals would be lost (NameError: 'model').
        let globals = PyDict::new_bound(py);
        globals.set_item("model", model_callable)?;
        globals.set_item("vmin", var_min)?;
        let shifted = py.eval_bound("lambda x: model(x) + vmin", Some(&globals), None)?;
        let kw = PyDict::new_bound(py);
        kw.set_item("dt_ratio", dt_ratio)?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                var_data.clone(),
                var_data.getattr("name")?,
                shifted,
                format!(
                    "$r^2$: {:.2}",
                    var_model.get_item("r2")?.unwrap().extract::<f64>()?
                ),
                data_xlabel_units,
                plt.call_method0("gca")?,
            ),
            Some(&kw),
        )?;
        plt.call_method1("title", (var_data.getattr("name")?,))?;
        plt.call_method0("tight_layout")?;
        pdf.call_method0("savefig")?;
        plt.call_method0("close")?;

        // Dispersion-index panel
        plt.call_method0("figure")?;
        let dispersion = mean_data.call_method1("__truediv__", (var_data,))?;
        let const_one = py.eval_bound("lambda x: [1]*len(x)", None, None)?;
        let kw = PyDict::new_bound(py);
        kw.set_item("dt_ratio", dt_ratio)?;
        kw.set_item("line_linestyle", "--")?;
        kw.set_item("line_color", "black")?;
        cls.call_method(
            "plot_single_data_and_model",
            (
                stats.getattr("index")?,
                dispersion,
                format!("dispersion index of {}", label),
                const_one,
                "Poissonian regime",
                data_xlabel_units,
                plt.call_method0("gca")?,
            ),
            Some(&kw),
        )?;
        plt.call_method1("title", (format!("Dispersion index of {}", label),))?;
        plt.call_method0("tight_layout")?;
        pdf.call_method0("savefig")?;
        plt.call_method0("close")?;

        Ok(())
    }
}

// ─────────────────────── public PyEvoMotion ───────────────────────

// The user-facing analysis class: the tip of the single-inheritance chain
// PyEvoMotion → _PyEvoMotionCore → PyEvoMotionParser → PyEvoMotionBase.
// Previously assembled in Python via multi-inheritance in
// PyEvoMotion/core/core.py; that constructor logic now lives here.
// `dict` gives instances a __dict__ so the constructor and parser can set
// Python attributes (data, reference, dt, dt_ratio, origin). The pre-port
// concrete class was a Python subclass, which provided __dict__ implicitly;
// now that PyEvoMotion is the instantiated Rust class it must declare it.
#[pyclass(subclass, dict, extends = PyEvoMotionCore, name = "PyEvoMotion", module = "PyEvoMotion")]
pub struct PyEvoMotion;

#[pymethods]
impl PyEvoMotion {
    // __new__ only builds the layout chain, so PyO3 allocates the correct
    // (possibly subclassed) type. The real work is in __init__.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyEvoMotionBase)
            .add_subclass(PyEvoMotionParser)
            .add_subclass(PyEvoMotionCore)
            .add_subclass(Self)
    }

    // Mirrors the former PyEvoMotion.__init__ (PyEvoMotion/core/core.py): set
    // the time window, run the parser (file I/O + MAFFT + mutation
    // extraction), compute the origin, and do the initial mutation-type
    // count. Method calls dispatch through `slf` so subclass overrides are
    // honoured.
    #[pyo3(signature = (input_fasta, input_meta, dt="7D", filters=None, positions=None, date_range=None, refseq=None, verbose=0, load_mutation_instructions=None, recount_mutation_types=false))]
    fn __init__<'py>(
        slf: &Bound<'py, Self>,
        input_fasta: &str,
        input_meta: &str,
        dt: &str,
        filters: Option<Bound<'py, PyDict>>,
        positions: Option<(i64, i64)>,
        date_range: Option<Bound<'py, PyAny>>,
        refseq: Option<&str>,
        verbose: i64,
        load_mutation_instructions: Option<&str>,
        recount_mutation_types: bool,
    ) -> PyResult<()> {
        let py = slf.py();

        slf.call_method1("_verify_dt", (dt,))?;
        slf.setattr("dt", dt)?;
        let ratio = slf.call_method1("_get_time_ratio", (dt,))?;
        slf.setattr("dt_ratio", ratio)?;

        // Invoke the parser's __init__ explicitly (as the old Python did).
        let filters = filters.unwrap_or_else(|| PyDict::new_bound(py));
        let positions = positions.unwrap_or((0, 0));
        let parser_type = py.get_type_bound::<PyEvoMotionParser>();
        parser_type.call_method1(
            "__init__",
            (
                slf,
                input_fasta,
                input_meta,
                filters,
                positions,
                date_range.clone(),
                refseq,
                verbose,
                load_mutation_instructions,
            ),
        )?;

        slf.call_method1(
            "_check_dataset_is_not_empty",
            (
                slf.getattr("data")?,
                "Perhaps there were no entries or the filters provided (if any) are too restrictive.",
            ),
        )?;

        // origin = data["date"].min(), tightened to the date-range start if one
        // was given (and is non-empty).
        let data = slf.getattr("data")?;
        let date_min = data
            .call_method1("__getitem__", ("date",))?
            .call_method0("min")?;
        let origin = match &date_range {
            Some(dr) => {
                let start = dr.call_method1("__getitem__", (0i64,))?;
                if start.is_truthy()? {
                    py.import_bound("builtins")?
                        .call_method1("min", (date_min, start))?
                } else {
                    date_min
                }
            }
            None => date_min,
        };
        slf.setattr("origin", origin)?;

        // A loaded TSV already carries the per-sequence counts; reuse them
        // unless asked to recount (or they are missing from the file).
        let columns: Vec<String> = data.getattr("columns")?.call_method0("tolist")?.extract()?;
        let has_counts = [
            "number of substitutions",
            "number of indels",
            "number of insertions",
            "number of deletions",
            "number of mutations",
        ]
        .iter()
        .all(|c| columns.iter().any(|col| col == c));
        if load_mutation_instructions.is_none() || recount_mutation_types || !has_counts {
            slf.call_method0("count_mutation_types")?;
        }
        Ok(())
    }
}

/// tp_init trampoline for `PyEvoMotion`.
///
/// PyO3 wires `#[new]` (tp_new) but never wires a `#[pymethods] __init__` to
/// the tp_init slot, so directly instantiating the Rust class would skip
/// `__init__`. `lib.rs` installs this trampoline on the type at module init.
/// It just invokes the Python-level `__init__` (the `#[pymethods]` one above)
/// via the instance, so subclasses that inherit this slot dispatch correctly
/// and subclass overrides are honoured. Subclasses that define their own
/// `__init__` get CPython's normal slot wiring instead and never reach here.
pub unsafe extern "C" fn pyevomotion_tp_init(
    slf: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> c_int {
    Python::with_gil(|py| {
        let slf = Bound::from_borrowed_ptr(py, slf);
        let args = Bound::from_borrowed_ptr(py, args);
        let args = match args.downcast_into::<PyTuple>() {
            Ok(t) => t,
            Err(e) => {
                PyErr::from(e).restore(py);
                return -1;
            }
        };
        let kwargs = if kwargs.is_null() {
            None
        } else {
            Some(Bound::from_borrowed_ptr(py, kwargs))
        };
        let kwargs = kwargs
            .as_ref()
            .map(|k| k.downcast::<PyDict>().expect("kwargs is always a dict"));

        match slf.call_method("__init__", args, kwargs) {
            Ok(_) => 0,
            Err(e) => {
                e.restore(py);
                -1
            }
        }
    })
}
