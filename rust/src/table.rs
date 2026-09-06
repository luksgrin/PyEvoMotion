//! The internal column store that replaces pandas inside the pipeline.
//!
//! See `rust/DESIGN_internal_table.md`. A `Table` is a list of named, typed
//! columns plus an index description. Its type system mirrors the dtypes
//! `pandas.read_csv` produces so that `from_pandas(df).to_pandas()` gives back
//! an equal DataFrame (values, dtypes, index) for everything PyEvoMotion
//! itself can produce. Anything outside that set survives untouched through
//! the `PyObject` / `Foreign` escape hatches.
//!
//! pandas is touched only in `from_pandas` / `to_pandas`; every other
//! operation here is plain Rust.

use std::collections::HashMap;
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};

/// NaT marker for datetime columns (numpy's convention).
pub const NAT: i64 = i64::MIN;

/// Missing marker inside `DictStr::codes`.
const NA_CODE: u32 = u32::MAX;

/// Time unit of a pandas `datetime64[unit]` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeUnit {
    S,
    Ms,
    Us,
    Ns,
}

impl TimeUnit {
    fn parse(s: &str) -> Option<TimeUnit> {
        match s {
            "datetime64[s]" => Some(TimeUnit::S),
            "datetime64[ms]" => Some(TimeUnit::Ms),
            "datetime64[us]" => Some(TimeUnit::Us),
            "datetime64[ns]" => Some(TimeUnit::Ns),
            _ => None,
        }
    }
    fn numpy_name(self) -> &'static str {
        match self {
            TimeUnit::S => "datetime64[s]",
            TimeUnit::Ms => "datetime64[ms]",
            TimeUnit::Us => "datetime64[us]",
            TimeUnit::Ns => "datetime64[ns]",
        }
    }
    /// Nanoseconds per unit.
    fn to_ns_factor(self) -> i64 {
        match self {
            TimeUnit::S => 1_000_000_000,
            TimeUnit::Ms => 1_000_000,
            TimeUnit::Us => 1_000,
            TimeUnit::Ns => 1,
        }
    }
}

/// Dictionary-encoded string column. Metadata columns are low-cardinality
/// (`betacoronavirus`, `Europe`, `?`), so this keeps memory small and lets
/// `to_pandas` create one Python string per distinct value.
#[derive(Clone, Default)]
pub struct DictStr {
    codes: Vec<u32>,
    values: Vec<Arc<str>>,
}

impl DictStr {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_options<I: IntoIterator<Item = Option<String>>>(items: I) -> Self {
        let mut d = DictStr::new();
        let mut lookup: HashMap<Arc<str>, u32> = HashMap::new();
        for item in items {
            match item {
                None => d.codes.push(NA_CODE),
                Some(s) => {
                    let code = match lookup.get(s.as_str()) {
                        Some(&c) => c,
                        None => {
                            let c = d.values.len() as u32;
                            let a: Arc<str> = Arc::from(s.as_str());
                            d.values.push(a.clone());
                            lookup.insert(a, c);
                            c
                        }
                    };
                    d.codes.push(code);
                }
            }
        }
        d
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn get(&self, i: usize) -> Option<&str> {
        let c = self.codes[i];
        if c == NA_CODE {
            None
        } else {
            Some(&self.values[c as usize])
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    fn take(&self, rows: &[usize]) -> DictStr {
        // Keep the dictionary; only the codes are re-selected.
        DictStr {
            codes: rows.iter().map(|&r| self.codes[r]).collect(),
            values: self.values.clone(),
        }
    }
}

/// One column. Missing values: NaN for `Float64`, `NA_CODE` for `Str`,
/// `None` for `StrList`, `NAT` for `DatetimeNs`.
pub enum Column {
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Float64(Vec<f64>),
    Bool(Vec<bool>),
    Str(DictStr),
    StrList(Vec<Option<Vec<String>>>),
    DatetimeNs { ns: Vec<i64>, unit: TimeUnit },
    /// An object column holding arbitrary Python objects (kept as they are).
    PyObject(Vec<Py<PyAny>>),
    /// Any other pandas dtype: the original Series is kept and row selections
    /// are composed into `take`, applied lazily with `Series.take`.
    Foreign { series: Py<PyAny>, take: Option<Vec<usize>> },
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(v) => v.len(),
            Column::UInt64(v) => v.len(),
            Column::Float64(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Str(d) => d.len(),
            Column::StrList(v) => v.len(),
            Column::DatetimeNs { ns, .. } => ns.len(),
            Column::PyObject(v) => v.len(),
            Column::Foreign { series, take } => match take {
                Some(t) => t.len(),
                None => Python::with_gil(|py| series.bind(py).len().unwrap_or(0)),
            },
        }
    }

    /// pandas dtype name, as `str(series.dtype)` would print it.
    pub fn dtype_name(&self, py: Python<'_>) -> String {
        match self {
            Column::Int64(_) => "int64".into(),
            Column::UInt64(_) => "uint64".into(),
            Column::Float64(_) => "float64".into(),
            Column::Bool(_) => "bool".into(),
            Column::Str(_) | Column::StrList(_) | Column::PyObject(_) => "object".into(),
            Column::DatetimeNs { unit, .. } => unit.numpy_name().into(),
            Column::Foreign { series, .. } => series
                .bind(py)
                .getattr("dtype")
                .and_then(|d| d.str())
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "object".into()),
        }
    }

    /// Row selection (mask or permutation expressed as row indices).
    pub fn take(&self, py: Python<'_>, rows: &[usize]) -> Column {
        match self {
            Column::Int64(v) => Column::Int64(rows.iter().map(|&r| v[r]).collect()),
            Column::UInt64(v) => Column::UInt64(rows.iter().map(|&r| v[r]).collect()),
            Column::Float64(v) => Column::Float64(rows.iter().map(|&r| v[r]).collect()),
            Column::Bool(v) => Column::Bool(rows.iter().map(|&r| v[r]).collect()),
            Column::Str(d) => Column::Str(d.take(rows)),
            Column::StrList(v) => Column::StrList(rows.iter().map(|&r| v[r].clone()).collect()),
            Column::DatetimeNs { ns, unit } => Column::DatetimeNs {
                ns: rows.iter().map(|&r| ns[r]).collect(),
                unit: *unit,
            },
            Column::PyObject(v) => {
                Column::PyObject(rows.iter().map(|&r| v[r].clone_ref(py)).collect())
            }
            Column::Foreign { series, take } => {
                let composed: Vec<usize> = match take {
                    Some(t) => rows.iter().map(|&r| t[r]).collect(),
                    None => rows.to_vec(),
                };
                Column::Foreign {
                    series: series.clone_ref(py),
                    take: Some(composed),
                }
            }
        }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Column {
        match self {
            Column::Int64(v) => Column::Int64(v.clone()),
            Column::UInt64(v) => Column::UInt64(v.clone()),
            Column::Float64(v) => Column::Float64(v.clone()),
            Column::Bool(v) => Column::Bool(v.clone()),
            Column::Str(d) => Column::Str(d.clone()),
            Column::StrList(v) => Column::StrList(v.clone()),
            Column::DatetimeNs { ns, unit } => Column::DatetimeNs { ns: ns.clone(), unit: *unit },
            Column::PyObject(v) => Column::PyObject(v.iter().map(|o| o.clone_ref(py)).collect()),
            Column::Foreign { series, take } => Column::Foreign {
                series: series.clone_ref(py),
                take: take.clone(),
            },
        }
    }
}

/// The DataFrame index, reproduced so that `to_pandas` gives back a
/// `RangeIndex` or an int64 `Index` exactly where pandas would have one.
pub enum IndexKind {
    Range { start: i64, len: usize },
    Labels(Vec<i64>),
    /// Any other index (MultiIndex, strings, datetimes): kept as the pandas
    /// object with a lazy row selection, like `Column::Foreign`.
    Foreign { index: Py<PyAny>, take: Option<Vec<usize>> },
}

impl IndexKind {
    pub fn range(len: usize) -> Self {
        IndexKind::Range { start: 0, len }
    }

    pub fn len(&self, py: Python<'_>) -> usize {
        match self {
            IndexKind::Range { len, .. } => *len,
            IndexKind::Labels(v) => v.len(),
            IndexKind::Foreign { index, take } => match take {
                Some(t) => t.len(),
                None => index.bind(py).len().unwrap_or(0),
            },
        }
    }

    /// Label of row `i` when labels are integers (Range or Labels).
    pub fn label(&self, i: usize) -> Option<i64> {
        match self {
            IndexKind::Range { start, .. } => Some(start + i as i64),
            IndexKind::Labels(v) => Some(v[i]),
            IndexKind::Foreign { .. } => None,
        }
    }

    /// Selecting rows keeps their labels (pandas: boolean mask / sort).
    pub fn take(&self, py: Python<'_>, rows: &[usize]) -> IndexKind {
        match self {
            IndexKind::Range { start, .. } => {
                IndexKind::Labels(rows.iter().map(|&r| start + r as i64).collect())
            }
            IndexKind::Labels(v) => IndexKind::Labels(rows.iter().map(|&r| v[r]).collect()),
            IndexKind::Foreign { index, take } => IndexKind::Foreign {
                index: index.clone_ref(py),
                take: Some(match take {
                    Some(t) => rows.iter().map(|&r| t[r]).collect(),
                    None => rows.to_vec(),
                }),
            },
        }
    }

    pub fn clone_ref(&self, py: Python<'_>) -> IndexKind {
        match self {
            IndexKind::Range { start, len } => IndexKind::Range { start: *start, len: *len },
            IndexKind::Labels(v) => IndexKind::Labels(v.clone()),
            IndexKind::Foreign { index, take } => IndexKind::Foreign {
                index: index.clone_ref(py),
                take: take.clone(),
            },
        }
    }
}

/// A table: ordered named columns plus an index.
pub struct Table {
    pub columns: Vec<(String, Column)>,
    pub index: IndexKind,
}

impl Table {
    pub fn empty() -> Self {
        Table {
            columns: Vec::new(),
            index: IndexKind::range(0),
        }
    }

    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        // pandas `DataFrame.empty`: no rows or no columns.
        self.columns.is_empty() || self.nrows() == 0
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|(n, _)| n == name)
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    pub fn column_mut(&mut self, name: &str) -> Option<&mut Column> {
        self.columns.iter_mut().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Add or replace a column (pandas `df[name] = ...` with a length-aligned
    /// value; the caller guarantees the length).
    pub fn set_column(&mut self, name: &str, col: Column) {
        if let Some(slot) = self.columns.iter_mut().find(|(n, _)| n == name) {
            slot.1 = col;
        } else {
            self.columns.push((name.to_string(), col));
        }
    }

    /// Row selection keeping labels (pandas boolean mask / sort_values).
    pub fn take(&self, py: Python<'_>, rows: &[usize]) -> Table {
        Table {
            columns: self
                .columns
                .iter()
                .map(|(n, c)| (n.clone(), c.take(py, rows)))
                .collect(),
            index: self.index.take(py, rows),
        }
    }

    /// pandas `reset_index(drop=True)`.
    pub fn reset_index(&mut self) {
        self.index = IndexKind::range(self.nrows());
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Table {
        Table {
            columns: self
                .columns
                .iter()
                .map(|(n, c)| (n.clone(), c.clone_ref(py)))
                .collect(),
            index: self.index.clone_ref(py),
        }
    }

    // ─────────────────────── pandas boundary ───────────────────────

    /// Ingest a pandas DataFrame (exact dtype mapping, see the design doc §3.3).
    pub fn from_pandas(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Table> {
        let pd = py.import_bound("pandas")?;
        let dataframe_type = pd.getattr("DataFrame")?;
        if !df.is_instance(&dataframe_type)? {
            return Err(PyTypeError::new_err(format!(
                "expected a pandas.DataFrame, got {}",
                df.get_type().name()?
            )));
        }

        let names: Vec<Bound<PyAny>> = df.getattr("columns")?.call_method0("tolist")?.extract()?;
        let mut columns = Vec::with_capacity(names.len());
        for (i, name_obj) in names.iter().enumerate() {
            let name: String = name_obj.extract().map_err(|_| {
                PyTypeError::new_err("Table requires string column names")
            })?;
            // iloc handles duplicate names too.
            let series = df.getattr("iloc")?.call_method1(
                "__getitem__",
                (PyTuple::new_bound(py, [py.Ellipsis(), i.into_py(py)]),),
            )?;
            columns.push((name, Self::ingest_series(py, &series)?));
        }

        let index = Self::ingest_index(py, &df.getattr("index")?)?;
        Ok(Table { columns, index })
    }

    fn ingest_series(py: Python<'_>, series: &Bound<'_, PyAny>) -> PyResult<Column> {
        let dtype = series.getattr("dtype")?.str()?.to_string();
        match dtype.as_str() {
            "int64" => {
                let arr: PyReadonlyArray1<i64> = series.call_method0("to_numpy")?.extract()?;
                Ok(Column::Int64(arr.as_slice()?.to_vec()))
            }
            "uint64" => {
                let arr: PyReadonlyArray1<u64> = series.call_method0("to_numpy")?.extract()?;
                Ok(Column::UInt64(arr.as_slice()?.to_vec()))
            }
            "float64" => {
                let arr: PyReadonlyArray1<f64> = series.call_method0("to_numpy")?.extract()?;
                Ok(Column::Float64(arr.as_slice()?.to_vec()))
            }
            "bool" => {
                let arr: PyReadonlyArray1<bool> = series.call_method0("to_numpy")?.extract()?;
                Ok(Column::Bool(arr.as_slice()?.to_vec()))
            }
            "object" => Self::ingest_object_series(py, series),
            other => {
                if let Some(unit) = TimeUnit::parse(other) {
                    let np = series.call_method0("to_numpy")?;
                    let as_i64 = np.call_method1("view", ("int64",))?;
                    let arr: PyReadonlyArray1<i64> = as_i64.extract()?;
                    let raw = arr.as_slice()?;
                    let f = unit.to_ns_factor();
                    let ns: Vec<i64> = raw
                        .iter()
                        .map(|&v| if v == NAT { NAT } else { v * f })
                        .collect();
                    Ok(Column::DatetimeNs { ns, unit })
                } else {
                    Ok(Column::Foreign {
                        series: series.clone().unbind(),
                        take: None,
                    })
                }
            }
        }
    }

    fn ingest_object_series(_py: Python<'_>, series: &Bound<'_, PyAny>) -> PyResult<Column> {
        let items: Bound<PyList> = series.call_method0("tolist")?.downcast_into()?;
        let n = items.len();

        // Pass 1: all str or missing (NaN/None) → Str.
        let mut strs: Vec<Option<String>> = Vec::with_capacity(n);
        let mut all_str = true;
        for item in items.iter() {
            if is_missing(&item) {
                strs.push(None);
            } else if let Ok(s) = item.downcast::<PyString>() {
                strs.push(Some(s.to_string()));
            } else {
                all_str = false;
                break;
            }
        }
        if all_str {
            return Ok(Column::Str(DictStr::from_options(strs)));
        }

        // Pass 2: all list/tuple of str, or missing → StrList.
        let mut lists: Vec<Option<Vec<String>>> = Vec::with_capacity(n);
        let mut all_lists = true;
        for item in items.iter() {
            if is_missing(&item) {
                lists.push(None);
                continue;
            }
            let seq = if item.downcast::<PyList>().is_ok() || item.downcast::<PyTuple>().is_ok() {
                item.iter()?
            } else {
                all_lists = false;
                break;
            };
            let mut out = Vec::new();
            let mut ok = true;
            for el in seq {
                match el?.downcast::<PyString>() {
                    Ok(s) => out.push(s.to_string()),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                all_lists = false;
                break;
            }
            lists.push(Some(out));
        }
        if all_lists {
            return Ok(Column::StrList(lists));
        }

        // Fallback: opaque Python objects.
        Ok(Column::PyObject(items.iter().map(|o| o.unbind()).collect()))
    }

    fn ingest_index(py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<IndexKind> {
        let pd = py.import_bound("pandas")?;
        if index.is_instance(&pd.getattr("RangeIndex")?)? {
            let start: i64 = index.getattr("start")?.extract()?;
            let step: i64 = index.getattr("step")?.extract()?;
            let len: usize = index.len()?;
            if step == 1 {
                return Ok(IndexKind::Range { start, len });
            }
        }
        let dtype = index.getattr("dtype")?.str()?.to_string();
        if dtype == "int64" {
            let arr: PyReadonlyArray1<i64> = index.call_method0("to_numpy")?.extract()?;
            return Ok(IndexKind::Labels(arr.as_slice()?.to_vec()));
        }
        Ok(IndexKind::Foreign {
            index: index.clone().unbind(),
            take: None,
        })
    }

    /// Materialise as a pandas DataFrame.
    pub fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import_bound("pandas")?;
        let data = PyDict::new_bound(py);
        let names = PyList::empty_bound(py);
        for (name, col) in &self.columns {
            names.append(name)?;
            data.set_item(name, Self::column_to_python(py, col)?)?;
        }
        let index = self.index_to_python(py)?;
        let kw = PyDict::new_bound(py);
        kw.set_item("index", index)?;
        kw.set_item("columns", &names)?;
        let df = pd.call_method("DataFrame", (data,), Some(&kw))?;
        Ok(df)
    }

    fn column_to_python<'py>(py: Python<'py>, col: &Column) -> PyResult<Bound<'py, PyAny>> {
        Ok(match col {
            Column::Int64(v) => v.clone().into_pyarray_bound(py).into_any(),
            Column::UInt64(v) => v.clone().into_pyarray_bound(py).into_any(),
            Column::Float64(v) => v.clone().into_pyarray_bound(py).into_any(),
            Column::Bool(v) => v.clone().into_pyarray_bound(py).into_any(),
            Column::DatetimeNs { ns, unit } => {
                let f = unit.to_ns_factor();
                let raw: Vec<i64> = ns.iter().map(|&v| if v == NAT { NAT } else { v / f }).collect();
                let arr = raw.into_pyarray_bound(py);
                arr.call_method1("view", (unit.numpy_name(),))?
            }
            Column::Str(d) => {
                // One PyString per distinct value; NaN for missing → object dtype.
                let py_values: Vec<Bound<PyString>> =
                    d.values.iter().map(|s| PyString::new_bound(py, s)).collect();
                let nan = PyFloat::new_bound(py, f64::NAN);
                let out = PyList::empty_bound(py);
                for &c in &d.codes {
                    if c == NA_CODE {
                        out.append(&nan)?;
                    } else {
                        out.append(&py_values[c as usize])?;
                    }
                }
                object_array(py, &out)?
            }
            Column::StrList(v) => {
                let out = PyList::empty_bound(py);
                for item in v {
                    match item {
                        None => out.append(py.None())?,
                        Some(list) => out.append(PyList::new_bound(py, list))?,
                    }
                }
                object_array(py, &out)?
            }
            Column::PyObject(v) => {
                let out = PyList::new_bound(py, v.iter().map(|o| o.bind(py)));
                object_array(py, &out)?
            }
            Column::Foreign { series, take } => {
                // `Series.array` is the dtype-preserving, index-free backing
                // array (ExtensionArray or NumpyExtensionArray); `take` is
                // positional on it. `to_numpy()` would drop e.g. categoricals.
                let arr = series.bind(py).getattr("array")?;
                match take {
                    None => arr,
                    Some(t) => arr.call_method1("take", (t.clone(),))?,
                }
            }
        })
    }

    fn index_to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import_bound("pandas")?;
        match &self.index {
            IndexKind::Range { start, len } => {
                pd.call_method1("RangeIndex", (*start, start + *len as i64))
            }
            IndexKind::Labels(v) => {
                let arr = v.clone().into_pyarray_bound(py);
                let kw = PyDict::new_bound(py);
                kw.set_item("dtype", "int64")?;
                pd.call_method("Index", (arr,), Some(&kw))
            }
            IndexKind::Foreign { index, take } => {
                let ix = index.bind(py);
                match take {
                    None => Ok(ix.clone()),
                    Some(t) => ix.call_method1("take", (t.clone(),)),
                }
            }
        }
    }
}

/// pandas' notion of a missing object cell: None or a float NaN.
fn is_missing(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_none() {
        return true;
    }
    if let Ok(f) = obj.downcast::<PyFloat>() {
        return f.value().is_nan();
    }
    false
}

/// A 1-D numpy object array from a list (so the DataFrame keeps object
/// dtype and does not try to infer anything else).
fn object_array<'py>(py: Python<'py>, list: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("empty", (list.len(), "object"))?;
    arr.call_method1("__setitem__", (py.Ellipsis(), list))?;
    Ok(arr)
}

// ─────────────────────── Python class ───────────────────────

/// Python-visible wrapper: `PyEvoMotion.Table`.
#[pyclass(name = "Table", module = "PyEvoMotion")]
pub struct TablePy {
    pub inner: Table,
}

#[pymethods]
impl TablePy {
    /// Build a Table from a pandas DataFrame (exact dtype mapping).
    #[staticmethod]
    fn from_pandas(py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(TablePy {
            inner: Table::from_pandas(py, &df)?,
        })
    }

    /// Materialise as a pandas DataFrame (values, dtypes and index as pandas
    /// itself would have produced them).
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.to_pandas(py)
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    /// Column names, in order.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns.iter().map(|(n, _)| n.clone()).collect()
    }

    /// True when there are no rows or no columns (pandas `DataFrame.empty`).
    #[getter]
    fn empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Mapping of column name to dtype name, using pandas' dtype names.
    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (n, c) in &self.inner.columns {
            d.set_item(n, c.dtype_name(py))?;
        }
        Ok(d)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.has_column(name)
    }

    /// Cell values of one column as a Python list (NaN/None for missing).
    fn column<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.column(name) {
            Some(col) => Table::column_to_python(py, col)?.call_method0("tolist"),
            None => Err(PyKeyError::new_err(name.to_string())),
        }
    }

    /// Write the table as delimited text exactly as
    /// ``to_pandas().to_csv(path, sep=sep, index=False)`` would. Falls back to
    /// pandas for content the Rust writer does not render (opaque objects,
    /// foreign dtypes, sub-second timestamps).
    #[pyo3(signature = (path, sep="\t"))]
    fn to_tsv(&self, py: Python<'_>, path: &str, sep: &str) -> PyResult<()> {
        let sep_char = sep.chars().next().unwrap_or('\t');
        if !crate::csv_write::write_delimited(py, &self.inner, path, sep_char)? {
            let kw = PyDict::new_bound(py);
            kw.set_item("sep", sep)?;
            kw.set_item("index", false)?;
            self.inner.to_pandas(py)?.call_method("to_csv", (path,), Some(&kw))?;
        }
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Table({} rows x {} columns: {})",
            self.inner.nrows(),
            self.inner.columns.len(),
            self.inner
                .columns
                .iter()
                .map(|(n, c)| format!("{}: {}", n, c.dtype_name(py)))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Validate that a Python object is a pandas DataFrame; used by the `data` setter.
pub fn is_dataframe(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let pd = py.import_bound("pandas")?;
    obj.is_instance(&pd.getattr("DataFrame")?)
}

pub fn type_error_not_table(obj: &Bound<'_, PyAny>) -> PyErr {
    PyValueError::new_err(format!(
        "`data` must be a pandas.DataFrame or a PyEvoMotion.Table, got {}",
        obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    ))
}
