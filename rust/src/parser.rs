use std::collections::{HashMap, HashSet};

use mafft::{AlignmentMode, MafftEngine, Sequence, SequenceSet, SeqType};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple, PyType};

use crate::base::PyEvoMotionBase;
use crate::fasta::{self, FastaReader, SequenceRecord};
use crate::csv_read;
use crate::dates;
use crate::table::{self, Column, Table, TablePy, TimeUnit};

// ─────────────────────── pure-Rust algorithms ───────────────────────

/// 0 = match (or N involved), 1 = substitution, 2 = insertion (gap in
/// reference), 3 = deletion (gap in target). Mirrors the original
/// PyEvoMotionParser._column_decision exactly.
fn column_decision(c0: char, c1: char) -> u8 {
    if c0 == c1 || c0 == 'N' || c1 == 'N' {
        0
    } else if c0 == '-' {
        2
    } else if c1 == '-' {
        3
    } else {
        1
    }
}

/// Group consecutive integers (assumed sorted ascending).
/// `[1,2,3,5,6,8]` → `[[1,2,3],[5,6],[8]]`.
fn get_consecutives(data: &[usize]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut prev: Option<usize> = None;
    for &v in data {
        match prev {
            Some(p) if v == p + 1 => current.push(v),
            _ => {
                if !current.is_empty() {
                    groups.push(std::mem::take(&mut current));
                }
                current.push(v);
            }
        }
        prev = Some(v);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// The hot path: build the sorted i_/d_/s_ mutation list from a 2-row
/// alignment. seq0 = reference, seq1 = target. Both already upper-cased.
///
/// Encoding: `s_P_B` (base B at position P), `i_P_BASES` (BASES inserted so
/// that the first inserted base sits at position P) and `d_P_BASES` (BASES
/// deleted starting at position P). P is 1-based for all three kinds and is
/// expressed in reference coordinates (positions after an insertion are
/// shifted back by its length, see below). Before v0.2.0 insertions and
/// deletions were 0-based while substitutions were 1-based.
fn create_modifs_from_strings(seq0: &str, seq1: &str) -> Vec<String> {
    let upper0: Vec<char> = seq0.chars().collect();
    let upper1: Vec<char> = seq1.chars().collect();
    let n = upper0.len().min(upper1.len());

    let mut mut_class = Vec::with_capacity(n);
    for i in 0..n {
        mut_class.push(column_decision(upper0[i], upper1[i]));
    }

    let mut subst: Vec<String> = Vec::new();
    for i in 0..n {
        if mut_class[i] == 1 {
            subst.push(format!("s_{}_{}", i + 1, upper1[i]));
        }
    }

    let ins_idxs: Vec<usize> = (0..n).filter(|&i| mut_class[i] == 2).collect();
    let mut insertions: Vec<String> = Vec::new();
    for group in get_consecutives(&ins_idxs) {
        let bases: String = group.iter().map(|&i| upper1[i]).collect();
        insertions.push(format!("i_{}_{}", group[0] + 1, bases));
    }

    let del_idxs: Vec<usize> = (0..n).filter(|&i| mut_class[i] == 3).collect();
    let mut deletions: Vec<String> = Vec::new();
    for group in get_consecutives(&del_idxs) {
        let bases: String = group.iter().map(|&i| upper0[i]).collect();
        deletions.push(format!("d_{}_{}", group[0] + 1, bases));
    }

    let mut mods: Vec<String> = subst
        .into_iter()
        .chain(insertions.into_iter())
        .chain(deletions.into_iter())
        .collect();
    mods.sort_by_key(|x| {
        let parts: Vec<&str> = x.split('_').collect();
        parts.get(1).and_then(|s| s.parse::<i64>().ok()).unwrap_or(0)
    });

    // For each insertion, shift positions of all later mutations down by
    // the insertion length so subsequent positions remain expressed in the
    // reference (pre-insertion) coordinate system.
    let reindex: Vec<(usize, i64)> = mods
        .iter()
        .enumerate()
        .filter(|(_, m)| m.starts_with('i'))
        .map(|(i, m)| {
            let last = m.rsplit('_').next().unwrap_or("");
            (i, last.chars().count() as i64)
        })
        .collect();

    for (idx, v) in reindex {
        for j in (idx + 1)..mods.len() {
            let parts: Vec<&str> = mods[j].split('_').collect();
            let pos: i64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let new_pos = pos - v;
            let tail = parts[2..].join("_");
            mods[j] = format!("{}_{}_{}", parts[0], new_pos, tail);
        }
    }

    mods
}

/// In-process alignment via the pure-Rust `mafft` crate — no external
/// binary. Returns the same aligned-FASTA string the old `mafft`
/// subprocess produced; generate_alignment parses it back into two
/// SequenceRecords.
///
/// Residues are lower-cased to mirror C MAFFT's default output: the
/// N-count in get_differing_mutations counts lowercase 'n', while
/// create_modifs re-uppercases before calling mutations. Releases the
/// GIL during the CPU-bound alignment.
fn run_mafft_inner(
    py: Python<'_>,
    seqs: &[(String, String)],
    outformat: &str,
) -> PyResult<String> {
    if outformat != "fasta" {
        // The pipeline only ever requests fasta; the legacy `--clustalout`
        // path had no callers or tests. Fail loudly rather than emit a
        // format the in-process aligner doesn't render here.
        return Err(PyValueError::new_err(format!(
            "Unsupported MAFFT output format '{}'; the in-process aligner emits 'fasta' only",
            outformat
        )));
    }

    py.allow_threads(|| {
        // PyEvoMotion only ever aligns DNA (viral genomes), so the type is
        // fixed rather than sniffed — a short/degenerate pair could
        // otherwise misdetect.
        let mut set = SequenceSet::new(SeqType::Dna);
        for (name, seq) in seqs {
            set.sequences.push(Sequence {
                name: name.clone(),
                data: seq.as_bytes().to_vec(),
            });
        }

        // FFT-NS-2 is MAFFT's no-flag default (what the subprocess used);
        // --c-compat replicates C MAFFT's scoring and gap placement so the
        // aligned columns — and therefore the called mutations — match.
        let engine = MafftEngine::new(AlignmentMode::FftNs2).with_c_compat(true);
        let msa = engine.align(&set);

        // Serialize to FASTA (unwrapped lines), lower-casing residues to
        // match C MAFFT's default output.
        let mut out = String::new();
        for (name, seq) in msa.names.iter().zip(msa.sequences.iter()) {
            out.push('>');
            out.push_str(name);
            out.push('\n');
            out.extend(seq.iter().map(|&b| (b as char).to_ascii_lowercase()));
            out.push('\n');
        }

        Ok(out)
    })
}

/// Emit an INFO record on the ``PyEvoMotion`` Python logger. Library users
/// configure it with the standard ``logging`` module; the CLI's -v flag
/// installs a basic handler.
fn log_info(py: Python<'_>, msg: &str) -> PyResult<()> {
    py.import_bound("logging")?
        .call_method1("getLogger", ("PyEvoMotion",))?
        .call_method1("info", (msg,))?;
    Ok(())
}

/// Verbosity level stored on the instance by __init__ (0 when unset, e.g.
/// when parser methods are driven directly from Python).
fn verbosity(slf: &Bound<'_, PyAny>) -> i64 {
    slf.getattr("verbose")
        .and_then(|v| v.extract::<i64>())
        .unwrap_or(0)
}

// ─────────────────────── pyclass ───────────────────────

// PyEvoMotionParser extends PyEvoMotionBase so that the user-facing
// PyEvoMotion(_PyEvoMotionCore, PyEvoMotionParser) multi-inheritance has
// a shared layout root (both bases trace back to PyEvoMotionBase).
#[pyclass(subclass, extends = PyEvoMotionBase, name = "PyEvoMotionParser", module = "PyEvoMotion")]
#[derive(Default)]
pub struct PyEvoMotionParser {
    // The instance's data lives in exactly one of two states (design doc §4.2):
    //   Rust-owned:     table = Some, df = None
    //   pandas-visible: df = Some (the DataFrame handed out through `.data`, or
    //                   assigned to it, is the truth until the next Rust stage
    //                   re-ingests it); table may be stale and is ignored.
    // Before __init__ has run both are None and `.data` raises AttributeError,
    // as an unset attribute did before.
    pub(crate) table: Option<Table>,
    pub(crate) df: Option<Py<PyAny>>,
}

#[pymethods]
impl PyEvoMotionParser {
    // __new__: cooperative no-op so multi-inheritance instantiation works.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> (Self, PyEvoMotionBase) {
        (Self::default(), PyEvoMotionBase)
    }

    // data property ------------------------------------------------------
    /// The parsed dataset as a ``pandas.DataFrame``. Reading it materialises
    /// the internal table (once, until the next pipeline stage); assigning a
    /// DataFrame or a :class:`Table` replaces the dataset.
    #[getter(data)]
    fn get_data<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        {
            let this = slf.borrow();
            if let Some(df) = &this.df {
                return Ok(df.bind(py).clone());
            }
        }
        let df = {
            let this = slf.borrow();
            match &this.table {
                Some(t) => t.to_pandas(py)?,
                None => {
                    return Err(PyAttributeError::new_err(format!(
                        "'{}' object has no attribute 'data'",
                        slf.get_type().name()?
                    )))
                }
            }
        };
        slf.borrow_mut().df = Some(df.clone().unbind());
        Ok(df)
    }

    #[setter(data)]
    fn set_data(slf: &Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if let Ok(t) = value.downcast::<TablePy>() {
            let inner = t.borrow().inner.clone_ref(py);
            let mut this = slf.borrow_mut();
            this.table = Some(inner);
            this.df = None;
        } else if table::is_dataframe(py, &value)? {
            let mut this = slf.borrow_mut();
            this.df = Some(value.unbind());
            this.table = None;
        } else {
            return Err(table::type_error_not_table(&value));
        }
        Ok(())
    }

    // __init__: orchestrates parse_metadata → parse_sequence_by_id →
    // filter_columns → (optional filter_by_daterange) → parse_data →
    // filter_by_position. Calls those methods through Python attribute
    // lookup so subclass overrides take effect.
    //
    // With `load_mutation_instructions` (path to a `<out>.tsv` written by an
    // earlier run) the metadata parsing and the alignment step are skipped:
    // the TSV already carries id, date, metadata columns and the mutation
    // instructions. Filters are still applied, so a previous result can be
    // re-analysed with a different date range / genome window / metadata
    // filter in seconds instead of re-aligning every sequence.
    #[pyo3(signature = (input_fasta, input_meta, filters, positions, date_range=None, refseq=None, verbose=0, load_mutation_instructions=None))]
    fn __init__<'py>(
        slf: &Bound<'py, Self>,
        input_fasta: &str,
        input_meta: &str,
        filters: Bound<'py, PyDict>,
        positions: (i64, i64),
        date_range: Option<Bound<'py, PyAny>>,
        refseq: Option<&str>,
        verbose: i64,
        load_mutation_instructions: Option<&str>,
    ) -> PyResult<()> {
        let py = slf.py();
        slf.setattr("verbose", verbose)?;

        match load_mutation_instructions {
            Some(path) => {
                log_info(py, "Loading mutation instructions from TSV")?;
                // Dispatch through Python so an override of parse_mutation_data
                // is honoured; the default returns a DataFrame built from the
                // Rust reader.
                let data = slf.call_method1("parse_mutation_data", (path,))?;
                slf.setattr("data", data)?;
                Self::with_table(slf, |py, t| drop_missing_instructions_table(py, t))?;
            }
            None => {
                log_info(py, "Parsing metadata file")?;
                let table = parse_metadata_inner(py, input_meta)?;
                Self::store_table(slf, table);
            }
        }

        log_info(py, "Loading reference sequence")?;
        slf.call_method1("load_reference", (refseq, input_fasta))?;

        log_info(py, "Filtering metadata columns")?;
        slf.call_method1("filter_columns", (filters,))?;

        if let Some(dr) = date_range {
            log_info(py, "Filtering by date range")?;
            let start_any = dr.call_method1("__getitem__", (0i64,))?;
            let end_any = dr.call_method1("__getitem__", (1i64,))?;
            slf.call_method1("filter_by_daterange", (start_any, end_any))?;
        }

        if load_mutation_instructions.is_none() {
            log_info(py, "Parsing FASTA file, aligning sequences and calling mutations")?;
            // parse_data keeps its public signature (a pandas Series of ids).
            let ids: Vec<Option<String>> = Self::with_table(slf, |_, t| match t.column("id") {
                Some(Column::Str(d)) => Ok(d.iter().map(|s| s.map(str::to_string)).collect()),
                Some(_) => Err(PyValueError::new_err("metadata `id` column must contain strings")),
                None => Err(PyValueError::new_err("Metadata file must contain an \"id\" column")),
            })?;
            let pd = py.import_bound("pandas")?;
            let ids = pd.call_method1("Series", (ids,))?;
            slf.call_method1("parse_data", (input_fasta, ids))?;
        }

        log_info(py, "Filtering by genome position")?;
        slf.call_method1("filter_by_position", (positions.0, positions.1))?;

        Ok(())
    }

    // load_reference ----------------------------------------------------
    /// Set ``self.reference``. With ``refseq`` (path to a FASTA file) the
    /// first record of that file is the reference; otherwise the reference
    /// is the sequence in ``input_fasta`` whose id is the first row of
    /// ``self.data`` (the earliest-dated entry, since the metadata is sorted
    /// by date).
    #[pyo3(signature = (refseq, input_fasta))]
    fn load_reference<'py>(
        slf: &Bound<'py, Self>,
        refseq: Option<&str>,
        input_fasta: &str,
    ) -> PyResult<()> {
        let py = slf.py();
        let reference: SequenceRecord = match refseq {
            Some(path) => fasta::first_record(path)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Reference FASTA file {} contains no sequences",
                        path
                    ))
                })?
                .into(),
            None => {
                let first_id: String = Self::with_table(slf, |_, t| {
                    if t.nrows() == 0 {
                        return Err(PyIndexError::new_err(
                            "single positional indexer is out-of-bounds",
                        ));
                    }
                    match t.column("id") {
                        Some(Column::Str(d)) => Ok(d.get(0).unwrap_or("").to_string()),
                        Some(Column::Int64(v)) => Ok(v[0].to_string()),
                        Some(_) => Err(PyValueError::new_err("metadata `id` column must contain strings")),
                        None => Err(PyValueError::new_err("Metadata file must contain an \"id\" column")),
                    }
                })?;
                fasta::find_by_id(input_fasta, &first_id)?
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Reference sequence '{}' (first metadata entry) was not found in {}",
                            first_id, input_fasta
                        ))
                    })?
                    .into()
            }
        };
        slf.setattr("reference", Py::new(py, reference)?)?;
        Ok(())
    }

    // parse_data --------------------------------------------------------
    fn parse_data<'py>(
        slf: &Bound<'py, Self>,
        input_fasta: &str,
        selection: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        // Dispatched through Python so overrides of get_differing_mutations
        // are honoured; the result is a small DataFrame (id, instructions,
        // N count) that is ingested and left-merged onto the table.
        let muts = slf.call_method1("get_differing_mutations", (input_fasta, selection))?;
        let right = Table::from_pandas(py, &muts)?;
        Self::with_table(slf, |py, t| {
            *t = left_merge_on_id(py, t, &right)?;
            drop_missing_instructions_table(py, t)
        })
    }

    // filter_by_daterange -----------------------------------------------
    /// Keep rows with `start <= date <= end`, each bound clamped to the data's
    /// own range (a bound outside the data is ignored). `None` leaves a bound
    /// open. Raises if the effective start is after the effective end.
    fn filter_by_daterange<'py>(
        slf: &Bound<'py, Self>,
        start: Option<Bound<'py, PyAny>>,
        end: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;
        let to_ns = |obj: &Bound<'py, PyAny>| -> PyResult<i64> {
            pd.call_method1("Timestamp", (obj,))?.getattr("value")?.extract()
        };
        let start_ns = match &start {
            Some(s) if !s.is_none() => Some(to_ns(s)?),
            _ => None,
        };
        let end_ns = match &end {
            Some(e) if !e.is_none() => Some(to_ns(e)?),
            _ => None,
        };
        Self::with_table(slf, |py, t| {
            let ns: Vec<i64> = match t.column("date") {
                Some(Column::DatetimeNs { ns, .. }) => ns.clone(),
                Some(other) => dates::to_datetime_ns(py, other)?,
                None => return Err(PyValueError::new_err("KeyError: 'date'")),
            };
            let valid = ns.iter().copied().filter(|&v| v != table::NAT);
            let (dmin, dmax) = match (valid.clone().min(), valid.max()) {
                (Some(a), Some(b)) => (a, b),
                // No dates at all (empty data): pandas' NaT comparisons are all
                // False, so nothing is kept and no error is raised.
                _ => {
                    *t = t.take(py, &[]);
                    return Ok(());
                }
            };
            let start_v = match start_ns {
                Some(s) if s > dmin => s,
                _ => dmin,
            };
            let end_v = match end_ns {
                Some(e) if e < dmax => e,
                _ => dmax,
            };
            if start_v > end_v {
                return Err(PyValueError::new_err(
                    "Start date must be smaller than end date",
                ));
            }
            let rows: Vec<usize> = (0..ns.len())
                .filter(|&i| ns[i] != table::NAT && start_v <= ns[i] && ns[i] <= end_v)
                .collect();
            *t = t.take(py, &rows);
            Ok(())
        })
    }

    // filter_by_position ------------------------------------------------
    /// Restrict mutations to a genome window. Window membership on 1-based
    /// positions: substitutions `start <= P < end`, insertions/deletions
    /// `start < P <= end` (the indel rule reproduces the counting behaviour
    /// of every release before 0.2.0, when indel positions were written
    /// 0-based; a deletion starting at the first base stays excluded and an
    /// insertion after the last base stays included). Rows whose mutations
    /// all fall outside the window are dropped; rows that had no mutations
    /// are kept. `end <= 0` means "to the end of the reference".
    fn filter_by_position<'py>(
        slf: &Bound<'py, Self>,
        start: i64,
        end: i64,
    ) -> PyResult<()> {
        // Early return on an already-empty dataset so the caller's
        // _check_dataset_is_not_empty surfaces the friendly message.
        if Self::with_table(slf, |_, t| Ok(t.is_empty()))? {
            return Ok(());
        }
        let ref_seq_len = Self::reference_len(slf)?;
        let start = start.max(1);
        let end_eff = if end > 0 { end } else { ref_seq_len + 1 };
        if start >= end_eff {
            return Err(PyValueError::new_err(
                "Start position must be smaller than end position",
            ));
        }
        if start > ref_seq_len {
            return Err(PyValueError::new_err("Start position is out of range"));
        }

        let in_window = |m: &str| -> bool {
            let pos: i64 = m
                .split('_')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if m.starts_with('s') {
                start <= pos && pos < end_eff
            } else {
                start < pos && pos <= end_eff
            }
        };

        Self::with_table(slf, |py, t| {
            let lists = match t.column("mutation instructions") {
                Some(Column::StrList(v)) => v,
                Some(_) => return Err(PyValueError::new_err(
                    "`mutation instructions` must be a column of lists of strings",
                )),
                None => return Err(PyValueError::new_err("KeyError: 'mutation instructions'")),
            };
            let mut keep: Vec<usize> = Vec::with_capacity(lists.len());
            let mut new_lists: Vec<Option<Vec<String>>> = Vec::with_capacity(lists.len());
            for (i, item) in lists.iter().enumerate() {
                match item {
                    // Missing cells pass through untouched (they are dropped
                    // with a warning elsewhere).
                    None => {
                        keep.push(i);
                        new_lists.push(None);
                    }
                    Some(x) if x.is_empty() => {
                        keep.push(i);
                        new_lists.push(Some(Vec::new()));
                    }
                    Some(x) => {
                        let filtered: Vec<String> =
                            x.iter().filter(|m| in_window(m)).cloned().collect();
                        if !filtered.is_empty() {
                            keep.push(i);
                            new_lists.push(Some(filtered));
                        }
                    }
                }
            }
            *t = t.take(py, &keep);
            t.set_column("mutation instructions", Column::StrList(new_lists));
            Ok(())
        })
    }

    // filter_columns ----------------------------------------------------
    /// Keep rows whose value in each filter key matches the given value(s),
    /// interpreted as regular expressions with `*` meaning "anything"
    /// (pandas `str.contains(regex=True)` semantics). Keys that are not
    /// columns are ignored.
    fn filter_columns<'py>(
        slf: &Bound<'py, Self>,
        filters: Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let mut specs: Vec<(String, String)> = Vec::new();
        for (k, v) in filters.iter() {
            let key: String = k.extract()?;
            let vals: Vec<String> = if let Ok(s) = v.extract::<String>() {
                vec![s]
            } else {
                v.extract()?
            };
            let pattern = vals
                .iter()
                .map(|val| val.replace('*', ".*"))
                .collect::<Vec<_>>()
                .join("|");
            specs.push((key, pattern));
        }
        Self::with_table(slf, |py, t| {
            for (key, pattern) in &specs {
                let rows: Vec<usize> = match t.column(key) {
                    None => continue,
                    Some(Column::Str(d)) => {
                        if d.iter().any(|c| c.is_none()) {
                            return Err(PyValueError::new_err(
                                "Cannot mask with non-boolean array containing NA / NaN values",
                            ));
                        }
                        let matcher = Matcher::new(py, pattern)?;
                        let mut rows = Vec::new();
                        for i in 0..d.len() {
                            if matcher.is_match(py, d.get(i).unwrap_or(""))? {
                                rows.push(i);
                            }
                        }
                        rows
                    }
                    Some(_) => {
                        return Err(PyAttributeError::new_err(
                            "Can only use .str accessor with string values!",
                        ))
                    }
                };
                *t = t.take(py, &rows);
            }
            Ok(())
        })
    }

    // _get_consecutives -------------------------------------------------
    #[staticmethod]
    fn _get_consecutives(data: Vec<usize>) -> Vec<Vec<usize>> {
        get_consecutives(&data)
    }

    // _column_decision --------------------------------------------------
    #[staticmethod]
    fn _column_decision<'py>(col: Bound<'py, PyAny>) -> PyResult<u8> {
        // Accept anything indexable (numpy array, list, tuple) returning
        // single-character strings at [0] and [1].
        let v0_str: String = col.call_method1("__getitem__", (0i64,))?.extract()?;
        let v1_str: String = col.call_method1("__getitem__", (1i64,))?.extract()?;
        let v0 = v0_str.chars().next().unwrap_or(' ');
        let v1 = v1_str.chars().next().unwrap_or(' ');
        Ok(column_decision(v0, v1))
    }

    // create_modifs -----------------------------------------------------
    #[classmethod]
    fn create_modifs<'py>(
        _cls: &Bound<'py, PyType>,
        alignment: Bound<'py, PyAny>,
    ) -> PyResult<Vec<String>> {
        // `alignment` is whatever generate_alignment returned: a list of two
        // SequenceRecords by default, but any indexable of objects with a
        // `.seq` attribute works (e.g. a Biopython alignment).
        let rec0 = alignment.call_method1("__getitem__", (0i64,))?;
        let rec1 = alignment.call_method1("__getitem__", (1i64,))?;
        let seq0: String = rec0.getattr("seq")?.call_method0("__str__")?.extract()?;
        let seq1: String = rec1.getattr("seq")?.call_method0("__str__")?.extract()?;
        Ok(create_modifs_from_strings(
            &seq0.to_ascii_uppercase(),
            &seq1.to_ascii_uppercase(),
        ))
    }

    // get_differing_mutations -------------------------------------------
    fn get_differing_mutations<'py>(
        slf: &Bound<'py, Self>,
        input_fasta: &str,
        selection: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;

        let selection_values = selection.getattr("values")?;
        let selection_list: Vec<String> =
            selection_values.call_method0("tolist")?.extract()?;
        let selection_set: HashSet<String> = selection_list.into_iter().collect();
        let total = selection_set.len();
        let verbose = verbosity(slf.as_any());

        let cls_bound = slf.getattr("__class__")?;
        let reference = slf.getattr("reference")?;

        let mut ids: Vec<String> = Vec::new();
        let mut all_modifs: Vec<Vec<String>> = Vec::new();
        let mut n_counts: Vec<i64> = Vec::new();

        let mut reader = fasta::open(input_fasta)?;
        loop {
            // Every selected id has been seen: no need to scan the rest of
            // the (possibly huge) FASTA file.
            if ids.len() >= total {
                break;
            }
            let raw = match reader.next_record().map_err(|e| {
                PyValueError::new_err(format!("Error reading FASTA file {}: {}", input_fasta, e))
            })? {
                Some(r) => r,
                None => break,
            };
            if !selection_set.contains(&raw.id) {
                continue;
            }
            let rec_id = raw.id.clone();
            let rec = Py::new(py, SequenceRecord::from(raw))?;
            // Dispatch generate_alignment + create_modifs through the
            // Python class so subclass overrides are honoured.
            let alignment = cls_bound
                .call_method1("generate_alignment", (reference.clone(), rec))?;
            let modifs: Vec<String> = cls_bound
                .call_method1("create_modifs", (alignment.clone(),))?
                .extract()?;
            // N count: lowercase 'n' in the aligned target (the aligner emits
            // lowercase residues, as C MAFFT does).
            let aligned_target: String = alignment
                .call_method1("__getitem__", (1i64,))?
                .getattr("seq")?
                .call_method0("__str__")?
                .extract()?;
            let n_count = aligned_target.chars().filter(|&c| c == 'n').count() as i64;

            ids.push(rec_id);
            all_modifs.push(modifs);
            n_counts.push(n_count);

            if verbose > 0 {
                eprint!("\rAligning sequences: {}/{}", ids.len(), total);
            }
        }
        if verbose > 0 {
            eprintln!();
        }

        // Build [(id, modifs, n_count), ...] and feed into pd.DataFrame.
        let rows = PyList::empty_bound(py);
        for ((id, modifs), nc) in ids.into_iter().zip(all_modifs).zip(n_counts) {
            let row = PyTuple::new_bound(
                py,
                &[id.into_py(py), modifs.into_py(py), nc.into_py(py)],
            );
            rows.append(row)?;
        }
        let kw = PyDict::new_bound(py);
        kw.set_item(
            "columns",
            PyList::new_bound(py, ["id", "mutation instructions", "N count"]),
        )?;
        Ok(pd.call_method("DataFrame", (rows,), Some(&kw))?)
    }

    // generate_alignment ------------------------------------------------
    /// Align ``seq1`` (reference) against ``seq2`` and return the two
    /// aligned records as a list ``[reference, target]`` of
    /// :class:`SequenceRecord` (residues lower-case, gaps as ``-``).
    #[classmethod]
    fn generate_alignment<'py>(
        cls: &Bound<'py, PyType>,
        seq1: Bound<'py, PyAny>,
        seq2: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = cls.py();
        let rec1 = fasta::record_from_any(&seq1)?;
        let rec2 = fasta::record_from_any(&seq2)?;
        let mut id_1 = rec1.id.clone();
        if id_1 == rec2.id {
            id_1.push_str("_ref");
        }
        let seqs_dict = PyDict::new_bound(py);
        seqs_dict.set_item(&id_1, &rec1.seq)?;
        seqs_dict.set_item(&rec2.id, &rec2.seq)?;
        // Dispatch _run_mafft via cls so subclass overrides are honoured.
        let mafft_out: String = cls.call_method1("_run_mafft", (seqs_dict,))?.extract()?;
        let records = fasta::parse_str(&mafft_out);
        if records.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected 2 aligned sequences from the aligner, got {}",
                records.len()
            )));
        }
        let out = PyList::empty_bound(py);
        for raw in records {
            out.append(Py::new(py, SequenceRecord::from(raw))?)?;
        }
        Ok(out)
    }

    // read_fasta --------------------------------------------------------
    /// Lazily iterate over the records of a FASTA file.
    #[staticmethod]
    fn read_fasta(input_fasta: &str) -> PyResult<FastaReader> {
        FastaReader::new(input_fasta)
    }

    // parse_sequence_by_id ----------------------------------------------
    /// The record with id ``_id``, or ``None`` if the file has none.
    #[staticmethod]
    fn parse_sequence_by_id(input_fasta: &str, _id: &str) -> PyResult<Option<SequenceRecord>> {
        Ok(fasta::find_by_id(input_fasta, _id)?.map(SequenceRecord::from))
    }

    // _run_mafft --------------------------------------------------------
    #[staticmethod]
    #[pyo3(signature = (seqs_dict, outformat="fasta"))]
    fn _run_mafft<'py>(
        py: Python<'py>,
        seqs_dict: Bound<'py, PyDict>,
        outformat: &str,
    ) -> PyResult<String> {
        let mut seqs: Vec<(String, String)> = Vec::new();
        for (k, v) in seqs_dict.iter() {
            let name: String = k.extract()?;
            let seq_str: String = if let Ok(s) = v.extract::<String>() {
                s
            } else {
                // Any other sequence-like object — coerce via str().
                v.call_method0("__str__")?.extract()?
            };
            seqs.push((name, seq_str));
        }
        run_mafft_inner(py, &seqs, outformat)
    }

    // parse_mutation_data -----------------------------------------------
    /// Read a data TSV written by a previous run (``<out>.tsv``: the metadata
    /// columns plus ``mutation instructions``, ``N count`` and the mutation
    /// counts). The ``mutation instructions`` cells are Python list literals
    /// and are parsed back into lists; empty cells become None (dropped by
    /// the caller with a warning). Dates are parsed and rows sorted by date
    /// (stable), like parse_metadata.
    #[staticmethod]
    fn parse_mutation_data<'py>(
        py: Python<'py>,
        input_tsv: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_mutation_data_inner(py, input_tsv)?.to_pandas(py)
    }

    // parse_metadata ----------------------------------------------------
    /// Read a metadata CSV/TSV, parse its ``date`` column and sort by date
    /// (stable; rows sharing a date keep their file order).
    #[staticmethod]
    fn parse_metadata<'py>(
        py: Python<'py>,
        input_meta: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_metadata_inner(py, input_meta)?.to_pandas(py)
    }
}

impl PyEvoMotionParser {
    /// Replace the dataset with a Rust table (Rust-owned state; the next
    /// `.data` access materialises a fresh DataFrame).
    pub(crate) fn store_table(slf: &Bound<'_, Self>, table: Table) {
        let mut this = slf.borrow_mut();
        this.table = Some(table);
        this.df = None;
    }

    /// True when a DataFrame has been handed out or assigned since the last
    /// Rust stage (pandas-visible state).
    pub(crate) fn is_pandas_visible(slf: &Bound<'_, Self>) -> bool {
        slf.borrow().df.is_some()
    }

    /// Move the dataset out as a Table, ingesting the pandas-visible
    /// DataFrame first if there is one (it may have been edited in place).
    fn take_table(slf: &Bound<'_, Self>) -> PyResult<Table> {
        let py = slf.py();
        let (df, table) = {
            let mut this = slf.borrow_mut();
            (this.df.take(), this.table.take())
        };
        if let Some(df) = df {
            return Table::from_pandas(py, df.bind(py));
        }
        match table {
            Some(t) => Ok(t),
            None => Err(PyAttributeError::new_err(format!(
                "'{}' object has no attribute 'data'",
                slf.get_type().name()?
            ))),
        }
    }

    /// Run `f` on the dataset as a Table and store the result back
    /// (Rust-owned state). `f` may edit the table in place or replace it.
    pub(crate) fn with_table<R>(
        slf: &Bound<'_, Self>,
        f: impl FnOnce(Python<'_>, &mut Table) -> PyResult<R>,
    ) -> PyResult<R> {
        let py = slf.py();
        let mut table = Self::take_table(slf)?;
        let result = f(py, &mut table);
        Self::store_table(slf, table);
        result
    }

    /// Length of the reference sequence (`self.reference.seq`).
    pub(crate) fn reference_len(slf: &Bound<'_, Self>) -> PyResult<i64> {
        Ok(slf.getattr("reference")?.getattr("seq")?.len()? as i64)
    }
}

/// A compiled filter pattern: the `regex` crate when it accepts the pattern,
/// Python's `re` otherwise (lookaround, backreferences, ...), so semantics
/// match pandas' `str.contains(regex=True)` in both cases.
enum Matcher {
    Rust(regex::Regex),
    Python(Py<PyAny>),
}

impl Matcher {
    fn new(py: Python<'_>, pattern: &str) -> PyResult<Self> {
        match regex::Regex::new(pattern) {
            Ok(re) => Ok(Matcher::Rust(re)),
            Err(_) => {
                let re = py.import_bound("re")?.call_method1("compile", (pattern,))?;
                Ok(Matcher::Python(re.unbind()))
            }
        }
    }

    fn is_match(&self, py: Python<'_>, s: &str) -> PyResult<bool> {
        match self {
            Matcher::Rust(re) => Ok(re.is_match(s)),
            Matcher::Python(re) => Ok(!re.bind(py).call_method1("search", (s,))?.is_none()),
        }
    }
}

/// pandas `merge(right, on="id", how="left")` on tables whose `id` column is
/// a string column: left order kept, every right match emitted (duplicates
/// multiply rows), unmatched left rows get missing values. Integer columns
/// that receive a missing value become float64, as in pandas.
fn left_merge_on_id(py: Python<'_>, left: &Table, right: &Table) -> PyResult<Table> {
    let left_ids = match left.column("id") {
        Some(Column::Str(d)) => d,
        _ => return Err(PyValueError::new_err("left table has no string `id` column")),
    };
    let right_ids = match right.column("id") {
        Some(Column::Str(d)) => d,
        _ => return Err(PyValueError::new_err("right table has no string `id` column")),
    };
    let mut positions: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, id) in right_ids.iter().enumerate() {
        if let Some(id) = id {
            positions.entry(id).or_default().push(i);
        }
    }

    // (left row, Some(right row) | None) pairs in output order.
    let mut pairs: Vec<(usize, Option<usize>)> = Vec::with_capacity(left.nrows());
    for (i, id) in left_ids.iter().enumerate() {
        match id.and_then(|s| positions.get(s)) {
            Some(rs) => pairs.extend(rs.iter().map(|&r| (i, Some(r)))),
            None => pairs.push((i, None)),
        }
    }
    let left_rows: Vec<usize> = pairs.iter().map(|(l, _)| *l).collect();
    let any_unmatched = pairs.iter().any(|(_, r)| r.is_none());

    let mut out = left.take(py, &left_rows);
    out.reset_index(); // pandas merge returns a RangeIndex
    for (name, col) in &right.columns {
        if name == "id" {
            continue;
        }
        let picked: Column = match col {
            Column::StrList(v) => Column::StrList(
                pairs.iter().map(|(_, r)| r.and_then(|r| v[r].clone())).collect(),
            ),
            Column::Int64(v) => {
                if any_unmatched {
                    Column::Float64(
                        pairs
                            .iter()
                            .map(|(_, r)| r.map(|r| v[r] as f64).unwrap_or(f64::NAN))
                            .collect(),
                    )
                } else {
                    Column::Int64(pairs.iter().map(|(_, r)| v[r.unwrap()]).collect())
                }
            }
            Column::Float64(v) => Column::Float64(
                pairs.iter().map(|(_, r)| r.map(|r| v[r]).unwrap_or(f64::NAN)).collect(),
            ),
            Column::Str(d) => Column::Str(table::DictStr::from_options(
                pairs.iter().map(|(_, r)| r.and_then(|r| d.get(r).map(str::to_string))),
            )),
            Column::DatetimeNs { ns, unit } => Column::DatetimeNs {
                ns: pairs.iter().map(|(_, r)| r.map(|r| ns[r]).unwrap_or(table::NAT)).collect(),
                unit: *unit,
            },
            other => {
                // Not produced by get_differing_mutations; select what we can.
                let rows: Vec<usize> = pairs.iter().map(|(_, r)| r.unwrap_or(0)).collect();
                other.take(py, &rows)
            }
        };
        out.set_column(name, picked);
    }
    Ok(out)
}

/// Rust side of drop_missing_instructions: rows whose "mutation
/// instructions" cell is missing are reported on stderr and removed; the
/// index is reset only when something was dropped (pandas parity).
fn drop_missing_instructions_table(py: Python<'_>, table: &mut Table) -> PyResult<()> {
    let missing: Vec<bool> = match table.column("mutation instructions") {
        Some(Column::StrList(v)) => v.iter().map(|x| x.is_none()).collect(),
        Some(Column::Float64(v)) => v.iter().map(|x| x.is_nan()).collect(),
        Some(_) => vec![false; table.nrows()],
        None => return Err(PyValueError::new_err("KeyError: 'mutation instructions'")),
    };
    let n_missing = missing.iter().filter(|&&m| m).count();
    if n_missing == 0 {
        return Ok(());
    }
    let ids: Vec<String> = match table.column("id") {
        Some(Column::Str(d)) => (0..d.len())
            .filter(|&i| missing[i])
            .map(|i| d.get(i).unwrap_or("").to_string())
            .collect(),
        _ => Vec::new(),
    };
    let shown: Vec<&str> = ids.iter().take(10).map(String::as_str).collect();
    eprintln!(
        "Warning: {} sequence(s) have no mutation instructions (id present in the metadata but not in the FASTA file, or empty cell) and will be excluded from the analysis.",
        n_missing
    );
    eprintln!(
        "         Example ids: {}{}",
        shown.join(", "),
        if ids.len() > shown.len() { ", ..." } else { "" }
    );
    let keep: Vec<usize> = (0..table.nrows()).filter(|&i| !missing[i]).collect();
    *table = table.take(py, &keep);
    table.reset_index();
    Ok(())
}

// ─────────────────────── module-private helpers ───────────────────────

fn parse_metadata_inner(py: Python<'_>, input_meta: &str) -> PyResult<Table> {
    let sep = if input_meta.ends_with(".csv") {
        ','
    } else if input_meta.ends_with(".tsv") {
        '\t'
    } else {
        // The original silently swallows unknown extensions and returns
        // None. Match that by raising a clean Python error instead — it
        // produces a useful message rather than a downstream None.
        return Err(PyValueError::new_err(format!(
            "Unsupported metadata extension for {}: expected .csv or .tsv",
            input_meta
        )));
    };
    let mut table = csv_read::read_table(py, input_meta, sep)?;
    if !table.has_column("date") {
        return Err(PyValueError::new_err(
            "Metadata file must contain a \"date\" column",
        ));
    }
    parse_date_column(py, &mut table)?;
    // Canonical order: stable sort by date (ties keep file order), labels kept.
    let ns = match table.column("date") {
        Some(Column::DatetimeNs { ns, .. }) => ns.clone(),
        _ => unreachable!("date column was just converted"),
    };
    let order = dates::stable_argsort(&ns);
    Ok(table.take(py, &order))
}

/// Convert the ``date`` column in place to datetime64[ns].
fn parse_date_column(py: Python<'_>, table: &mut Table) -> PyResult<()> {
    let ns = {
        let col = table.column("date").expect("caller checked the column exists");
        dates::to_datetime_ns(py, col)?
    };
    table.set_column("date", Column::DatetimeNs { ns, unit: TimeUnit::Ns });
    Ok(())
}

/// Rust side of `parse_mutation_data`: a ``<out>.tsv`` back into a table with
/// list-valued "mutation instructions", parsed dates, stable date order and a
/// fresh RangeIndex.
pub(crate) fn parse_mutation_data_inner(py: Python<'_>, input_tsv: &str) -> PyResult<Table> {
    let mut table = csv_read::read_table(py, input_tsv, '\t')?;
    for required in ["id", "date", "mutation instructions"] {
        if !table.has_column(required) {
            return Err(PyValueError::new_err(format!(
                "Mutation instructions file {} must contain a \"{}\" column (expected a <out>.tsv written by PyEvoMotion)",
                input_tsv, required
            )));
        }
    }

    // ids for error messages
    let ids: Vec<String> = match table.column("id") {
        Some(Column::Str(d)) => d.iter().map(|s| s.unwrap_or("").to_string()).collect(),
        Some(other) => (0..other.len()).map(|i| i.to_string()).collect(),
        None => unreachable!(),
    };

    let parsed: Vec<Option<Vec<String>>> = match table.column("mutation instructions") {
        Some(Column::Str(d)) => {
            let mut out = Vec::with_capacity(d.len());
            for (i, cell) in d.iter().enumerate() {
                match cell {
                    None => out.push(None),
                    Some(text) => match csv_read::parse_str_list_literal(text) {
                        Ok(list) => out.push(Some(list)),
                        Err(e) => {
                            return Err(PyValueError::new_err(format!(
                                "Could not parse the mutation instructions of '{}' in {}: {}",
                                ids[i], input_tsv, e
                            )))
                        }
                    },
                }
            }
            out
        }
        // An entirely empty column is read as float64 (all missing).
        Some(Column::Float64(v)) if v.iter().all(|x| x.is_nan()) => vec![None; v.len()],
        Some(_) => {
            return Err(PyValueError::new_err(format!(
                "Mutation instructions in {} are not list literals",
                input_tsv
            )))
        }
        None => unreachable!(),
    };
    table.set_column("mutation instructions", Column::StrList(parsed));

    parse_date_column(py, &mut table)?;
    let ns = match table.column("date") {
        Some(Column::DatetimeNs { ns, .. }) => ns.clone(),
        _ => unreachable!(),
    };
    let order = dates::stable_argsort(&ns);
    let mut sorted = table.take(py, &order);
    sorted.reset_index();
    Ok(sorted)
}
