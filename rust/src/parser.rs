use std::collections::HashSet;

use mafft::{AlignmentMode, MafftEngine, Sequence, SequenceSet, SeqType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple, PyType};

use crate::base::PyEvoMotionBase;
use crate::fasta::{self, FastaReader, SequenceRecord};

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

/// Drop rows whose "mutation instructions" cell is missing (NaN). This
/// happens when a metadata id has no sequence in the FASTA file (the merge
/// in parse_data is a left join) or when a loaded TSV has empty cells.
/// Downstream code extracts that column as lists of strings, so leaving
/// NaN in place would fail with an opaque TypeError; instead warn on stderr
/// with the count and a few example ids, and continue with the rest.
fn drop_missing_instructions<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let col = data.call_method1("__getitem__", ("mutation instructions",))?;
    let missing = col.call_method0("isna")?;
    let n_missing: i64 = missing.call_method0("sum")?.extract()?;
    if n_missing == 0 {
        return Ok(data.clone());
    }
    let ids: Vec<String> = data
        .call_method1("__getitem__", (missing.clone(),))?
        .call_method1("__getitem__", ("id",))?
        .call_method0("tolist")?
        .extract()?;
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
    let keep = missing.call_method0("__invert__")?;
    let kept = data.call_method1("__getitem__", (keep,))?;
    let kw = PyDict::new_bound(py);
    kw.set_item("drop", true)?;
    kept.call_method("reset_index", (), Some(&kw))
}

// ─────────────────────── pyclass ───────────────────────

// PyEvoMotionParser extends PyEvoMotionBase so that the user-facing
// PyEvoMotion(_PyEvoMotionCore, PyEvoMotionParser) multi-inheritance has
// a shared layout root (both bases trace back to PyEvoMotionBase).
#[pyclass(subclass, extends = PyEvoMotionBase, name = "PyEvoMotionParser", module = "PyEvoMotion")]
pub struct PyEvoMotionParser;

#[pymethods]
impl PyEvoMotionParser {
    // __new__: cooperative no-op so multi-inheritance instantiation works.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> (Self, PyEvoMotionBase) {
        (Self, PyEvoMotionBase)
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
                let data = slf.call_method1("parse_mutation_data", (path,))?;
                slf.setattr("data", drop_missing_instructions(py, &data)?)?;
            }
            None => {
                log_info(py, "Parsing metadata file")?;
                let data = parse_metadata_inner(py, input_meta)?;
                slf.setattr("data", data)?;
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
            let data_attr = slf.getattr("data")?;
            let ids = data_attr.call_method1("__getitem__", ("id",))?;
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
                let data_attr = slf.getattr("data")?;
                let iloc = data_attr.getattr("iloc")?;
                let first_row = iloc.call_method1("__getitem__", (0i64,))?;
                let first_id: String =
                    first_row.call_method1("__getitem__", ("id",))?.extract()?;
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
        let muts = slf.call_method1("get_differing_mutations", (input_fasta, selection))?;
        let data = slf.getattr("data")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("on", "id")?;
        kw.set_item("how", "left")?;
        let merged = data.call_method("merge", (muts,), Some(&kw))?;
        let kw = PyDict::new_bound(py);
        kw.set_item("drop", true)?;
        let reset = merged.call_method("reset_index", (), Some(&kw))?;
        let complete = drop_missing_instructions(py, &reset)?;
        slf.setattr("data", complete)?;
        Ok(())
    }

    // filter_by_daterange -----------------------------------------------
    fn filter_by_daterange<'py>(
        slf: &Bound<'py, Self>,
        start: Option<Bound<'py, PyAny>>,
        end: Option<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        let data = slf.getattr("data")?;
        let dates = data.call_method1("__getitem__", ("date",))?;
        let dmin = dates.call_method0("min")?;
        let dmax = dates.call_method0("max")?;

        let start_v = match start {
            Some(s) if !s.is_none() => {
                let cmp: bool = s.gt(&dmin)?;
                if cmp { s } else { dmin }
            }
            _ => dmin,
        };
        let end_v = match end {
            Some(e) if !e.is_none() => {
                let cmp: bool = e.lt(&dmax)?;
                if cmp { e } else { dmax }
            }
            _ => dmax,
        };

        if start_v.gt(&end_v)? {
            return Err(PyValueError::new_err(
                "Start date must be smaller than end date",
            ));
        }

        let mask_lo = dates.call_method1("__ge__", (start_v,))?;
        let mask_hi = dates.call_method1("__le__", (end_v,))?;
        let mask = mask_lo.call_method1("__and__", (mask_hi,))?;
        let filtered = data.call_method1("__getitem__", (mask,))?;
        slf.setattr("data", filtered)?;
        Ok(())
    }

    // filter_by_position ------------------------------------------------
    fn filter_by_position<'py>(
        slf: &Bound<'py, Self>,
        start: i64,
        end: i64,
    ) -> PyResult<()> {
        let py = slf.py();
        let pd = py.import_bound("pandas")?;

        // If an earlier filter already emptied the dataset, there is nothing to
        // position-filter. Return early so the caller's
        // _check_dataset_is_not_empty surfaces the friendly "dataset empty"
        // message rather than a confusing KeyError from the column rebuild on
        // an empty frame.
        if slf.getattr("data")?.getattr("empty")?.extract::<bool>()? {
            return Ok(());
        }

        let reference = slf.getattr("reference")?;
        let ref_seq_len: i64 = reference
            .getattr("seq")?
            .call_method0("__len__")?
            .extract()?;

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

        // Window membership, on 1-based positions.
        //   substitutions:            start <= pos <  end
        //   insertions and deletions: start <  pos <= end
        // The indel rule reproduces the counting behaviour of every release
        // before 0.2.0, where indel positions were written 0-based but went
        // through the substitution test above. Keeping it means the switch
        // to 1-based coordinates only changes how positions are *written*,
        // not which mutations are counted: a deletion that starts at the
        // first reference base (a sequence lacking 5' coverage) stays
        // excluded, an insertion after the last reference base stays
        // included. Whether terminal indels should be counted at all is an
        // open question (see luksgrin/PyEvoMotion#1); revisit both together.
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

        // Build the new "mutation instructions" column row-by-row in Rust,
        // then assign as a list (length-aligned with self.data).
        let data = slf.getattr("data")?;
        let mut_instr = data.call_method1("__getitem__", ("mutation instructions",))?;
        let lists: Vec<Vec<String>> = mut_instr.call_method0("tolist")?.extract()?;

        let new_lists: Vec<Vec<String>> = lists
            .iter()
            .map(|x| {
                if x.is_empty() {
                    vec!["NO_MUTATION".to_string()]
                } else {
                    x.iter()
                        .filter(|m| in_window(m))
                        .cloned()
                        .collect::<Vec<_>>()
                }
            })
            .collect();

        // Preserve the existing index when assigning back.
        let idx = data.getattr("index")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("index", idx.clone())?;
        let new_series = pd.call_method("Series", (new_lists.clone(),), Some(&kw))?;
        data.call_method1("__setitem__", ("mutation instructions", new_series))?;

        // Filter to rows whose new mutation list is non-empty.
        let mask: Vec<bool> = new_lists.iter().map(|x| !x.is_empty()).collect();
        let kw = PyDict::new_bound(py);
        kw.set_item("index", idx)?;
        let mask_series = pd.call_method("Series", (mask,), Some(&kw))?;
        let filtered = data.call_method1("__getitem__", (mask_series,))?;
        slf.setattr("data", filtered.clone())?;

        // Replace ["NO_MUTATION"] with []
        let mut_instr2 = filtered.call_method1("__getitem__", ("mutation instructions",))?;
        let lists2: Vec<Vec<String>> = mut_instr2.call_method0("tolist")?.extract()?;
        let final_lists: Vec<Vec<String>> = lists2
            .into_iter()
            .map(|x| {
                if x.len() == 1 && x[0] == "NO_MUTATION" {
                    Vec::new()
                } else {
                    x
                }
            })
            .collect();
        let idx2 = filtered.getattr("index")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("index", idx2)?;
        let final_series = pd.call_method("Series", (final_lists,), Some(&kw))?;
        filtered.call_method1("__setitem__", ("mutation instructions", final_series))?;

        Ok(())
    }

    // filter_columns ----------------------------------------------------
    fn filter_columns<'py>(
        slf: &Bound<'py, Self>,
        filters: Bound<'py, PyDict>,
    ) -> PyResult<()> {
        let py = slf.py();
        let mut current = slf.getattr("data")?;
        let columns: Vec<String> = current
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        let cols_set: HashSet<String> = columns.into_iter().collect();

        for (k, v) in filters.iter() {
            let key: String = k.extract()?;
            if !cols_set.contains(&key) {
                continue;
            }
            let vals: Vec<String> = if let Ok(s) = v.extract::<String>() {
                vec![s]
            } else {
                v.extract()?
            };
            let regex_pattern = vals
                .iter()
                .map(|val| val.replace('*', ".*"))
                .collect::<Vec<_>>()
                .join("|");

            let col_series = current.call_method1("__getitem__", (&key,))?;
            let str_acc = col_series.getattr("str")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("regex", true)?;
            let mask = str_acc.call_method("contains", (regex_pattern,), Some(&kw))?;
            current = current.call_method1("__getitem__", (mask,))?;
        }
        slf.setattr("data", current)?;
        Ok(())
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
    /// the caller with a warning). Dates are parsed and rows sorted by date,
    /// like parse_metadata.
    #[staticmethod]
    fn parse_mutation_data<'py>(
        py: Python<'py>,
        input_tsv: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import_bound("pandas")?;
        let ast = py.import_bound("ast")?;
        let kw = PyDict::new_bound(py);
        kw.set_item("sep", "\t")?;
        let df = pd.call_method("read_csv", (input_tsv,), Some(&kw))?;

        let columns: Vec<String> = df.getattr("columns")?.call_method0("tolist")?.extract()?;
        for required in ["id", "date", "mutation instructions"] {
            if !columns.iter().any(|c| c == required) {
                return Err(PyValueError::new_err(format!(
                    "Mutation instructions file {} must contain a \"{}\" column (expected a <out>.tsv written by PyEvoMotion)",
                    input_tsv, required
                )));
            }
        }

        let ids: Vec<String> = df
            .call_method1("__getitem__", ("id",))?
            .call_method0("tolist")?
            .extract()?;
        let cells = df
            .call_method1("__getitem__", ("mutation instructions",))?
            .call_method0("tolist")?;
        let parsed = PyList::empty_bound(py);
        for (cell, id) in cells.iter()?.zip(ids.iter()) {
            let cell = cell?;
            match cell.extract::<String>() {
                Ok(text) => {
                    let value = ast
                        .call_method1("literal_eval", (text.trim(),))
                        .map_err(|e| {
                            PyValueError::new_err(format!(
                                "Could not parse the mutation instructions of '{}' in {}: {}",
                                id, input_tsv, e
                            ))
                        })?;
                    let items: Vec<String> = value.extract().map_err(|_| {
                        PyValueError::new_err(format!(
                            "Mutation instructions of '{}' in {} are not a list of strings",
                            id, input_tsv
                        ))
                    })?;
                    parsed.append(items)?;
                }
                // NaN (empty cell) → None; the caller drops these rows.
                Err(_) => parsed.append(py.None())?,
            }
        }
        df.call_method1("__setitem__", ("mutation instructions", parsed))?;

        let dates_col = df.call_method1("__getitem__", ("date",))?;
        let dates = pd.call_method1("to_datetime", (dates_col,))?;
        df.call_method1("__setitem__", ("date", dates))?;
        // Stable sort: a TSV written by PyEvoMotion is already date-ordered,
        // and rows sharing a date must keep their order so a reload
        // reproduces the original run exactly.
        let kw = PyDict::new_bound(py);
        kw.set_item("by", "date")?;
        kw.set_item("kind", "stable")?;
        let sorted = df.call_method("sort_values", (), Some(&kw))?;
        let kw = PyDict::new_bound(py);
        kw.set_item("drop", true)?;
        sorted.call_method("reset_index", (), Some(&kw))
    }

    // parse_metadata ----------------------------------------------------
    #[staticmethod]
    fn parse_metadata<'py>(
        py: Python<'py>,
        input_meta: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_metadata_inner(py, input_meta)
    }
}

// ─────────────────────── module-private helpers ───────────────────────

fn parse_metadata_inner<'py>(
    py: Python<'py>,
    input_meta: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import_bound("pandas")?;
    let kw = PyDict::new_bound(py);
    if input_meta.ends_with(".csv") {
        kw.set_item("sep", ",")?;
    } else if input_meta.ends_with(".tsv") {
        kw.set_item("sep", "\t")?;
    } else {
        // The original silently swallows unknown extensions and returns
        // None. Match that by raising a clean Python error instead — it
        // produces a useful message rather than a downstream None.
        return Err(PyValueError::new_err(format!(
            "Unsupported metadata extension for {}: expected .csv or .tsv",
            input_meta
        )));
    }
    let df = pd.call_method("read_csv", (input_meta,), Some(&kw))?;
    let columns: Vec<String> = df.getattr("columns")?.call_method0("tolist")?.extract()?;
    if !columns.iter().any(|c| c == "date") {
        return Err(PyValueError::new_err(
            "Metadata file must contain a \"date\" column",
        ));
    }
    let dates_col = df.call_method1("__getitem__", ("date",))?;
    let parsed = pd.call_method1("to_datetime", (dates_col,))?;
    df.call_method1("__setitem__", ("date", parsed))?;
    let kw = PyDict::new_bound(py);
    kw.set_item("by", "date")?;
    Ok(df.call_method("sort_values", (), Some(&kw))?)
}
