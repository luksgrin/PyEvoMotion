//! Command-line interface for PyEvoMotion, ported from PyEvoMotion/cli.py.
//!
//! Exposes a single `_main` pyfunction wired to the `PyEvoMotion` console
//! script. It reproduces the original argparse behaviour: positional
//! seqs/meta/out, the same flags, the custom --filter/-gp/-dr parsers, JSON
//! import (-ij) / export (-xj), then constructs `PyEvoMotion`, runs the
//! analysis and writes the TSV/JSON outputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

const BANNER: &str = r#"
Welcome to Rodrigolab's
 _____       ______          __  __       _   _
|  __ \     |  ____|        |  \/  |     | | (_)
| |__) |   _| |____   _____ | \  / | ___ | |_ _  ___  _ __
|  ___/ | | |  __\ \ / / _ \| |\/| |/ _ \| __| |/ _ \| '_ \
| |   | |_| | |___\ V / (_) | |  | | (_) | |_| | (_) | | | |
|_|    \__, |______\_/ \___/|_|  |_|\___/ \__|_|\___/|_| |_|
        __/ |
       |___/
"#;

const HELP: &str = r#"usage: PyEvoMotion [-h] [-dt DELTA_T] [-sh] [-ep] [-cl CONFIDENCE_LEVEL]
                   [-l LENGTH_FILTER] [-xj] [-ij IMPORT_JSON]
                   [-k {all,total,substitutions,indels}]
                   [-f FILTER [FILTER ...]] [-gp GENOME_POSITIONS]
                   [-dr DATE_RANGE] [-ref REFSEQ] [-v]
                   [-load LOAD_MUTATION_INSTRUCTIONS] [-recount]
                   seqs meta out

PyEvoMotion

positional arguments:
  seqs                  Path to the input fasta file containing the sequences.
  meta                  Path to the corresponding metadata file for the
                        sequences.
  out                   Path to the output filename prefix used to save the
                        different results.

options:
  -h, --help            show this help message and exit
  -dt DELTA_T, --delta_t DELTA_T
                        Time interval to calculate the statistics. Default is
                        7 days (7D).
  -sh, --show           Show the plots of the analysis.
  -ep, --export_plots   Export the plots of the analysis.
  -cl CONFIDENCE_LEVEL, --confidence_level CONFIDENCE_LEVEL
                        Confidence level for parameter confidence intervals
                        (default 0.95 for 95% CI). Must be between 0 and 1.
  -l LENGTH_FILTER, --length_filter LENGTH_FILTER
                        Length filter for the sequences (removes sequences
                        with length less than the specified value). Default is
                        0.
  -xj, --export_json    Export the run arguments to a json file.
  -ij IMPORT_JSON, --import_json IMPORT_JSON
                        Import the run arguments from a JSON file. If this
                        argument is passed, the other arguments are ignored.
                        The JSON file must contain the mandatory keys 'seqs',
                        'meta', and 'out'.
  -k {all,total,substitutions,indels}, --kind {all,total,substitutions,indels}
                        Kind of mutations to consider for the analysis.
                        Default is 'all'.
  -f FILTER [FILTER ...], --filter FILTER [FILTER ...]
                        Specify filters to be applied on the data with keys
                        followed by values. If the values are multiple, they
                        must be enclosed in square brackets. Example: --filter
                        key1 value1 key2 [value2 value3] key3 value4. If
                        either the keys or values contain spaces, they must be
                        enclosed in quotes. keys must be present in the
                        metadata file as columns for the filter to be applied.
                        Use '*' as a wildcard, for example Bio* to filter all
                        columns starting with 'Bio'.
  -gp GENOME_POSITIONS, --genome_positions GENOME_POSITIONS
                        Genome positions to restrict the analysis. The
                        positions must be separated by two dots. Example:
                        1..1000. Open start or end positions are allowed by
                        omitting the first or last position, respectively. If
                        not specified, the whole reference genome is
                        considered.
  -dr DATE_RANGE, --date_range DATE_RANGE
                        Date range to filter the data. The date range must be
                        separated by two dots and the format must be YYYY-MM-
                        DD. Example: 2020-01-01..2020-12-31. If not specified,
                        the whole dataset is considered. Note that if the
                        origin is specified, the most restrictive date range
                        is considered.
  -ref REFSEQ, --refseq REFSEQ
                        FASTA file with the reference sequence (its first
                        record is used). Default is to use the sequence with
                        the earliest date in the metadata.
  -v, --verbose         Print progress information to stderr (repeat for
                        debug-level output).
  -load LOAD_MUTATION_INSTRUCTIONS, --load_mutation_instructions LOAD_MUTATION_INSTRUCTIONS
                        Load previously determined mutation instructions from
                        the '<out>.tsv' file written by an earlier run,
                        skipping the alignment step. 'meta' is then ignored
                        and 'seqs' is only read to fetch the reference
                        sequence (unless -ref is given). Filters (-f, -gp,
                        -dr) are still applied. The data TSV is not rewritten
                        in this mode.
  -recount, --recount_mutation_types
                        Recount the number of substitutions and indels from
                        the loaded mutation instructions instead of reusing
                        the counts stored in the TSV.
"#;

const KIND_CHOICES: [&str; 4] = ["all", "total", "substitutions", "indels"];

// ─────────────────────── helpers ───────────────────────

/// Raise SystemExit(code) — sys.exit always raises, so the Ok arm never fires.
fn sys_exit(py: Python<'_>, code: i32) -> PyErr {
    match py
        .import_bound("sys")
        .and_then(|sys| sys.call_method1("exit", (code,)))
    {
        Ok(_) => PyValueError::new_err("sys.exit did not raise"),
        Err(e) => e,
    }
}

/// Mirror `_ArgumentParserWithHelpOnError.error`: print help, the error
/// message, and exit with status 2.
fn arg_error(py: Python<'_>, msg: &str) -> PyErr {
    print!("{}", HELP);
    print!("\nError: {}\n\n", msg);
    sys_exit(py, 2)
}

/// Port of `_ParseFilter.parse_filters`: keys followed by values, with
/// square-bracket groups collected into lists. Returns a Python dict whose
/// values are str or list[str]. Behaviour matches the original exactly.
fn parse_filters<'py>(py: Python<'py>, values: &[String]) -> PyResult<Bound<'py, PyDict>> {
    // cleaned holds either a single string or a Vec<String> (bracket group).
    enum Item {
        S(String),
        L(Vec<String>),
    }
    let mut cleaned: Vec<Item> = Vec::new();
    let mut buffer: Vec<String> = Vec::new();
    let mut inside = false;

    for value in values {
        let starts = value.starts_with('[');
        let ends = value.ends_with(']');
        if starts && ends {
            cleaned.push(Item::S(value[1..value.len() - 1].to_string()));
        }
        if starts {
            inside = true;
            buffer.push(value[1..].to_string());
        } else if ends {
            buffer.push(value[..value.len() - 1].to_string());
            cleaned.push(Item::L(std::mem::take(&mut buffer)));
            inside = false;
        } else if inside {
            buffer.push(value.clone());
        } else {
            cleaned.push(Item::S(value.clone()));
        }
    }

    // dict(zip(cleaned[::2], cleaned[1::2]))
    let dict = PyDict::new_bound(py);
    let mut i = 0;
    while i + 1 < cleaned.len() {
        let key = match &cleaned[i] {
            Item::S(s) => s.clone(),
            // A list key would stringify oddly; the original would raise on a
            // non-hashable key, so surface a clear error instead.
            Item::L(_) => return Err(arg_error(py, "filter keys must be single values")),
        };
        match &cleaned[i + 1] {
            Item::S(s) => dict.set_item(key, s)?,
            Item::L(l) => dict.set_item(key, l.clone())?,
        }
        i += 2;
    }
    Ok(dict)
}

/// Port of `_ParseGenomePosition.parse_genome_position`.
fn parse_genome_position(py: Python<'_>, value: &str) -> PyResult<(i64, i64)> {
    if !value.contains("..") {
        return Err(arg_error(
            py,
            "The genome positions must be separated by two dots. Example: 1..1000",
        ));
    }
    let parts: Vec<&str> = value.split("..").collect();
    let mut out = Vec::new();
    for el in &parts {
        if !el.is_empty() && !el.chars().all(|c| c.is_ascii_digit()) {
            return Err(arg_error(py, "The genome positions must be positive integers"));
        }
        out.push(if el.is_empty() {
            0
        } else {
            el.parse::<i64>().unwrap_or(0)
        });
    }
    // The original returns tuple(positions); a "a..b" string yields two parts.
    Ok((out[0], *out.get(1).unwrap_or(&0)))
}

/// Port of `_ParseDateRange.parse_date_range`: returns a Python tuple of
/// (datetime|None, datetime|None) and the original "start..end" string for
/// JSON export.
fn parse_date_range<'py>(
    py: Python<'py>,
    value: &str,
) -> PyResult<(Bound<'py, PyTuple>, String)> {
    if !value.contains("..") {
        return Err(arg_error(
            py,
            "The date range must be separated by two dots. Example: 2020-01-01..2020-12-31",
        ));
    }
    if value.matches('.').count() > 2 {
        return Err(arg_error(py, "The date range must contain '..' as separator"));
    }
    let datetime = py.import_bound("datetime")?.getattr("datetime")?;
    let parts: Vec<&str> = value.split("..").collect();
    let mut elems: Vec<Bound<'py, PyAny>> = Vec::new();
    for date in &parts {
        if date.is_empty() {
            elems.push(py.None().into_bound(py));
            continue;
        }
        match datetime.call_method1("strptime", (*date, "%Y-%m-%d")) {
            Ok(dt) => elems.push(dt),
            Err(_) => return Err(arg_error(py, "Incorrect date format, should be YYYY-MM-DD")),
        }
    }
    Ok((PyTuple::new_bound(py, &elems), value.to_string()))
}

// ─────────────────────── parsed args ───────────────────────

struct Args {
    seqs: String,
    meta: String,
    out: String,
    delta_t: String,
    show: bool,
    export_plots: bool,
    confidence_level: f64,
    length_filter: i64,
    export_json: bool,
    kind: String,
    filter: Option<Py<PyDict>>,
    genome_positions: Option<(i64, i64)>,
    date_range: Option<Py<PyTuple>>,
    date_range_str: Option<String>,
    refseq: Option<String>,
    verbose: i64,
    load_mutation_instructions: Option<String>,
    recount_mutation_types: bool,
}

impl Args {
    fn defaults() -> Self {
        Args {
            seqs: String::new(),
            meta: String::new(),
            out: String::new(),
            delta_t: "7D".to_string(),
            show: false,
            export_plots: false,
            confidence_level: 0.95,
            length_filter: 0,
            export_json: false,
            kind: "all".to_string(),
            filter: None,
            genome_positions: None,
            date_range: None,
            date_range_str: None,
            refseq: None,
            verbose: 0,
            load_mutation_instructions: None,
            recount_mutation_types: false,
        }
    }
}

/// Take the value following an option, or argparse-error if absent.
fn take_value<'a>(
    py: Python<'_>,
    argv: &'a [String],
    i: &mut usize,
    opt: &str,
) -> PyResult<&'a str> {
    *i += 1;
    if *i >= argv.len() {
        return Err(arg_error(py, &format!("argument {}: expected one argument", opt)));
    }
    Ok(argv[*i].as_str())
}

/// Parse argv[1:] into Args, mirroring argparse + the custom actions.
fn parse_args(py: Python<'_>, argv: &[String]) -> PyResult<Args> {
    // -ij short-circuits everything (matches the two-pass behaviour).
    let mut idx = 1;
    while idx < argv.len() {
        let a = &argv[idx];
        if a == "-ij" || a == "--import_json" {
            idx += 1;
            if idx >= argv.len() {
                return Err(arg_error(py, "argument -ij/--import_json: expected one argument"));
            }
            return load_args_from_json(py, &argv[idx]);
        }
        idx += 1;
    }

    let mut args = Args::defaults();
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].clone();
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", HELP);
                return Err(sys_exit(py, 0));
            }
            "-dt" | "--delta_t" => args.delta_t = take_value(py, argv, &mut i, "-dt/--delta_t")?.to_string(),
            "-sh" | "--show" => args.show = true,
            "-ep" | "--export_plots" => args.export_plots = true,
            "-xj" | "--export_json" => args.export_json = true,
            "-cl" | "--confidence_level" => {
                let v = take_value(py, argv, &mut i, "-cl/--confidence_level")?;
                args.confidence_level = v.parse::<f64>().map_err(|_| {
                    arg_error(
                        py,
                        &format!("argument -cl/--confidence_level: invalid float value: '{}'", v),
                    )
                })?;
            }
            "-l" | "--length_filter" => {
                let v = take_value(py, argv, &mut i, "-l/--length_filter")?;
                args.length_filter = v.parse::<i64>().map_err(|_| {
                    arg_error(
                        py,
                        &format!("argument -l/--length_filter: invalid int value: '{}'", v),
                    )
                })?;
            }
            "-k" | "--kind" => {
                let v = take_value(py, argv, &mut i, "-k/--kind")?.to_string();
                if !KIND_CHOICES.contains(&v.as_str()) {
                    return Err(arg_error(
                        py,
                        &format!(
                            "argument -k/--kind: invalid choice: '{}' (choose from 'all', 'total', 'substitutions', 'indels')",
                            v
                        ),
                    ));
                }
                args.kind = v;
            }
            "-gp" | "--genome_positions" => {
                let v = take_value(py, argv, &mut i, "-gp/--genome_positions")?.to_string();
                args.genome_positions = Some(parse_genome_position(py, &v)?);
            }
            "-dr" | "--date_range" => {
                let v = take_value(py, argv, &mut i, "-dr/--date_range")?.to_string();
                let (tup, s) = parse_date_range(py, &v)?;
                args.date_range = Some(tup.unbind());
                args.date_range_str = Some(s);
            }
            "-v" | "--verbose" => args.verbose += 1,
            "-vv" => args.verbose += 2,
            "-load" | "--load_mutation_instructions" => {
                args.load_mutation_instructions = Some(
                    take_value(py, argv, &mut i, "-load/--load_mutation_instructions")?.to_string(),
                );
            }
            "-recount" | "--recount_mutation_types" => args.recount_mutation_types = true,
            "-ref" | "--refseq" => {
                args.refseq = Some(take_value(py, argv, &mut i, "-ref/--refseq")?.to_string());
            }
            "-f" | "--filter" => {
                // nargs='+': consume following tokens until the next option.
                let mut vals: Vec<String> = Vec::new();
                while i + 1 < argv.len() && !argv[i + 1].starts_with('-') {
                    i += 1;
                    vals.push(argv[i].clone());
                }
                if vals.is_empty() {
                    return Err(arg_error(py, "argument -f/--filter: expected at least one argument"));
                }
                args.filter = Some(parse_filters(py, &vals)?.unbind());
            }
            other if other.starts_with('-') && other.len() > 1 && !other[1..].starts_with(|c: char| c.is_ascii_digit()) => {
                return Err(arg_error(py, &format!("unrecognized arguments: {}", other)));
            }
            _ => positionals.push(a),
        }
        i += 1;
    }

    // Required positionals.
    let names = ["seqs", "meta", "out"];
    if positionals.len() < 3 {
        let missing: Vec<&str> = names[positionals.len()..].to_vec();
        return Err(arg_error(
            py,
            &format!("the following arguments are required: {}", missing.join(", ")),
        ));
    }
    if positionals.len() > 3 {
        return Err(arg_error(
            py,
            &format!("unrecognized arguments: {}", positionals[3..].join(" ")),
        ));
    }
    args.seqs = positionals[0].clone();
    args.meta = positionals[1].clone();
    args.out = positionals[2].clone();
    Ok(args)
}

/// Load run args from a JSON file (-ij). Must contain seqs/meta/out.
fn load_args_from_json(py: Python<'_>, path: &str) -> PyResult<Args> {
    let json = py.import_bound("json")?;
    let builtins = py.import_bound("builtins")?;
    let file = builtins.call_method1("open", (path, "r"))?;
    let data: Bound<PyDict> = json.call_method1("load", (&file,))?.extract()?;
    file.call_method0("close")?;

    let has = |k: &str| -> PyResult<bool> { Ok(data.contains(k)?) };
    if !(has("seqs")? && has("meta")? && has("out")?) {
        return Err(arg_error(
            py,
            "The JSON file must contain the keys 'seqs', 'meta', and 'out'",
        ));
    }

    let mut args = Args::defaults();
    let get_str = |k: &str| -> PyResult<Option<String>> {
        match data.get_item(k)? {
            Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
            _ => Ok(None),
        }
    };

    args.seqs = get_str("seqs")?.unwrap_or_default();
    args.meta = get_str("meta")?.unwrap_or_default();
    args.out = get_str("out")?.unwrap_or_default();
    if let Some(v) = get_str("delta_t")? {
        args.delta_t = v;
    }
    if let Some(v) = data.get_item("show")? {
        if !v.is_none() {
            args.show = v.extract()?;
        }
    }
    if let Some(v) = data.get_item("export_plots")? {
        if !v.is_none() {
            args.export_plots = v.extract()?;
        }
    }
    if let Some(v) = data.get_item("confidence_level")? {
        if !v.is_none() {
            args.confidence_level = v.extract()?;
        }
    }
    if let Some(v) = data.get_item("length_filter")? {
        if !v.is_none() {
            args.length_filter = v.extract()?;
        }
    }
    if let Some(v) = get_str("kind")? {
        args.kind = v;
    }
    // filter: a dict (or null).
    if let Some(v) = data.get_item("filter")? {
        if !v.is_none() {
            let d: Bound<PyDict> = v.extract()?;
            args.filter = Some(d.unbind());
        }
    }
    // genome_positions: a [start, end] list (or null).
    if let Some(v) = data.get_item("genome_positions")? {
        if !v.is_none() {
            let t: (i64, i64) = v.extract()?;
            args.genome_positions = Some(t);
        }
    }
    // date_range: a "start..end" string (or null).
    if let Some(v) = get_str("date_range")? {
        if !v.is_empty() {
            let (tup, s) = parse_date_range(py, &v)?;
            args.date_range = Some(tup.unbind());
            args.date_range_str = Some(s);
        }
    }
    if let Some(v) = get_str("refseq")? {
        args.refseq = Some(v);
    }
    if let Some(v) = data.get_item("verbose")? {
        if !v.is_none() {
            args.verbose = v.extract()?;
        }
    }
    if let Some(v) = get_str("load_mutation_instructions")? {
        args.load_mutation_instructions = Some(v);
    }
    if let Some(v) = data.get_item("recount_mutation_types")? {
        if !v.is_none() {
            args.recount_mutation_types = v.extract()?;
        }
    }
    Ok(args)
}

// ─────────────────────── regression-result reshaping ───────────────────────

/// Port of `_remove_model_functions`: recursively drop "model" keys (the
/// non-serialisable lambda/callable) from nested dicts.
fn remove_model_functions<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(d) = obj.downcast::<PyDict>() {
        let out = PyDict::new_bound(py);
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            if key == "model" {
                continue;
            }
            if v.downcast::<PyDict>().is_ok() {
                out.set_item(k, remove_model_functions(py, &v)?)?;
            } else {
                out.set_item(k, v)?;
            }
        }
        Ok(out.into_any())
    } else {
        Ok(obj.clone())
    }
}

/// Port of `_restructure_regression_results`.
fn restructure_regression_results<'py>(
    py: Python<'py>,
    reg: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let restructured = PyDict::new_bound(py);
    for (k, v) in reg.iter() {
        let key: String = k.extract()?;
        if let Some(base) = key.strip_suffix("_full_results") {
            let val: Bound<PyDict> = v.extract()?;
            let entry = PyDict::new_bound(py);
            for model_key in ["linear_model", "power_law_model"] {
                let m: Bound<PyDict> = val.get_item(model_key)?.unwrap().extract()?;
                let sub = PyDict::new_bound(py);
                for f in ["parameters", "confidence_intervals", "expression", "r2", "confidence_level"] {
                    if let Some(fv) = m.get_item(f)? {
                        sub.set_item(f, fv)?;
                    }
                }
                entry.set_item(model_key, sub)?;
            }
            if let Some(ms) = val.get_item("model_selection")? {
                entry.set_item("model_selection", ms)?;
            }
            restructured.set_item(base, entry)?;
        } else {
            // Keep non-full-results entries, unless a *_full_results twin exists.
            let twin = format!("{}_full_results", key);
            if !reg.contains(&twin)? {
                restructured.set_item(&key, v)?;
            }
        }
    }
    Ok(restructured)
}

// ─────────────────────── orchestration ───────────────────────

/// Write the run arguments to `{out}_run_args.json` (-xj). Unlike the
/// original, a missing date_range serialises to null rather than crashing.
fn export_run_args(py: Python<'_>, args: &Args) -> PyResult<()> {
    let d = PyDict::new_bound(py);
    d.set_item("seqs", &args.seqs)?;
    d.set_item("meta", &args.meta)?;
    d.set_item("out", &args.out)?;
    d.set_item("delta_t", &args.delta_t)?;
    d.set_item("show", args.show)?;
    d.set_item("export_plots", args.export_plots)?;
    d.set_item("confidence_level", args.confidence_level)?;
    d.set_item("length_filter", args.length_filter)?;
    d.set_item("kind", &args.kind)?;
    match &args.filter {
        Some(f) => d.set_item("filter", f.bind(py))?,
        None => d.set_item("filter", py.None())?,
    }
    match args.genome_positions {
        Some((a, b)) => d.set_item("genome_positions", PyList::new_bound(py, [a, b]))?,
        None => d.set_item("genome_positions", py.None())?,
    }
    match &args.date_range_str {
        Some(s) => d.set_item("date_range", s)?,
        None => d.set_item("date_range", py.None())?,
    }
    match &args.refseq {
        Some(s) => d.set_item("refseq", s)?,
        None => d.set_item("refseq", py.None())?,
    }
    d.set_item("verbose", args.verbose)?;
    match &args.load_mutation_instructions {
        Some(s) => d.set_item("load_mutation_instructions", s)?,
        None => d.set_item("load_mutation_instructions", py.None())?,
    }
    d.set_item("recount_mutation_types", args.recount_mutation_types)?;

    let json = py.import_bound("json")?;
    let builtins = py.import_bound("builtins")?;
    let file = builtins.call_method1("open", (format!("{}_run_args.json", args.out), "w"))?;
    let kw = PyDict::new_bound(py);
    kw.set_item("indent", 4)?;
    json.call_method("dump", (&d, &file), Some(&kw))?;
    file.call_method0("close")?;
    Ok(())
}

fn run(py: Python<'_>, args: Args) -> PyResult<()> {
    // Validate confidence level (matches the post-parse check in _main).
    if !(args.confidence_level > 0.0 && args.confidence_level < 1.0) {
        return Err(arg_error(py, "Confidence level must be between 0 and 1 (exclusive)"));
    }

    if args.export_json {
        export_run_args(py, &args)?;
    }

    // -v: INFO messages from the PyEvoMotion logger on stderr; -vv: DEBUG
    // with timestamps (mirrors logging.basicConfig in the original PR).
    if args.verbose > 0 {
        let logging = py.import_bound("logging")?;
        let kw = PyDict::new_bound(py);
        if args.verbose == 1 {
            kw.set_item("level", logging.getattr("INFO")?)?;
            kw.set_item("format", "%(message)s")?;
        } else {
            kw.set_item("level", logging.getattr("DEBUG")?)?;
            kw.set_item("format", "%(asctime)s:%(name)s:%(levelname)s:%(message)s")?;
        }
        logging.call_method("basicConfig", (), Some(&kw))?;
    }

    // Construct PyEvoMotion(seqs, meta, dt=..., filters=..., positions=..., date_range=...).
    // tp_init is wired in lib.rs, so calling the type runs the Rust __init__.
    let cls = py.get_type_bound::<crate::core::PyEvoMotion>();
    let ckw = PyDict::new_bound(py);
    ckw.set_item("dt", &args.delta_t)?;
    if let Some(f) = &args.filter {
        ckw.set_item("filters", f.bind(py))?;
    }
    if let Some(p) = args.genome_positions {
        ckw.set_item("positions", p)?;
    }
    if let Some(dr) = &args.date_range {
        ckw.set_item("date_range", dr.bind(py))?;
    }
    if let Some(r) = &args.refseq {
        ckw.set_item("refseq", r)?;
    }
    ckw.set_item("verbose", args.verbose)?;
    if let Some(p) = &args.load_mutation_instructions {
        ckw.set_item("load_mutation_instructions", p)?;
    }
    ckw.set_item("recount_mutation_types", args.recount_mutation_types)?;
    let instance = cls.call((&args.seqs, &args.meta), Some(&ckw))?;

    // Export the parsed data to a TSV file — unless the data was itself
    // loaded from such a file, in which case rewriting it would at best be
    // redundant and at worst clobber the input when `out` matches.
    if args.load_mutation_instructions.is_none() {
        let csv_kw = PyDict::new_bound(py);
        csv_kw.set_item("sep", "\t")?;
        csv_kw.set_item("index", false)?;
        instance
            .getattr("data")?
            .call_method("to_csv", (format!("{}.tsv", args.out),), Some(&csv_kw))?;
    }

    // Run the analysis.
    let akw = PyDict::new_bound(py);
    akw.set_item("length", args.length_filter)?;
    akw.set_item("show", args.show)?;
    akw.set_item("mutation_kind", &args.kind)?;
    akw.set_item(
        "export_plots_filename",
        if args.export_plots {
            Some(format!("{}_plots", args.out))
        } else {
            None
        },
    )?;
    akw.set_item("confidence_level", args.confidence_level)?;
    let result = instance.call_method("analysis", (), Some(&akw))?;
    let (stats, reg): (Bound<PyAny>, Bound<PyDict>) = result.extract()?;

    // Restructure + strip non-serialisable model callables.
    let restructured = restructure_regression_results(py, &reg)?;
    let cleaned = PyDict::new_bound(py);
    for (k, v) in restructured.iter() {
        cleaned.set_item(k, remove_model_functions(py, &v)?)?;
    }

    // Export stats TSV.
    let stats_kw = PyDict::new_bound(py);
    stats_kw.set_item("sep", "\t")?;
    stats_kw.set_item("index", false)?;
    stats.call_method("to_csv", (format!("{}_stats.tsv", args.out),), Some(&stats_kw))?;

    // Export regression models JSON.
    let json = py.import_bound("json")?;
    let builtins = py.import_bound("builtins")?;
    let file = builtins.call_method1("open", (format!("{}_regression_results.json", args.out), "w"))?;
    let jkw = PyDict::new_bound(py);
    jkw.set_item("indent", 4)?;
    json.call_method("dump", (&cleaned, &file), Some(&jkw))?;
    file.call_method0("close")?;
    println!("Regression results saved to {}_regression_results.json", args.out);

    Err(sys_exit(py, 0))
}

/// Console-script entry point (pyproject: PyEvoMotion = "PyEvoMotion:_main").
#[pyfunction]
pub fn _main(py: Python<'_>) -> PyResult<()> {
    println!("{}", BANNER);
    let argv: Vec<String> = py.import_bound("sys")?.getattr("argv")?.extract()?;
    let args = parse_args(py, &argv)?;
    run(py, args)
}
