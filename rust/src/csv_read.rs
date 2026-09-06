//! Delimited-text reader reproducing the defaults of `pandas.read_csv` that
//! PyEvoMotion relied on (design doc §6.1): tokenisation (quotes, CRLF, BOM,
//! blank lines), header naming (`Unnamed: N`, `name.1` for duplicates),
//! ragged rows (short → padded with missing, long → ParserError) and per-column
//! dtype inference (int64 / uint64 / float64 / bool / object).

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use crate::table::{Column, DictStr, IndexKind, Table};

/// pandas' default `na_values`.
const NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

pub fn is_na(cell: &str) -> bool {
    NA_VALUES.contains(&cell)
}

/// Read the whole file and split it into rows of fields.
fn tokenize(text: &str, sep: char) -> Vec<Vec<String>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    // `row_started`: anything (text, separator, quote) seen on this line, so
    // an entirely empty line can be skipped (`skip_blank_lines=True`).
    // `field_dirty`: anything consumed into the current field, so a quote
    // only opens a quoted field at the very start of a field.
    let mut row_started = false;
    let mut field_dirty = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        if c == '"' && !field_dirty {
            in_quotes = true;
            field_dirty = true;
            row_started = true;
        } else if c == sep {
            row.push(std::mem::take(&mut field));
            field_dirty = false;
            row_started = true;
        } else if c == '\n' || c == '\r' {
            if c == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            if row_started {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            row_started = false;
            field_dirty = false;
        } else {
            field.push(c);
            field_dirty = true;
            row_started = true;
        }
    }
    if row_started {
        row.push(std::mem::take(&mut field));
        rows.push(row);
    }
    rows
}

/// pandas header naming: empty → `Unnamed: i`, duplicates → `name.k`.
fn header_names(raw: &[String]) -> Vec<String> {
    let mut names: Vec<String> = raw
        .iter()
        .enumerate()
        .map(|(i, h)| if h.is_empty() { format!("Unnamed: {}", i) } else { h.clone() })
        .collect();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for i in 0..names.len() {
        let base = names[i].clone();
        let count = seen.entry(base.clone()).or_insert(0);
        if *count > 0 {
            names[i] = format!("{}.{}", base, count);
        }
        *count += 1;
    }
    names
}

fn parse_i64(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let t = t.strip_prefix('+').unwrap_or(t);
    t.parse::<i64>().ok()
}

fn parse_u64(s: &str) -> Option<u64> {
    let t = s.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    t.parse::<u64>().ok()
}

fn parse_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    // Rust accepts "inf", "-inf", "infinity", "nan", exponents; pandas likewise.
    // Rust does not accept underscores or hex, and neither does pandas.
    t.parse::<f64>().ok()
}

const TRUE_SET: &[&str] = &["True", "TRUE", "true"];
const FALSE_SET: &[&str] = &["False", "FALSE", "false"];

/// Infer one column from its raw cells (pandas defaults).
fn infer_column(py: Python<'_>, cells: &[Option<&str>]) -> Column {
    let present: Vec<&str> = cells.iter().filter_map(|c| *c).collect();
    let any_missing = present.len() < cells.len();

    if present.is_empty() {
        return Column::Float64(vec![f64::NAN; cells.len()]);
    }

    // Integers.
    if present.iter().all(|c| parse_i64(c).is_some()) {
        if any_missing {
            return Column::Float64(
                cells
                    .iter()
                    .map(|c| c.and_then(parse_i64).map(|v| v as f64).unwrap_or(f64::NAN))
                    .collect(),
            );
        }
        return Column::Int64(cells.iter().map(|c| parse_i64(c.unwrap()).unwrap()).collect());
    }
    if present.iter().all(|c| parse_u64(c).is_some()) {
        if any_missing {
            return Column::Float64(
                cells
                    .iter()
                    .map(|c| c.and_then(parse_u64).map(|v| v as f64).unwrap_or(f64::NAN))
                    .collect(),
            );
        }
        return Column::UInt64(cells.iter().map(|c| parse_u64(c.unwrap()).unwrap()).collect());
    }

    // Floats.
    if present.iter().all(|c| parse_f64(c).is_some()) {
        return Column::Float64(
            cells
                .iter()
                .map(|c| c.and_then(parse_f64).unwrap_or(f64::NAN))
                .collect(),
        );
    }

    // Booleans.
    if present
        .iter()
        .all(|c| TRUE_SET.contains(c) || FALSE_SET.contains(c))
    {
        if !any_missing {
            return Column::Bool(cells.iter().map(|c| TRUE_SET.contains(&c.unwrap())).collect());
        }
        // pandas: object column of Python bools and NaN.
        let objs: Vec<Py<PyAny>> = cells
            .iter()
            .map(|c| match c {
                None => PyFloat::new_bound(py, f64::NAN).into_any().unbind(),
                Some(v) => TRUE_SET.contains(v).into_py(py),
            })
            .collect();
        return Column::PyObject(objs);
    }

    // Strings.
    Column::Str(DictStr::from_options(
        cells.iter().map(|c| c.map(|s| s.to_string())),
    ))
}

/// Raise `pandas.errors.ParserError` with pandas' wording.
fn parser_error(py: Python<'_>, expected: usize, line: usize, saw: usize) -> PyErr {
    let msg = format!(
        "Error tokenizing data. C error: Expected {} fields in line {}, saw {}\n",
        expected, line, saw
    );
    match py
        .import_bound("pandas.errors")
        .and_then(|m| m.getattr("ParserError"))
    {
        Ok(cls) => PyErr::from_value_bound(cls.call1((msg.clone(),)).unwrap_or_else(|_| py.None().into_bound(py))),
        Err(_) => PyValueError::new_err(msg),
    }
}

/// Read a delimited file into a `Table` with a RangeIndex.
pub fn read_table(py: Python<'_>, path: &str, sep: char) -> PyResult<Table> {
    let bytes = std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyFileNotFoundError::new_err(format!(
                "[Errno 2] No such file or directory: '{}'",
                path
            ))
        } else {
            PyValueError::new_err(format!("Cannot read {}: {}", path, e))
        }
    })?;
    let text = String::from_utf8(bytes).map_err(|e| {
        PyValueError::new_err(format!("{} is not valid UTF-8: {}", path, e))
    })?;
    read_table_from_str(py, &text, sep)
}

pub fn read_table_from_str(py: Python<'_>, text: &str, sep: char) -> PyResult<Table> {
    let mut rows = tokenize(text, sep);
    if rows.is_empty() {
        return Err(PyValueError::new_err("No columns to parse from file"));
    }
    let header = rows.remove(0);
    let names = header_names(&header);
    let ncols = names.len();

    // Ragged rows: pandas pads short rows with NaN and rejects long rows.
    for (i, row) in rows.iter_mut().enumerate() {
        if row.len() > ncols {
            return Err(parser_error(py, ncols, i + 2, row.len()));
        }
        while row.len() < ncols {
            row.push(String::new());
        }
    }

    let nrows = rows.len();
    let mut columns: Vec<(String, Column)> = Vec::with_capacity(ncols);
    for (j, name) in names.into_iter().enumerate() {
        let cells: Vec<Option<&str>> = rows
            .iter()
            .map(|r| {
                let c = r[j].as_str();
                if is_na(c) {
                    None
                } else {
                    Some(c)
                }
            })
            .collect();
        columns.push((name, infer_column(py, &cells)));
    }
    Ok(Table {
        columns,
        index: IndexKind::range(nrows),
    })
}

/// Parse a Python list literal of string literals (`['s_1_A', "d_2_C"]`), as
/// written by `DataFrame.to_csv` for the "mutation instructions" column.
pub fn parse_str_list_literal(text: &str) -> Result<Vec<String>, String> {
    let t = text.trim();
    let inner = if let Some(i) = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        i
    } else if let Some(i) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        i
    } else {
        return Err(format!("not a list literal: {}", t));
    };
    let mut out = Vec::new();
    let mut chars = inner.chars().peekable();
    loop {
        // skip whitespace and commas
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() || c == ',' {
                chars.next();
            } else {
                break;
            }
        }
        let quote = match chars.next() {
            None => break,
            Some(q) if q == '\'' || q == '"' => q,
            Some(other) => return Err(format!("unexpected character {:?} in list literal", other)),
        };
        let mut s = String::new();
        loop {
            match chars.next() {
                None => return Err("unterminated string in list literal".into()),
                Some(c) if c == quote => break,
                Some('\\') => match chars.next() {
                    Some('\\') => s.push('\\'),
                    Some('\'') => s.push('\''),
                    Some('"') => s.push('"'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('0') => s.push('\0'),
                    Some('x') => {
                        let h: String = chars.by_ref().take(2).collect();
                        let v = u32::from_str_radix(&h, 16).map_err(|_| "bad \\x escape".to_string())?;
                        s.push(char::from_u32(v).ok_or("bad \\x escape")?);
                    }
                    Some('u') => {
                        let h: String = chars.by_ref().take(4).collect();
                        let v = u32::from_str_radix(&h, 16).map_err(|_| "bad \\u escape".to_string())?;
                        s.push(char::from_u32(v).ok_or("bad \\u escape")?);
                    }
                    Some('U') => {
                        let h: String = chars.by_ref().take(8).collect();
                        let v = u32::from_str_radix(&h, 16).map_err(|_| "bad \\U escape".to_string())?;
                        s.push(char::from_u32(v).ok_or("bad \\U escape")?);
                    }
                    Some(other) => {
                        s.push('\\');
                        s.push(other);
                    }
                    None => return Err("unterminated escape in list literal".into()),
                },
                Some(c) => s.push(c),
            }
        }
        out.push(s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_quotes_crlf_and_blank_lines() {
        let rows = tokenize("a\tb\r\n1\t\"x\ty\"\r\n\r\n2\t\"q\"\"q\"\n", '\t');
        assert_eq!(rows, vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["1".to_string(), "x\ty".to_string()],
            vec!["2".to_string(), "q\"q".to_string()],
        ]);
    }

    #[test]
    fn header_naming() {
        let names = header_names(&["a".into(), "".into(), "a".into(), "a".into()]);
        assert_eq!(names, vec!["a", "Unnamed: 1", "a.1", "a.2"]);
    }

    #[test]
    fn list_literals() {
        assert_eq!(parse_str_list_literal("[]").unwrap(), Vec::<String>::new());
        assert_eq!(parse_str_list_literal("['s_1_A', \"it's\"]").unwrap(), vec!["s_1_A", "it's"]);
        assert_eq!(parse_str_list_literal("['a\\\\b', '\\n']").unwrap(), vec!["a\\b", "\n"]);
        assert!(parse_str_list_literal("{}").is_err());
    }
}
