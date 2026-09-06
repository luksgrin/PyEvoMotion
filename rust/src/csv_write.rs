//! TSV/CSV writer reproducing `DataFrame.to_csv(sep=..., index=False)` byte
//! for byte for the columns the pipeline produces (design doc §6.9): Python
//! `repr` for floats and for lists of strings, `YYYY-MM-DD` dates (with time
//! when any value is not midnight), empty fields for missing values, and
//! minimal quoting with doubled quotes.

use std::fmt::Write as _;

use pyo3::prelude::*;

use crate::table::{Column, Table, TimeUnit, NAT};

/// Python's `repr(float)`: shortest round-trip digits, fixed notation for
/// exponents in [-4, 16), otherwise `d.ddde±XX`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` gives the shortest round-trip mantissa: "d.ddde<exp>".
    let sci = format!("{:e}", x);
    let (mantissa, exp) = sci.split_once('e').expect("scientific format");
    let exp: i32 = exp.parse().expect("exponent");
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-');
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                for _ in digits.len()..int_len {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// Python's `repr(str)`.
pub fn str_repr(s: &str) -> String {
    let use_double = s.contains('\'') && !s.contains('"');
    let quote = if use_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if !c.is_ascii() && !is_printable(c) => {
                let cp = c as u32;
                if cp <= 0xff {
                    let _ = write!(out, "\\x{:02x}", cp);
                } else if cp <= 0xffff {
                    let _ = write!(out, "\\u{:04x}", cp);
                } else {
                    let _ = write!(out, "\\U{:08x}", cp);
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Approximation of Python's `str.isprintable` for non-ASCII: everything
/// except separators/controls/unassigned. Metadata is ASCII in practice;
/// the writer falls back to pandas for tables holding exotic text (see
/// `needs_pandas`).
fn is_printable(c: char) -> bool {
    !(c.is_control() || c.is_whitespace() && c != ' ')
}

/// Python's `repr(list[str])`.
pub fn list_repr(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&str_repr(s));
    }
    out.push(']');
    out
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

const NS_PER_DAY: i64 = 86_400_000_000_000;

fn format_datetime(ns: i64, with_time: bool) -> String {
    let days = ns.div_euclid(NS_PER_DAY);
    let rem = ns.rem_euclid(NS_PER_DAY);
    let (y, m, d) = civil_from_days(days);
    if !with_time {
        return format!("{:04}-{:02}-{:02}", y, m, d);
    }
    let secs = rem / 1_000_000_000;
    let (h, mi, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", y, m, d, h, mi, s)
}

/// QUOTE_MINIMAL: quote when the field contains the separator, a quote, or a
/// line break; double embedded quotes.
fn quote_field(field: &str, sep: char, out: &mut String) {
    if field.contains(sep) || field.contains('"') || field.contains('\n') || field.contains('\r') {
        out.push('"');
        for c in field.chars() {
            if c == '"' {
                out.push('"'); // doubled
            }
            out.push(c);
        }
        out.push('"');
    } else {
        out.push_str(field);
    }
}

/// Columns whose text this writer cannot reproduce (opaque objects, foreign
/// dtypes, sub-second datetimes): the caller should use pandas instead.
pub fn needs_pandas(table: &Table) -> bool {
    table.columns.iter().any(|(_, c)| match c {
        Column::PyObject(_) | Column::Foreign { .. } => true,
        Column::DatetimeNs { ns, .. } => ns.iter().any(|&v| v != NAT && v.rem_euclid(1_000_000_000) != 0),
        Column::Str(d) => d.iter().flatten().any(|s| s.chars().any(|c| !c.is_ascii() && !is_printable(c))),
        _ => false,
    })
}

/// Render the table as `DataFrame.to_csv(sep=sep, index=False)` would.
pub fn to_delimited(table: &Table, sep: char) -> String {
    let nrows = table.nrows();
    let mut out = String::new();

    // header
    for (i, (name, _)) in table.columns.iter().enumerate() {
        if i > 0 {
            out.push(sep);
        }
        quote_field(name, sep, &mut out);
    }
    out.push('\n');

    // per-column pre-rendered cells
    let rendered: Vec<Vec<String>> = table
        .columns
        .iter()
        .map(|(_, col)| render_column(col))
        .collect();

    let single_column = rendered.len() == 1;
    for r in 0..nrows {
        for (c, cells) in rendered.iter().enumerate() {
            if c > 0 {
                out.push(sep);
            }
            // csv.writer rule inherited by pandas: a row consisting of one
            // empty field is written as `""` so it is not a blank line.
            if single_column && cells[r].is_empty() {
                out.push_str("\"\"");
            } else {
                quote_field(&cells[r], sep, &mut out);
            }
        }
        out.push('\n');
    }
    out
}

fn render_column(col: &Column) -> Vec<String> {
    match col {
        Column::Int64(v) => v.iter().map(|x| x.to_string()).collect(),
        Column::UInt64(v) => v.iter().map(|x| x.to_string()).collect(),
        Column::Float64(v) => v
            .iter()
            .map(|&x| if x.is_nan() { String::new() } else { float_repr(x) })
            .collect(),
        Column::Bool(v) => v.iter().map(|&b| if b { "True".into() } else { "False".into() }).collect(),
        Column::Str(d) => d.iter().map(|s| s.unwrap_or("").to_string()).collect(),
        Column::StrList(v) => v
            .iter()
            .map(|x| match x {
                Some(list) => list_repr(list),
                None => String::new(),
            })
            .collect(),
        Column::DatetimeNs { ns, unit: _ } => {
            let with_time = ns.iter().any(|&v| v != NAT && v.rem_euclid(NS_PER_DAY) != 0);
            ns.iter()
                .map(|&v| if v == NAT { String::new() } else { format_datetime(v, with_time) })
                .collect()
        }
        // Not reachable when `needs_pandas` is honoured; render something sane.
        Column::PyObject(_) | Column::Foreign { .. } => vec![String::new(); col.len()],
    }
}

/// Write `table` to `path`; returns Ok(false) when the caller must fall back
/// to pandas (see `needs_pandas`).
pub fn write_delimited(_py: Python<'_>, table: &Table, path: &str, sep: char) -> PyResult<bool> {
    if needs_pandas(table) {
        return Ok(false);
    }
    std::fs::write(path, to_delimited(table, sep))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write {}: {}", path, e)))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (x, s) in [
            (1.0, "1.0"), (0.5, "0.5"), (1e-5, "1e-05"), (1e16, "1e+16"), (1e15, "1000000000000000.0"),
            (0.30000000000000004, "0.30000000000000004"), (1.2345678901234568e17, "1.2345678901234568e+17"),
            (0.0001, "0.0001"), (123456.789, "123456.789"), (-2.5, "-2.5"), (6.249999999999999, "6.249999999999999"),
            (100.0, "100.0"), (1e-7, "1e-07"), (f64::INFINITY, "inf"), (-0.0, "-0.0"), (2.5e-3, "0.0025"),
        ] {
            assert_eq!(float_repr(x), s, "{}", x);
        }
    }

    #[test]
    fn str_and_list_repr() {
        assert_eq!(str_repr("s_1_A"), "'s_1_A'");
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(str_repr("a\\b"), "'a\\\\b'");
        assert_eq!(str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(list_repr(&["s_1_A".into(), "d_2_C".into()]), "['s_1_A', 'd_2_C']");
        assert_eq!(list_repr(&[]), "[]");
    }

    #[test]
    fn dates() {
        assert_eq!(format_datetime(18262 * NS_PER_DAY, false), "2020-01-01");
        assert_eq!(format_datetime(18262 * NS_PER_DAY + 3_600_000_000_000, true), "2020-01-01 01:00:00");
    }

    #[test]
    fn quoting() {
        let mut s = String::new();
        quote_field("a\tb", '\t', &mut s);
        assert_eq!(s, "\"a\tb\"");
        let mut s = String::new();
        quote_field("say \"hi\"", '\t', &mut s);
        assert_eq!(s, "\"say \"\"hi\"\"\"");
        let mut s = String::new();
        quote_field("plain", '\t', &mut s);
        assert_eq!(s, "plain");
    }
}
