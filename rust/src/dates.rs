//! Date parsing and ordering for the `date` column (design doc §6.2 / §6.3,
//! canonical variant).
//!
//! `to_datetime_ns` reproduces `pandas.to_datetime(Series[str])` for the ISO
//! shapes PyEvoMotion's data uses, applied strictly to the whole column like
//! pandas does (one format, inferred from the first value). Anything else is
//! delegated to pandas itself, so exotic formats and every error message stay
//! pandas'. `stable_argsort` is the canonical row order: stable by date, NaT
//! last.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::table::{Column, NAT};

const NS_PER_SEC: i64 = 1_000_000_000;
const NS_PER_DAY: i64 = 86_400 * NS_PER_SEC;

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// The ISO-like shapes handled natively; everything else goes to pandas.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// `YYYY-M?M-D?D`
    Date,
    /// `YYYY-M?M-D?D<sep>HH:MM:SS` with `sep` fixed to the first value's
    /// separator (' ' or 'T'), whole seconds only. pandas infers one format
    /// from the first value, so mixing separators or adding fractional
    /// seconds is an error there; such columns take the pandas path.
    DateTime(char),
    /// `YYYY-MM` (first of the month)
    YearMonth,
    /// `YYYYMMDD`
    Compact,
}

fn split_dash(s: &str) -> Option<(i64, u32, Option<u32>)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 && parts.len() != 3 {
        return None;
    }
    if parts[0].len() != 4 || !parts[0].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let y: i64 = parts[0].parse().ok()?;
    let m: u32 = num(parts[1], 1, 2)?;
    let d = if parts.len() == 3 { Some(num(parts[2], 1, 2)?) } else { None };
    Some((y, m, d))
}

fn num(s: &str, min_len: usize, max_len: usize) -> Option<u32> {
    if s.len() < min_len || s.len() > max_len || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Parse one cell under `shape`; `None` when it does not fit.
fn parse_cell(s: &str, shape: Shape) -> Option<i64> {
    let s = s.trim();
    match shape {
        Shape::Compact => {
            if s.len() != 8 || !s.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let y: i64 = s[0..4].parse().ok()?;
            let m: u32 = s[4..6].parse().ok()?;
            let d: u32 = s[6..8].parse().ok()?;
            civil_ns(y, m, d, 0)
        }
        Shape::YearMonth => {
            let (y, m, d) = split_dash(s)?;
            if d.is_some() {
                return None;
            }
            civil_ns(y, m, 1, 0)
        }
        Shape::Date => {
            let (y, m, d) = split_dash(s)?;
            civil_ns(y, m, d?, 0)
        }
        Shape::DateTime(sep) => {
            let (date_part, time_part) = s.split_once(sep)?;
            let (y, m, d) = split_dash(date_part)?;
            let time_ns = parse_time_ns(time_part)?;
            civil_ns(y, m, d?, time_ns)
        }
    }
}

fn parse_time_ns(t: &str) -> Option<i64> {
    if t.contains('.') {
        return None; // fractional seconds: pandas decides
    }
    let (hms, frac) = (t, None::<&str>);
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h = num(parts[0], 2, 2)? as i64;
    let mi = num(parts[1], 2, 2)? as i64;
    let se = num(parts[2], 2, 2)? as i64;
    if h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let mut ns = ((h * 60 + mi) * 60 + se) * NS_PER_SEC;
    if let Some(f) = frac {
        if f.is_empty() || f.len() > 9 || !f.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut v: i64 = f.parse().ok()?;
        for _ in f.len()..9 {
            v *= 10;
        }
        ns += v;
    }
    Some(ns)
}

fn civil_ns(y: i64, m: u32, d: u32, time_ns: i64) -> Option<i64> {
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    // Stay inside the datetime64[ns] range (years 1678..2262) like pandas.
    if !(1678..=2262).contains(&y) {
        return None;
    }
    Some(days_from_civil(y, m, d) * NS_PER_DAY + time_ns)
}

fn detect_shape(first: &str) -> Option<Shape> {
    let s = first.trim();
    if parse_cell(s, Shape::Date).is_some() {
        return Some(Shape::Date);
    }
    for sep in [' ', 'T'] {
        if parse_cell(s, Shape::DateTime(sep)).is_some() {
            return Some(Shape::DateTime(sep));
        }
    }
    if parse_cell(s, Shape::YearMonth).is_some() {
        return Some(Shape::YearMonth);
    }
    if parse_cell(s, Shape::Compact).is_some() {
        return Some(Shape::Compact);
    }
    None
}

/// `pandas.to_datetime` on a string column → nanoseconds since the epoch
/// (`NAT` for missing). Native for ISO shapes, pandas for everything else.
pub fn to_datetime_ns(py: Python<'_>, col: &Column) -> PyResult<Vec<i64>> {
    match col {
        Column::DatetimeNs { ns, .. } => return Ok(ns.clone()),
        Column::Str(d) => {
            let first = d.iter().flatten().next();
            if let Some(shape) = first.and_then(detect_shape) {
                let mut out = Vec::with_capacity(d.len());
                let mut ok = true;
                for cell in d.iter() {
                    match cell {
                        None => out.push(NAT),
                        Some(s) => match parse_cell(s, shape) {
                            Some(v) => out.push(v),
                            None => {
                                ok = false;
                                break;
                            }
                        },
                    }
                }
                if ok {
                    return Ok(out);
                }
            }
            // Fallback: let pandas parse (and raise) exactly as before.
            let objs: Vec<Py<PyAny>> = d
                .iter()
                .map(|s| match s {
                    Some(x) => x.into_py(py),
                    None => py.None(),
                })
                .collect();
            let items = PyList::new_bound(py, objs);
            pandas_to_datetime_ns(py, items.as_any())
        }
        // An all-missing column read as float64, or any other dtype: pandas.
        _ => {
            let obj = crate::table::Table {
                columns: vec![("date".to_string(), col.clone_ref(py))],
                index: crate::table::IndexKind::range(col.len()),
            }
            .to_pandas(py)?
            .call_method1("__getitem__", ("date",))?;
            pandas_to_datetime_ns(py, &obj)
        }
    }
}

fn pandas_to_datetime_ns(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let pd = py.import_bound("pandas")?;
    let series = pd.call_method1("Series", (values,))?;
    let parsed = pd.call_method1("to_datetime", (series,))?;
    let as_ns = parsed.call_method1("astype", ("datetime64[ns]",))?;
    let arr = as_ns.call_method0("to_numpy")?.call_method1("view", ("int64",))?;
    let ro: numpy::PyReadonlyArray1<i64> = arr.extract()?;
    Ok(ro.as_slice()?.to_vec())
}

/// Canonical row order: stable sort by date, NaT last.
pub fn stable_argsort(ns: &[i64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..ns.len()).collect();
    idx.sort_by_key(|&i| if ns[i] == NAT { (1u8, 0i64) } else { (0u8, ns[i]) });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2020, 3, 1), 18322);
        assert_eq!(parse_cell("2020-3-5", Shape::Date), parse_cell("2020-03-05", Shape::Date));
        assert!(parse_cell("2020-02-30", Shape::Date).is_none());
        assert_eq!(
            parse_cell("2020-01-01 00:00:01", Shape::DateTime(' ')).unwrap(),
            days_from_civil(2020, 1, 1) * NS_PER_DAY + NS_PER_SEC
        );
        assert!(parse_cell("2020-01-01 00:00:01.5", Shape::DateTime(' ')).is_none());
        assert!(parse_cell("2020-01-01T00:00:01", Shape::DateTime(' ')).is_none());
        assert_eq!(parse_cell("20200301", Shape::Compact), parse_cell("2020-03-01", Shape::Date));
        assert_eq!(parse_cell("2020-03", Shape::YearMonth), parse_cell("2020-03-01", Shape::Date));
    }

    #[test]
    fn stable_sort_nat_last() {
        let v = [5, NAT, 3, 5, 1];
        assert_eq!(stable_argsort(&v), vec![4, 2, 0, 3, 1]);
    }
}
