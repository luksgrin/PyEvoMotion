//! Window statistics on the internal table (design doc §6.6–6.8, canonical
//! variant): reproduce `groupby(pd.Grouper(key="date", freq=DT, origin=...))`
//! bin edges for tick frequencies, keep only bins with at least two rows,
//! and compute size / mean / variance per bin with a fixed summation order
//! and no fused multiply-add, so every platform produces the same bytes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::table::{Column, IndexKind, Table, TimeUnit, NAT};

/// Nanoseconds for a pandas "tick" offset alias (`7D`, `12h`, `30min`, ...).
/// Anchored offsets (`W`, `MS`, ...) return `None`; callers fall back to pandas.
pub fn parse_tick_ns(dt: &str) -> Option<i64> {
    let s = dt.trim();
    let digits_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, unit) = s.split_at(digits_end);
    let n: i64 = if num.is_empty() { 1 } else { num.parse().ok()? };
    let unit_ns: i64 = match unit.trim() {
        "D" | "d" | "day" | "days" => 86_400_000_000_000,
        "h" | "H" | "hr" | "hour" | "hours" => 3_600_000_000_000,
        "min" | "T" | "minute" | "minutes" => 60_000_000_000,
        "s" | "S" | "sec" | "second" | "seconds" => 1_000_000_000,
        "ms" | "L" | "millisecond" | "milliseconds" => 1_000_000,
        "us" | "U" | "microsecond" | "microseconds" => 1_000,
        "ns" | "N" | "nanosecond" | "nanoseconds" => 1,
        _ => return None,
    };
    n.checked_mul(unit_ns).filter(|&v| v > 0)
}

/// Python-style floor modulo for i64.
fn pymod(a: i64, m: i64) -> i64 {
    ((a % m) + m) % m
}

/// Left edge of the first bin and the number of bins covering
/// `[min, max]` for `pd.Grouper(freq=f, origin=o)` with closed="left".
fn bin_layout(min: i64, max: i64, origin: i64, f: i64) -> (i64, usize) {
    let first = min - pymod(min - origin, f);
    let r = pymod(max - origin, f);
    let last = if r != 0 { max + (f - r) } else { max + f };
    let nbins = ((last - first) / f) as usize;
    (first, nbins.max(1))
}

fn bin_of(ns: i64, first: i64, f: i64) -> usize {
    ((ns - first) / f) as usize
}

/// The window statistics table: `date`, `mean <level>`..., `var <level>`...,
/// `size` (pandas column order of compute_stats), RangeIndex.
pub fn compute_stats_table(
    py: Python<'_>,
    data: &Table,
    dt_ns: i64,
    origin_ns: i64,
    levels: &[String],
) -> PyResult<Table> {
    let dates: Vec<i64> = match data.column("date") {
        Some(Column::DatetimeNs { ns, .. }) => ns.clone(),
        _ => return Err(PyValueError::new_err("KeyError: 'date'")),
    };
    let mut values: Vec<Vec<f64>> = Vec::with_capacity(levels.len());
    for level in levels {
        let v: Vec<f64> = match data.column(level) {
            Some(Column::Int64(v)) => v.iter().map(|&x| x as f64).collect(),
            Some(Column::Float64(v)) => v.clone(),
            Some(Column::UInt64(v)) => v.iter().map(|&x| x as f64).collect(),
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "column {:?} must be numeric",
                    level
                )))
            }
            None => return Err(PyValueError::new_err(format!("KeyError: {:?}", level))),
        };
        values.push(v);
    }

    // Working rows: (date, per-level values); the "duplicate the first row
    // when it sits alone on the origin" edge case is applied by the caller.
    let n = dates.len();
    let valid: Vec<usize> = (0..n).filter(|&i| dates[i] != NAT).collect();
    if valid.is_empty() {
        return Err(PyValueError::new_err(
            "No groups with at least 2 observations. Consider widening the time interval.",
        ));
    }

    // First grouping: count rows per bin, keep rows whose bin has >= 2.
    let min = valid.iter().map(|&i| dates[i]).min().unwrap();
    let max = valid.iter().map(|&i| dates[i]).max().unwrap();
    let (first, nbins) = bin_layout(min, max, origin_ns, dt_ns);
    let mut counts = vec![0usize; nbins];
    for &i in &valid {
        counts[bin_of(dates[i], first, dt_ns)] += 1;
    }
    let kept: Vec<usize> = valid
        .iter()
        .copied()
        .filter(|&i| counts[bin_of(dates[i], first, dt_ns)] >= 2)
        .collect();
    if kept.is_empty() {
        return Err(PyValueError::new_err(
            "No groups with at least 2 observations. Consider widening the time interval.",
        ));
    }

    // Second grouping on the kept rows (pandas re-groups the filtered frame;
    // origin anchoring keeps the edges aligned, the range may shrink).
    let min2 = kept.iter().map(|&i| dates[i]).min().unwrap();
    let max2 = kept.iter().map(|&i| dates[i]).max().unwrap();
    let (first2, nbins2) = bin_layout(min2, max2, origin_ns, dt_ns);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); nbins2];
    for &i in &kept {
        members[bin_of(dates[i], first2, dt_ns)].push(i);
    }

    // Aggregates, canonical arithmetic: mean = plain sum / n in row order;
    // var (ddof=1) = two-pass sum of squared deviations / (n - 1).
    let mut date_col: Vec<i64> = Vec::with_capacity(nbins2);
    let mut size_col: Vec<i64> = Vec::with_capacity(nbins2);
    let mut mean_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(nbins2); levels.len()];
    let mut var_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(nbins2); levels.len()];
    for (b, rows) in members.iter().enumerate() {
        date_col.push(first2 + b as i64 * dt_ns);
        size_col.push(rows.len() as i64);
        for (k, v) in values.iter().enumerate() {
            let m = rows.len();
            if m == 0 {
                mean_cols[k].push(f64::NAN);
                var_cols[k].push(f64::NAN);
                continue;
            }
            let mut sum = 0.0f64;
            for &i in rows {
                sum += v[i];
            }
            let mean = sum / m as f64;
            mean_cols[k].push(mean);
            if m < 2 {
                var_cols[k].push(f64::NAN);
            } else {
                let mut ss = 0.0f64;
                for &i in rows {
                    let d = v[i] - mean;
                    ss += d * d;
                }
                var_cols[k].push(ss / (m as f64 - 1.0));
            }
        }
    }

    let mut columns: Vec<(String, Column)> = Vec::with_capacity(2 + 2 * levels.len());
    columns.push((
        "date".to_string(),
        Column::DatetimeNs {
            ns: date_col,
            unit: TimeUnit::Ns,
        },
    ));
    for (k, level) in levels.iter().enumerate() {
        columns.push((format!("mean {}", level), Column::Float64(std::mem::take(&mut mean_cols[k]))));
    }
    for (k, level) in levels.iter().enumerate() {
        columns.push((format!("var {}", level), Column::Float64(std::mem::take(&mut var_cols[k]))));
    }
    columns.push(("size".to_string(), Column::Int64(size_col)));
    let _ = py;
    Ok(Table {
        columns,
        index: IndexKind::range(nbins2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: i64 = 86_400_000_000_000;

    #[test]
    fn tick_parsing() {
        assert_eq!(parse_tick_ns("7D"), Some(7 * D));
        assert_eq!(parse_tick_ns("D"), Some(D));
        assert_eq!(parse_tick_ns("12h"), Some(12 * 3_600_000_000_000));
        assert_eq!(parse_tick_ns("W"), None);
        assert_eq!(parse_tick_ns("MS"), None);
    }

    #[test]
    fn bin_layout_matches_grouper_rules() {
        // origin later than the data anchors backwards: origin Jan 5, min Jan 1 → first bin Dec 29
        let jan1 = 18262 * D; // 2020-01-01
        let jan5 = jan1 + 4 * D;
        let (first, n) = bin_layout(jan1, jan1, jan5, 7 * D);
        assert_eq!(first, jan1 - 3 * D);
        assert_eq!(n, 1);
        // max exactly on an edge gets its own extra bin
        let (first, n) = bin_layout(jan1, jan1 + 7 * D, jan1, 7 * D);
        assert_eq!(first, jan1);
        assert_eq!(n, 2);
    }
}
