# Internal table: replacing the pandas internals of PyEvoMotion with a Rust column store

Status: **implemented** (phases 1–6, 2026-09-06) in the canonical variant (see the
decision box below); kept as the reference for the table's semantics. Target branch: `master`. Companion to `rust/README.md`.

> **Decision (2026-09-06): canonical, not faithful.** The investigation in
> sections 6.3 and 6.7 showed that the current outputs are only reproducible
> per machine: rows sharing a date are ordered by numpy's unstable quicksort,
> and the variance of a window depends on that order and on whether the pandas
> wheel used fused multiply-add (Apple Silicon: yes; x86_64: no). Rather than
> port those accidents, the implementation uses a **stable sort by date** and
> a **deterministic variance** (no FMA, fixed summation order), so every
> platform produces the same bytes. This is a documented behaviour change:
> rows that share a collection date may appear in a different order in
> `<out>.tsv`, and variance values may differ from 0.1.x in their last digits.
> Fitted models are unaffected at any meaningful precision. Section 6.3's
> numpy introsort port and the per-architecture switch in 6.7 are therefore
> **not implemented**; Appendix A is kept for reference only. The committed
> goldens are regenerated once, at the end of the port, and a golden test
> compares bytes on every CI platform from then on.

## 0. Summary of decisions

| Topic | Decision |
|---|---|
| Representation | Hand-rolled column store (`rust/src/table.rs`): `Table { columns: Vec<(String, Column)>, index: IndexKind }`, `Column` is a typed enum mirroring the pandas dtypes `read_csv` can produce, plus two escape hatches (`PyObject`, `Foreign`) for anything a subclass may put into `.data`. No polars, no arrow. |
| pandas boundary | `data` becomes a property on `PyEvoMotionParser`. Internally the instance is in one of two states: Rust-owned (`Table` only) or pandas-visible (a materialised `DataFrame` that is the source of truth until the next Rust mutation). This reproduces today's object-identity and in-place-mutation semantics exactly (section 4). |
| Conversion API for subclass authors | `PyEvoMotion.Table.to_pandas()` / `Table.from_pandas(df)`, the `data` setter accepting a `DataFrame` or a `Table`, and `_check_dataset_is_not_empty` accepting either. Nothing else. |
| Sorting | ~~Exact port of numpy's introsort~~ **Canonical: stable sort by date** (see decision box). Original text: `aquicksort_` (`SMALL_QUICKSORT = 15`, depth `2*msb(n)`, heapsort fallback). Validated bit-for-bit against `np.argsort(kind="quicksort")` on every fixture and on adversarial arrays (section 6.3). |
| Grouping and statistics | Rust bin computation reproducing `pd.Grouper(freq=<Tick>, origin=...)` edges; **canonical: mean = exact sum / n, variance = two-pass Σ(x-mean)²/(n-1) in row order, no FMA** (original text: mean by Kahan sum, variance by Welford with fused multiply-add on aarch64 (this is what the pandas wheels actually compute; section 6.7). Non-Tick frequencies fall back to the existing pandas path. |
| TSV/CSV reading | Rust reader implementing pandas `read_csv` default semantics (section 6.1); dates: ISO fast path with a pandas fallback for anything else (6.2). |
| TSV writing | Keep `DataFrame.to_csv` until the last phase; then a Rust writer that is byte-identical (6.9), switched on only when the golden and oracle tests are green on every CI platform. |
| Regex filters | Rust `regex` fast path; Python `re` fallback when the pattern does not compile in Rust; pandas' error behaviour reproduced. |
| Tests | Regenerated goldens (the committed ones are stale, section 8.1), a same-machine pandas oracle, hypothesis property tests for the table operations, byte-equality tests for the writer. |

A Python user observes nothing except a faster `-load` path and construction; a subclass author observes nothing unless they inspect `type(instance).__dict__["data"]` (now a property) or rely on `parse_metadata` overrides taking effect in `__init__` (they do not today either). Section 10 states this fully.

## 1. Goals, non-goals, constraints

Goals:

1. Remove pandas from the *internals* of the parser and core pipeline so that construction, `-load`, filters and `compute_stats` run in Rust on an in-memory table.
2. Keep pandas as the *boundary* type: `instance.data` is a `pandas.DataFrame`; `analysis()` returns `(pandas.DataFrame, dict)`; `date_grouper`, `parse_metadata`, `parse_mutation_data`, `get_differing_mutations`, `compute_stats` keep returning pandas objects.
3. Byte-identical CLI outputs (`<out>.tsv`, `<out>_stats.tsv`, `<out>_regression_results.json`) on test1, test4, test5 and the CI subset of test3, compared against the current 0.2.0 build **on the same platform** (see 8.1 for why "same platform" is unavoidable and already true today).
4. Every constructor keyword, filter, window rule, date-range rule, origin rule and error path keeps its exact semantics (section 5 maps each method).
5. Subclass override points survive (section 4.5).

Non-goals (explicitly out of scope, decided by the user):

- Plotting stays in matplotlib (`plot_results`, `export_plot_results`, `plot_single_data_and_model` untouched).
- Dropping pandas from `dependencies` is a later step; this design makes it *possible* (section 9, "Future").
- Changing behaviour that is arguably wrong (unstable tie order, the `length_filter`/`n_filter` no-ops) is not done here; one latent bug is fixed and called out (4.6).

Constraints found in the code base that shape the design:

- Method dispatch goes through Python attribute lookup (`slf.call_method1(...)`) so overrides work; `__init__` sets `self.data` after each stage; tests set `p.data = pd.DataFrame(...)` on a bare subclass and then call `filter_by_position` (`/Users/apophis/Documents/rodrigolab/PyEvoMotion-private/rust/tests/test_core_smoke.py`, `test_filter_by_position_window_rules`).
- `tests/helpers/test_UK_USA_dataset_helpers.py` and `tests/data/test3/sample_and_run*.py` call `PyEvoMotion.date_grouper(df, DT, origin)` and use the returned `DataFrameGroupBy`; it must remain a pandas object.
- Metadata is arbitrary: test1 has 28 columns (27 object, 1 int64; `location` is NaN in 98 of 101 rows; `age` mixes `"49"` and `"unknown"`); test3-ci carries an unnamed leading index column that pandas names `Unnamed: 0` and writes back under that header; test4/test5 have only `id` and `date`.

## 2. What pandas does today (inventory)

Read-only survey of `rust/src/parser.rs`, `rust/src/core.rs`, `rust/src/base.rs`, `rust/src/cli.rs`.

### 2.1 Parser (`PyEvoMotionParser`)

| Method | pandas operations today | Notes |
|---|---|---|
| `parse_metadata_inner` | `read_csv(sep by extension)`, `to_datetime(df["date"])`, `sort_values(by="date")` (default unstable quicksort, index labels preserved) | Called directly from `__init__`, not dispatched. Raises `ValueError` on unknown extension or missing `date`. |
| `parse_mutation_data` | `read_csv(sep="\t")`, `ast.literal_eval` per cell, `to_datetime`, `sort_values(kind="stable")`, `reset_index(drop=True)` | `-load` path. NaN cells become `None` and are dropped by `drop_missing_instructions`. |
| `drop_missing_instructions` | `isna`, boolean mask, `reset_index(drop=True)` | Only resets when something was dropped. Prints a warning to stderr. |
| `load_reference` | `data.iloc[0]["id"]` | Empty data raises `IndexError: single positional indexer is out-of-bounds`. |
| `filter_columns` | for each filter key present in `columns`: `col.str.contains(pattern, regex=True)`, boolean mask | Pattern: `*` becomes `.*`, alternatives joined with `|`. NaN in the column raises `ValueError: Cannot mask with non-boolean array containing NA / NaN values`; a non-object column raises `AttributeError: Can only use .str accessor with string values!`. |
| `filter_by_daterange` | `min()`, `max()`, comparisons, boolean mask | `start_v = start if start > dmin else dmin`, `end_v = end if end < dmax else dmax`, error if `start_v > end_v`; inclusive mask. On empty data `min()` is `NaT` and every comparison is `False`; no error. |
| `parse_data` | `merge(on="id", how="left")`, `reset_index(drop=True)`, then `drop_missing_instructions` | Left merge preserves the left order and yields NaN for ids absent from the FASTA. |
| `filter_by_position` | `tolist()` of list column, two `pd.Series(..., index=idx)` assignments, boolean mask | Early return on `data.empty`. Window: substitutions `start <= P < end`, indels `start < P <= end`. Rows whose list becomes empty are dropped; `["NO_MUTATION"]` sentinel restores `[]` for rows that had no mutations. Index labels are **not** reset. |
| `get_differing_mutations` | `selection.values.tolist()`, `pd.DataFrame(rows, columns=[...])` | Returns a DataFrame; dispatches `generate_alignment` and `create_modifs` through the class. |

### 2.2 Core (`_PyEvoMotionCore`, `PyEvoMotion`)

| Method | pandas operations today | Notes |
|---|---|---|
| `PyEvoMotion.__init__` | `data["date"].min()`, `builtins.min(date_min, start)`, `columns.tolist()` | `origin` is a `Timestamp` unless the `-dr` start is earlier than the data, in which case it is the `datetime.datetime` passed in (verified: `min(Timestamp, datetime)` returns the `datetime`). Count columns are recomputed unless loaded and present. |
| `count_mutation_types` | `tolist()`, four `pd.Series(counts)` assignments **without index**, `__add__` | Latent bug: the Series has a `RangeIndex`, so if `data.index` has gaps (rows dropped by `filter_by_position` or by `-dr`/`-f` in `-load` mode) the assignment aligns by label and produces float64 columns with NaN. Not triggered by any bundled dataset (no rows are dropped in test1 or test4; verified by id-set comparison). See 4.6. |
| `get_lengths` | `tolist()`, `pd.Series(lengths)` | Returns a Series with a fresh `RangeIndex`. |
| `length_filter`, `n_filter` | build a mask, discard it, `reset_index(drop=True, inplace=True)` | Deliberate no-ops (the original bug is preserved). Their only effect is resetting the index in place. |
| `compute_stats` | `copy()`, `iloc[0]`, `==`, `concat`, `sort_values(inplace=True)` (quicksort), `reset_index`, `date_grouper` twice, `groupby.filter(len>=2)`, `mean`, `var`, `size`, `rename`, `concat(axis=1)`, `reset_index(level=["date"])` | The first row is duplicated when it is alone at the origin date, then the whole frame is re-sorted (unstable) — this permutes tied rows in every group, and the variance is order-sensitive (6.7). Empty bins are kept (size 0, NaN mean/var). |
| `analysis` | column iteration, `stats["size"]`, `to_numpy().flatten()`, `stats[col] - min`, `Timedelta("7D")` division for `dt_idx` | Regressions already run in Rust (`base::linear_regression`, `adjust_model`). |
| `plot_results`, `export_plot_results` | pandas indexing of the stats frame, matplotlib | Out of scope. |

### 2.3 Base (`PyEvoMotionBase`)

Pure Rust already: `count_prefixes`, `mutation_length_modification`, `_remove_nan`, `_weighting_function`, `_compute_confidence_intervals`, `linear_regression`, `power_law_fit`, `F_test`, `AIC`, `adjust_model` (except one `numpy.zeros_like` call). Still pandas/numpy/matplotlib: `date_grouper` (`pd.Grouper`), `_get_time_ratio` and `_verify_dt` (`pd.Timedelta`), `_check_dataset_is_not_empty` (`df.empty`), `_power_law` (numpy), `plot_single_data_and_model` (matplotlib).

### 2.4 CLI (`cli.rs`)

`instance.data.to_csv(f"{out}.tsv", sep="\t", index=False)` (skipped in `-load` mode), `stats.to_csv(f"{out}_stats.tsv", sep="\t", index=False)`, `json.dump(indent=4)` for the regression results.

### 2.5 Empirical facts the design depends on (pandas 2.3.3, numpy 2.4.6, CPython 3.12, macOS arm64; all reproduced by probes during this study)

- `read_csv` infers `int64` for `length`, `object` for the other 27 test1 columns, `int64` for `Unnamed: 0` in test3-ci; a fully empty column is `float64`; `9223372036854775808` makes a column `uint64`; `True/False` columns are `bool`, `bool` with an empty cell is `object` holding Python bools and `nan`; `" 1"` in an otherwise int column still yields `int64` (numeric parsing tolerates surrounding whitespace) while `"  x "` keeps its spaces; default NA strings (`""`, `NA`, `N/A`, `NaN`, `nan`, `-nan`, `null`, `None`, `#N/A`, ...) become NaN; `inf` parses as infinity; duplicate headers become `a`, `a.1`; empty headers become `Unnamed: <pos>`; a short row is NaN-padded (turning an int column into float64); a long row raises `ParserError`; blank lines are skipped; CRLF is accepted; quoted fields with doubled quotes are unescaped.
- `to_datetime(Series[str])` infers **one** format from the first non-null value and applies it strictly: `["2020-03-30", "2020-03"]`, `["2020-03-30", "2020-03-30 10:00:00"]` and `["2020-03-30", "2020/03/31"]` all raise `ValueError`; `"2020-3-5"` is accepted under `%Y-%m-%d`; `""`/`None` become `NaT`; `dd/mm/yyyy`, `mm/dd/yyyy`, `yyyymmdd`, `March 30, 2020` are inferred via dateutil. All bundled fixtures use `YYYY-MM-DD` exclusively.
- `sort_values(by="date")` order equals `np.argsort(values, kind="quicksort")` on the `datetime64[ns]` array; the exported row order of every bundled output equals that order and **not** the stable order (test1: 81 of 101 rows share a date; S1: 2000 of 2001).
- `to_csv`: floats are written with Python `repr` (`1e-05`, `1e+16`, `0.30000000000000004`, `1.2345678901234568e+17`), NaN as empty, datetime columns as `YYYY-MM-DD` when every value is midnight, otherwise `YYYY-MM-DD HH:MM:SS`, NaT as `""` (quoted empty), lists via `repr` (`['s_1_A', 'd_2_C']`, `["it's"]`, `['a\\b']`, `[None]`), QUOTE_MINIMAL with `"` doubling when a field contains the separator, a quote or a newline, `True`/`False` for bools.
- `groupby(Grouper).var()` on this platform equals Welford's algorithm **with the M2 update fused** (`m2 = fma(x - mean_new, x - mean_old, m2)`): 3000/3000 random integer groups and 1000/1000 float groups match bit-for-bit; plain Welford matches only 2353/3000. Apple clang's default `-ffp-contract=on` fuses that Cython statement on aarch64; manylinux x86_64 wheels (baseline SSE2) cannot. Mean matches every candidate (integer sums are exact).
- Bin edges for `Grouper(freq="7D", origin=o)`: `first = min - ((min - o) mod f)` with floor modulo (so an origin later than the data still anchors, e.g. origin 2020-01-05 and min 2020-01-01 gives a first bin at 2019-12-29), `last = max + (f - ((max - o) mod f))` if that modulo is non-zero else `max + f`; bins are `[first + k f, first + (k+1) f)` labelled by the left edge; NaT rows belong to no bin and are dropped by `filter`.
- `d["n"] = pd.Series([0, 1, 2])` on a frame whose index is `[0, 2, 3]` gives `[0.0, 2.0, nan]` (float64): the `count_mutation_types` alignment hazard.

## 3. Representation choice

### 3.1 Operations actually needed

Read TSV/CSV with pandas inference; parse a date column; sort rows by date (unstable, numpy-exact) or stably; boolean-mask rows by regex on a string column, by date range, by "list non-empty"; left-join two tables on `id`; per-row processing of a list-of-strings column; append integer columns; group rows into date bins and compute count/mean/var per bin; write TSV. That is all. There is no arithmetic across columns, no joins beyond one left merge on a unique key, no string ops beyond regex search.

### 3.2 Options

| | Hand-rolled column store | `polars` | `arrow` (arrow-rs) |
|---|---|---|---|
| Fit to the operations above | Exact; ~1,500 lines including reader/writer/sort | Everything present, but with polars semantics (stable sort, its own CSV inference, its own NaN/null model); a pandas-compat layer would still be written on top | Storage only; every operation still written by hand; null bitmap model differs from pandas NaN model |
| pandas dtype parity | Designed in (section 3.3) | Must convert polars dtypes (`Utf8`, `Null`, `Int64` nullable) to pandas defaults on every boundary crossing; nullable ints have no pandas-default equivalent (`read_csv` gives float64) | Same problem; zero-copy to pandas needs `pyarrow`, which is not a dependency, and pandas default dtypes are numpy-backed anyway |
| Build time (release, `lto = true`, `codegen-units = 1`) | ~1 min today for ~100 crates; +20–30 s for `regex` | Adds several hundred crates; typically +3–6 min with LTO | +100 crates; +1–2 min |
| Binary / wheel size | `.so` is 1.52 MB, wheel 774 KB today; expect +0.7–1.2 MB (`regex` with Unicode tables dominates) | Expect +15–30 MB | Expect +4–8 MB |
| Behavioural risk | Every rule is ours, tested against pandas | Two foreign semantics to reconcile (polars vs pandas) | One foreign semantic (arrow nulls vs NaN) |
| Reproducing numpy's unstable sort | Straightforward port | Impossible without bypassing polars | Same as hand-rolled |

Estimates in the table are order-of-magnitude from experience with those crates, not measurements; they are large enough that measuring would not change the decision. Decision: hand-rolled column store. The `regex` crate is the only new dependency (Unicode classes must match Python's `re`; `regex-lite` lacks them).

### 3.3 Type system

```rust
pub enum Column {
    Int64(Vec<i64>),                     // pandas int64
    UInt64(Vec<u64>),                    // pandas uint64 (read_csv overflow rule)
    Float64(Vec<f64>),                   // pandas float64, NaN = missing
    Bool(Vec<bool>),                     // pandas bool
    Str(DictStr),                        // pandas object column of str, NaN = missing
    StrList(Vec<Option<Vec<String>>>),   // object column of Python lists of str ("mutation instructions")
    DatetimeNs { ns: Vec<i64>, unit: Unit }, // datetime64[unit], i64::MIN = NaT; unit remembered for round trip
    PyObject(Vec<Py<PyAny>>),            // object column holding arbitrary Python objects (opaque)
    Foreign { series: Py<PyAny>, take: Vec<usize> }, // any other pandas dtype: kept as the original Series,
                                                     // row selection applied lazily with Series.array.take()
}

pub struct DictStr { codes: Vec<u32>, values: Vec<Arc<str>> } // u32::MAX = NaN; low-cardinality metadata columns

pub enum IndexKind { Range { start: i64, len: usize }, Labels(Vec<i64>) }

pub struct Table { columns: Vec<(String, Column)>, index: IndexKind }
```

Rules:

- Missing values follow pandas: `Str` missing materialises as `float('nan')` (not `None`), because that is what `read_csv` yields and what `.tolist()` shows; `StrList` missing materialises as `None` (only ever present transiently in `parse_mutation_data`, dropped before the user sees it).
- `Str` is dictionary-encoded. Metadata columns are low-cardinality (`betacoronavirus`, `Europe`, `?`); dictionary encoding cuts memory and, more importantly, lets `to_pandas` create one Python string per distinct value.
- `IndexKind` reproduces pandas' index exactly: `Range` after `reset_index`, `Labels` after `sort_values` or a boolean mask (pandas keeps the original labels). The materialiser emits `RangeIndex` or `Index(dtype=int64)` accordingly; the oracle test compares `type(df.index)` and the labels so that the `RangeIndex`-vs-`Index` rule for masks is pinned to what the pinned pandas does rather than to my memory of it.
- Every row operation (mask, permutation) is a `Vec<usize>` selection applied to all columns; for `Foreign` it is composed into `take`, so exotic dtypes survive any number of filters and sorts with their dtype intact.

Ingest (`from_pandas`) mapping by dtype: `int64→Int64`, `uint64→UInt64`, `float64→Float64`, `bool→Bool`, naive `datetime64[*]→DatetimeNs` (converted to ns; unit remembered), `object` → inspect elements: all `str`/NaN → `Str`; all `list`/`tuple` of `str` or `None`/NaN → `StrList`; otherwise `PyObject`; anything else (`category`, nullable `Int64`, `string[pyarrow]`, tz-aware datetimes, `timedelta64`, `period`, `interval`) → `Foreign`. A `MultiIndex` or a non-integer index is stored as `Foreign` index (same mechanism) — only `Range`/`Labels` are needed by the pipeline; `filter_by_position` and the sort produce `Labels` from whatever labels came in, which is what pandas does too.

Pipeline requirements on dtypes (checked at method entry with the same error type pandas would raise): `id` must be `Str`, `date` must be `DatetimeNs`, `mutation instructions` must be `StrList`, `N count` and the count columns must be `Int64`/`Float64`. Filter keys may be any column; `filter_columns` on a non-`Str` column raises the `AttributeError` pandas raises today.

## 4. Where the pandas boundary lives

### 4.1 The two options the brief asks to evaluate

Option A — materialise once at the end of `__init__` and after every mutating method; `data` stays a plain attribute in `__dict__`. Every method reads `self.data`, ingests to a `Table`, computes, and writes a new DataFrame back. Pros: no property, no state machine, in-place user mutation is visible because the DataFrame is the only truth. Cons: one ingest + one materialise per method (six in `__init__`, three in `analysis`); overrides work unchanged. Cost per conversion at 10,000 rows is small (4.4), so this is viable, but it does not let pandas become optional later, and it converts even when nobody looks.

Option B — `data` is a property whose getter materialises on access and whose setter ingests. Pros: zero conversions on the common path; pandas becomes optional later. Cons, if done naively: (1) `inst.data["x"] = ...` mutates a throw-away DataFrame and the change is silently lost; (2) `inst.data is inst.data` becomes `False`; (3) `count_mutation_types` today mutates the existing object in place, so `df = inst.data; inst.count_mutation_types(); "number of mutations" in df` is `True` today and would be `False`.

### 4.2 Chosen design: property with two-state ownership

The property plus a small state machine gives Option B's cost profile with Option A's semantics.

```rust
#[pyclass(subclass, extends = PyEvoMotionBase, name = "PyEvoMotionParser")]
pub struct PyEvoMotionParser {
    table: Option<Table>,        // Some in state R
    df: Option<Py<PyAny>>,       // Some in state P: the DataFrame handed out (or assigned) since the last Rust mutation
}
```

- State R (Rust-owned): `table = Some`, `df = None`. `.data` getter: materialise, store in `df`, return it (state P).
- State P (pandas-visible): `df = Some`. Getter returns the *same* object (identity stable, as today). Every Rust method entry calls `self.take_table()`: if `df` is present, ingest it into `table` (the user may have mutated it in place) — this is the only ingest the design ever performs, and it happens only if someone actually looked.
- Method exit: methods that *reassign* `self.data` today (`filter_columns`, `filter_by_daterange`, `filter_by_position`, `parse_data`, `drop_missing_instructions`, the loads in `__init__`) drop `df` (state R): the next getter returns a new object, exactly as pandas produced a new frame. Methods that *mutate in place* today (`count_mutation_types`, `n_filter`, `length_filter`) apply their effect to the handed-out DataFrame in place when one exists (`df[col] = np.ndarray`, `df.reset_index(drop=True, inplace=True)`) and stay in state P with the same object; otherwise they mutate the `Table`.
- Setter: a `DataFrame` → `df = Some(obj)`, `table = None` (so `inst.data = X; inst.data is X` is `True`, as today); a `Table` → `table = Some`, `df = None`; anything else → `TypeError`.
- `__init__`'s own internal reads (`ids` for `parse_data`, `iloc[0]["id"]` for `load_reference`, `min()` for `origin`, the columns check) use `take_table()` and never materialise. The public signature of `parse_data(input_fasta, selection: pd.Series)` is kept by building a one-column `Series` for `selection` (cheap; `get_differing_mutations` overrides read `selection.values`).

Because every method still dispatches through Python (`slf.call_method1`), an override that reads or writes `self.data` simply moves the instance into state P and back; no override detection is needed for correctness. One optimisation uses it: `parse_data` calls the Rust `get_differing_mutations` directly (Table output) when `type(self).get_differing_mutations is PyEvoMotionParser.get_differing_mutations` (method descriptors are identical when not overridden), and dispatches through Python plus ingest otherwise. The same check lets `analysis` consume the Rust `StatsTable` directly when `compute_stats` is not overridden.

### 4.3 Explicit conversion API (coordinator's addition)

(a) `PyEvoMotion.Table` Python class (`#[pyclass(name = "Table", module = "PyEvoMotion")]`):

```python
Table.from_pandas(df: pandas.DataFrame) -> Table      # staticmethod; exact ingest per 3.3
Table.to_pandas(self) -> pandas.DataFrame             # exact materialisation
len(table); table.columns -> list[str]; table.empty -> bool; table.dtypes -> dict[str, str]; repr(table)
Table.to_tsv(self, path, sep="\t")                    # phase 5, byte-identical to DataFrame.to_csv(index=False)
```

Round-trip guarantee (tested with `DataFrame.equals`, `dtypes.equals`, index type and labels, and `to_csv` bytes): for every DataFrame that PyEvoMotion itself can produce — `read_csv(sep)` output of any TSV/CSV, with `date` converted by `to_datetime`, with `mutation instructions` converted to Python lists, with int64 count columns appended, after any sequence of masks/sorts/resets — `Table.from_pandas(df).to_pandas()` equals `df`. Concretely: object/str columns with NaN (NaN stays `float('nan')`, strings stay `str`), int64, uint64, float64 (bit-exact, NaN preserved), bool, `datetime64[ns]` with NaT, object columns of lists of str, `RangeIndex` vs int64 `Index`. For columns outside that set the guarantee is weaker but still strong: `Foreign` returns `series.array.take(rows)` so dtype and values survive any row selection; `PyObject` returns the same Python objects.

(b) The `data` property as specified in 4.2: getter materialises (cached per state), setter accepts a `DataFrame` or a `Table`. The classic override idiom

```python
def filter_columns(self, filters):
    super().filter_columns(filters)
    self.data = self.data[self.data["host"] == "Human"]
```

works unchanged: the getter materialises, pandas builds a new frame, the setter stores it (state P), the next Rust stage ingests it.

(c) Helpers on `PyEvoMotionBase`: only `_check_dataset_is_not_empty(df_or_table, msg)` is widened to accept either (it is called from `__init__` and from `adjust_model`). No `with_dataframe()` context manager and no `_as_table` helper: the property already covers in-place edits, and extra API is surface to keep byte-compatible forever. `date_grouper` stays exactly as it is (a pandas utility used by test helpers); the pipeline no longer calls it (today's Rust `compute_stats` already calls the Rust static function rather than dispatching, so overriding `date_grouper` has no effect on `compute_stats` before or after this change).

### 4.4 Performance cost per conversion (10,000 rows)

Measured proxies on this machine, pandas 2.3.3, on a synthetic 10,000 × 30 frame built from test1's 28 metadata columns plus a 30-element `mutation instructions` list per row and `N count` (270,000 object cells, 300,000 strings inside lists):

| Operation | Measured |
|---|---|
| Pull every column out of the DataFrame into Python lists/arrays (`tolist()`/`to_numpy()`) | 1.3 ms |
| Rebuild an equal DataFrame from arrays and 1-D object arrays | 5.3 ms |
| `DataFrame.to_csv(sep="\t", index=False)` (9.7 MB) | 99 ms |
| `read_csv` of the same text | 43 ms |

The Rust side adds the Python-object ↔ Rust conversions: 270,000 `str` cells plus 300,000 list strings at roughly 50–150 ns each. Estimated `from_pandas`: 30–80 ms without dictionary de-duplication, 10–25 ms with it (hash each `PyUnicode` pointer once; pandas shares identical string objects within a column only sometimes, so hash by value). Estimated `to_pandas`: 20–60 ms (one `PyString` per distinct value per column, one `PyList` per row for the list column, then the 5 ms DataFrame constructor). Numeric and datetime columns are single `memcpy`s into numpy arrays via the `numpy` crate already in `Cargo.toml`. For comparison, aligning 10,000 sequences takes minutes; a whole `-load` run on 10,000 rows currently spends ~150 ms in pandas I/O and the rest in Python-object shuffling. Conversions happen at most twice per `analysis()` (CLI: one `to_pandas` for the export, one `from_pandas` at `analysis` entry because the export accessed `.data`), so the boundary costs well under 0.2 s at 10,000 rows and scales linearly.

### 4.5 Subclass patterns: covered and not covered

Covered, unchanged:

- Overriding any pipeline method and using `self.data` as a DataFrame inside it, including `self.data = self.data[mask]`, `self.data.reset_index(drop=True, inplace=True)`, `self.data["col"] = ...` (in-place edits are picked up by the next Rust stage because state P re-ingests).
- A bare subclass with its own `__init__` that sets `self.reference` and `self.data = pd.DataFrame(...)` then calls individual methods (the `Bare` test).
- Overriding `parse_mutation_data`, `get_differing_mutations`, `compute_stats` to return DataFrames (they are ingested).
- Overriding `generate_alignment`, `create_modifs`, `_run_mafft`, `_column_decision`, `_get_consecutives`: these never touch the table; `get_differing_mutations` still dispatches `generate_alignment` and `create_modifs` through `type(self)` per record, and `generate_alignment` dispatches `_run_mafft` through `cls`.
- Overriding `linear_regression`, `power_law_fit`, `adjust_model`, `_remove_nan`, `AIC`, `F_test`: unchanged (`analysis` calls `linear_regression`/`adjust_model` as Rust associated functions today, which already bypasses overrides — this design does not change that; `adjust_model` internally dispatches via `cls`).
- Reading `.origin`, `.dt`, `.dt_ratio`, `.reference`, `.verbose`: plain attributes, unchanged, same Python types (`origin` stays `Timestamp` or `datetime` exactly as today).

Changed but compatible:

- `type(instance).data` is now a property object instead of being absent from the class; `instance.__dict__` no longer contains `"data"`. Code doing `vars(instance)["data"]` breaks (no such code exists in the repository).
- Overriding `parse_metadata` has no effect on `__init__` today (it calls `parse_metadata_inner` directly); this remains so. Documented rather than "fixed", to avoid a behaviour change.

Not covered:

- Passing a `Table` where pandas is expected by *user* code (e.g. calling `date_grouper(table, ...)`): `date_grouper` remains pandas-only.
- Subclasses that stored non-DataFrame objects in `self.data` (e.g. a dict) and only used their own methods: the setter now raises `TypeError` on assignment instead of failing later. No such usage exists in the repository or docs.
- Pickling instances: not supported by the 0.2.0 extension classes and still not supported.

### 4.6 Latent bug fixed by construction

`count_mutation_types` on a `Table` computes the counts row-wise, as the original Python `apply` did. This differs from the current Rust build only when `data.index` has gaps (rows removed by `filter_by_position`, or by `-dr`/`-f` in `-load` mode followed by `-recount` or a file without count columns), where the current build silently produces float64 columns with NaN. No bundled dataset has gaps, so the goldens are unaffected; a regression test is added (8.4). This is listed in the CHANGELOG under "Fixed".

## 5. Method-by-method mapping

Legend: **Rust** = pure Rust on `Table`; **wrapper** = thin pandas call at the boundary; **unchanged** = not touched by this work.

### 5.1 Parser

| Method | Becomes | How semantics are preserved |
|---|---|---|
| `__init__` | Rust orchestration on `Table`, dispatch through Python kept | Same stage order, same log messages, same `verbose` attribute. `data` set via the internal table, not the property. |
| `parse_metadata_inner` | Rust: `csv::read(path, sep)` → `dates::to_datetime(col)` → `sort::numpy_argsort` → `Table` with `Labels` index | Section 6.1–6.3. Errors: unknown extension (`ValueError`, existing message), missing `date` (existing message), missing file (`FileNotFoundError`, pandas-style `[Errno 2] No such file or directory: '<path>'`), decode errors (`UnicodeDecodeError`), ragged long rows (`pandas.errors.ParserError` — raised by importing pandas' exception class; message copied from pandas' format `Error tokenizing data. C error: Expected N fields in line L, saw M`). |
| `parse_metadata` (public staticmethod) | wrapper: Rust read, then `to_pandas()` | Returns the same DataFrame (labels preserved, unsorted RangeIndex-derived labels). |
| `parse_mutation_data` (public staticmethod) | Rust read + Rust list-literal parser + ISO dates + **stable** sort + `Range` index; `to_pandas()` for the public return; the `__init__` path uses the `Table` directly via an internal function | Existing error messages kept verbatim. The literal parser accepts `[...]`/`(...)` of Python string literals (single/double quotes, `\\`, `\'`, `\"`, `\n`, `\t`, `\xNN`, `\uNNNN`); anything else raises the existing "Could not parse"/"not a list of strings" messages. Empty cell → missing → dropped by `drop_missing_instructions` with the same warning text. |
| `drop_missing_instructions` | Rust | Same stderr text, same "only reset when something dropped" rule. |
| `load_reference` | Rust: first row of `id` from the `Table` | Empty table raises `IndexError("single positional indexer is out-of-bounds")` to match today. |
| `filter_columns` | Rust: regex mask per key (6.4) | Keys not in `columns` skipped; `str`/`list[str]` values; `*`→`.*`, `|` join; NaN cell → `ValueError("Cannot mask with non-boolean array containing NA / NaN values")`; non-`Str` column → `AttributeError("Can only use .str accessor with string values!")`. Index becomes `Labels`. |
| `filter_by_daterange` | Rust; `start`/`end` coerced with one `pandas.Timestamp(x).value` call each (wrapper) | Same `max/min` clamping, same inclusive mask, same `ValueError`, same no-error behaviour on empty data (NaT comparisons are `False`). Accepts `datetime`, `Timestamp`, `date`, `numpy.datetime64` — whatever `pd.Timestamp` accepts, exactly as today's comparisons did. |
| `filter_by_position` | Rust | Same early return on empty, same errors, same window rules, same sentinel logic, index stays `Labels` (not reset). |
| `parse_data` | Rust left merge on `id` (first match wins, as pandas does when keys are unique; duplicated ids in the right table multiply rows in pandas — the right table is built from a `HashSet` of ids so it has no duplicates), `Range` index, then `drop_missing_instructions` | `selection` argument stays a `pd.Series` (built cheaply). |
| `get_differing_mutations` | Rust core producing `(ids, lists, n_counts)`; public method wraps into a DataFrame; `parse_data` uses the core directly unless overridden | Progress output and early stop unchanged. |
| `read_fasta`, `parse_sequence_by_id`, `generate_alignment`, `create_modifs`, `_run_mafft`, `_column_decision`, `_get_consecutives` | unchanged | |

### 5.2 Core

| Method | Becomes | How semantics are preserved |
|---|---|---|
| `PyEvoMotion.__init__` | Rust on `Table` | `origin`: `Timestamp(min)` built through pandas (one call) unless `-dr` start is earlier, in which case the original Python object is stored — the same `min()` outcome and the same type as today. `has_counts` check on table columns. |
| `count_mutation_types` | Rust; in state P applied in place to the handed-out DataFrame (`df[col] = np.int64 array`) | Row-wise counts; 4.6. |
| `get_lengths` | Rust; returns `pd.Series(int64)` with `RangeIndex` (wrapper) | Same values, same type (`test_get_lengths_returns_series`). |
| `length_filter`, `n_filter` | Rust: validate `how`/`threshold` with the same errors, compute (and discard) the mask by calling `get_lengths` through Python (so overrides still run), then reset the index (`Range`), in place on the DataFrame in state P | The no-op is preserved deliberately; the only visible effect (index reset) is preserved. |
| `compute_stats` | Rust (6.5–6.8) for Tick frequencies; falls back to today's pandas code path for non-Tick `DT` (e.g. `"W"`, `"MS"`) | Returns a DataFrame (`date` datetime64[ns] column, float64 mean/var columns in the same order, int64 `size`, `RangeIndex`); `analysis` uses the Rust `StatsTable` when `compute_stats` is not overridden. |
| `analysis` | Rust for the stats handling (`weights`, column loop, `var - min`, `dt_idx`); regressions unchanged | `dt_idx = (ns - ns_min) as f64 / (7 days in ns) as f64`, the same double division numpy performs for `timedelta64 / timedelta64`. Plot calls unchanged, receiving `stats[[...]]` DataFrame subsets as today. |
| `_apply_scaling_correction_to_model`, `_mutation_type_switch`, `plot_results`, `export_plot_results` | unchanged | |

### 5.3 Base and CLI

| Item | Becomes |
|---|---|
| `date_grouper` | unchanged (pandas utility). |
| `_get_time_ratio`, `_verify_dt` | unchanged (two `pd.Timedelta` calls at construction). The Rust binner parses `DT` itself into nanoseconds for the Tick forms `^\s*(\d+)?\s*(D|d|h|H|min|T|s|S|ms|L|us|U|ns|N)\s*$` (pandas aliases); `W`/`w` is a `Tick` for `Timedelta` but an anchored `Week` for `Grouper`, so it takes the pandas fallback. |
| `_check_dataset_is_not_empty` | accepts `DataFrame` or `Table`. |
| `cli.rs` data export | phases 1–4: `instance.data.to_csv(...)` unchanged; phase 5: `Table.to_tsv` (6.9) once byte-identical on every CI platform. `-load` mode still does not rewrite. |
| `cli.rs` stats export | keep `stats.to_csv` (tiny frame; no benefit in porting) unless the pandas dependency is dropped later, in which case the same Rust writer handles it. |
| JSON export | unchanged. |

## 6. Exact-semantics specifications

### 6.1 TSV/CSV reading (pandas `read_csv` defaults)

Tokeniser: separator `\t` or `,` as today; quote char `"`, `doublequote=True`, no escape char; line terminators `\n`, `\r\n`, `\r`; UTF-8 with an optional BOM stripped from the first header; blank lines skipped; the first non-blank line is the header. Header names: kept verbatim (no strip); empty → `Unnamed: <position>`; duplicates → `name`, `name.1`, `name.2`. Rows shorter than the header are NaN-padded; rows longer raise `ParserError`.

Per-column dtype inference, applied to the cell texts of the whole column, in this order (a cell matching the default NA set counts as missing for every branch): all non-missing cells parse as `i64` (surrounding whitespace allowed) → `Int64` if no missing, else `Float64`; all parse as integers but some exceed `i64` and all fit `u64` → `UInt64`; all parse as floats (`1e3`, `inf`, `-inf`, `nan`; whitespace allowed) → `Float64`; all in `{True, TRUE, true, False, FALSE, false}` → `Bool` if no missing, else `PyObject` of Python bools and NaN (materialised as object dtype, as pandas does); otherwise `Str` with missing = NaN. A column that is entirely missing is `Float64`. Default NA set: `""`, `#N/A`, `#N/A N/A`, `#NA`, `-1.#IND`, `-1.#QNAN`, `-NaN`, `-nan`, `1.#IND`, `1.#QNAN`, `<NA>`, `N/A`, `NA`, `NULL`, `NaN`, `None`, `n/a`, `nan`, `null`.

Float parsing: Rust's correctly rounded `str::parse::<f64>`. pandas' default `float_precision="high"` is documented as correct in "most" cases; a differential test over the fixtures and a hypothesis test over random decimal strings decide whether a port of pandas' `precise_xstrtod` is necessary (risk R2).

### 6.2 Date parsing (`to_datetime` inference)

Rust fast path when the first non-missing string matches one ISO shape, applied strictly to every cell: `YYYY-M?M-D?D`, `YYYY-M?M-D?D[ T]HH:MM:SS(.f{1,9})?`, `YYYY-MM`, `YYYYMMDD`. Any cell that does not match the shape of the first, or any first value outside these shapes, triggers the fallback: the whole column is handed to `pandas.to_datetime(Series)` (one call) so that formats such as `31/03/2020` or `March 30, 2020`, and every error message (`time data "…" doesn't match format "%Y-%m-%d"…`), are pandas' own. Missing → NaT (`i64::MIN`). Result unit is `ns` in both paths (pandas 2.x behaviour; the oracle test pins it).

### 6.3 Sorting

`parse_metadata_inner` and the `compute_stats` re-sort use `sort::numpy_argsort(ns: &[i64]) -> Vec<usize>`: a line-by-line port of numpy's `aquicksort_<npy::datetime_tag>` (`numpy/_core/src/npysort/quicksort.cpp`) with `SMALL_QUICKSORT = 15`, `PYA_QS_STACK` bookkeeping, median-of-three pivot with the three conditional swaps, Hoare partition using strict `<`, "push the larger partition" rule, insertion sort for `pr - pl <= 15`, depth limit `2 * floor(log2 n)` and `aheapsort_` fallback, and numpy's `less(a, b) = a < b || (b == NaT && a != NaT)`. pandas removes NaT rows before calling numpy and appends them last (`na_position="last"`), which the Rust wrapper reproduces. Validation performed during this study with a Python transliteration of the same pseudo-code: identical to `np.argsort(kind="quicksort")` on the `datetime64[ns]` arrays of test1 (101 rows), S1 (2001), test3-ci UK and USA (1001 each), 100,000 random values over 300 distinct dates, an organ-pipe array of 10,000 (which reaches the heapsort fallback 48 times), a sawtooth of 20,000, a median-of-three killer of 4,000, reversed ties of 12,000, and n = 2, 16, 17. numpy does not SIMD-dispatch the datetime argsort (it does for plain int64 on x86 with AVX-512/AVX2), which is why the datetime path is portable; the oracle test on the x86_64 runner confirms this on CI. Should it ever fail there, the fallback is a one-line boundary call to `numpy.argsort` on the ns array (risk R3).

`parse_mutation_data` uses a stable sort (`slice::sort_by_key` is stable), matching `kind="stable"`.

### 6.4 Regex filters

Pattern construction as today. Compile with `regex::Regex::new`; on success, mask = `re.is_match(cell)` (unanchored search, like `re.search`). On a compile error (lookaround, backreferences, possessive quantifiers, `(?P=name)`) fall back to Python: `re.compile(pattern)` and `search` per cell through PyO3 — identical semantics at pandas' speed. Known residual differences when both compile: Python's `$` also matches before a trailing `\n`; Rust's does not without `(?m)`. Cells cannot contain newlines unless the input was quoted, so the differential test includes such a case and the fallback is forced when the pattern contains `$` and the column contains `\n`. `\d`, `\w`, `\s`, `\b` are Unicode-aware in both engines; `.` excludes `\n` in both. Empty pattern matches everything in both.

### 6.5 Left merge and missing instructions

`parse_data`: for each left row look up `id` in a `HashMap<&str, usize>` built from the alignment results; unmatched rows get missing `StrList`/`Float64` cells (pandas gives NaN for `N count` there, hence `Float64` in that transient state; after the drop the column is `Int64` again — pandas gets int64 back only because the merged column was int64 before the NaN appeared? No: pandas leaves `N count` as float64 if any NaN appeared, then `drop_missing_instructions` removes those rows but the dtype stays float64). This is a real subtlety: today, when at least one metadata id is missing from the FASTA, `N count` is written as `291.0`, `0.0`, … The Table reproduces it: the merge produces `Float64` for `N count` whenever a row was unmatched, `Int64` otherwise. The test `test_metadata_ids_missing_from_fasta_are_dropped_with_warning` gains an assertion on `N count` dtype/text to pin this.

### 6.6 Time bins

For `f` = frequency in ns, `o` = origin in ns (a `Timestamp`'s `.value`, or the `datetime` converted the same way), over the non-NaT dates of the (possibly duplicated and re-sorted) rows: `first = min - pymod(min - o, f)`, `last = max + (f - r)` where `r = pymod(max - o, f)` if `r != 0` else `max + f`, with `pymod` the floor modulo. Bin `k` for a row is `(date - first) / f` (floor), `k in 0..(last-first)/f`; label `first + k f`. Rows with NaT get no bin. `filter(len >= 2)`: keep rows whose bin has at least two rows, in original order; if none remain raise the existing `ValueError`. Recompute bins on the kept rows (new `min`/`max`). Output one row per bin including empty ones.

Validated against `pd.Grouper` on: origin later than the data (`2020-01-05` with min `2020-01-01` → first bin `2019-12-29`), NaT rows (excluded, empty bin kept), and the 3-day golden run of test1 (bins identical).

### 6.7 Aggregations

Values are the count columns converted to `f64` (pandas casts int64 to float64 for `mean`/`var`). Per bin, in row order:

- `size`: row count (`i64`).
- `mean`: Kahan-compensated sum divided by count (pandas `group_mean`); for integer-valued inputs any method is exact, the Kahan form is kept for parity on float inputs.
- `var` (ddof = 1): Welford — `n += 1; old = mean; mean += (x - old) / n; m2 = fma(x - mean, x - old, m2)` on `aarch64`, and `m2 += (x - mean) * (x - old)` elsewhere; `var = m2 / (n - 1)`; NaN for `n <= 1` (only empty bins after the filter). The `cfg(target_arch)` split mirrors how the pandas wheels were compiled (clang contracts on Apple Silicon and GCC on Linux aarch64; x86_64 manylinux wheels are baseline SSE2 without FMA; MSVC does not contract by default). The oracle test asserts equality with `groupby(Grouper).var()` on the running platform; a mismatch on some platform flips that platform's `cfg` and is a one-line fix.

Evidence: on this machine pandas equals Welford+FMA on 3000/3000 random integer groups and 1000/1000 float groups, plain Welford on 2353/3000; the variance depends on within-group row order (18 of 51 test1 groups change value when reversed), which is why 6.3 must be exact.

### 6.8 `analysis` details

`weights = size` as f64; `mean` columns → `linear_regression(index as f64, mean, weights)` after `_remove_nan`; `var` columns → `adjust_model(index, var - min(var), weights)`; `dt_idx[k] = (label_k - label_0) as f64 / 604_800_000_000_000f64` — the same `(double)a / (double)b` numpy performs; both casts are round-to-nearest in C and Rust. Materialise `stats` once at the end (or, if `compute_stats` was overridden, keep the ingested DataFrame and append `dt_idx` with pandas as today).

### 6.9 TSV writing (`DataFrame.to_csv(sep, index=False)` parity)

- Header: column names joined by `sep`, quoted by the same rule as fields. Line terminator `\n`. No trailing separator.
- `Int64`/`UInt64`: decimal. `Bool`: `True`/`False`. Missing anywhere: empty field (`na_rep=""`), except a NaT in a datetime column which pandas writes as `""` (two quote characters) — reproduce.
- `Float64`: Python `repr`. Algorithm: obtain the shortest round-trip digit string and decimal exponent (Rust's `{:e}` formatting gives exactly the shortest digits, `d.ddde±X`); let `e` be the exponent of the leading digit; if `-4 <= e < 16` write fixed notation with at least one fractional digit (`1.0`, `6.249999999999999`, `0.0001`), otherwise write `d[.ddd]e±XX` with a signed exponent of at least two digits (`1e-05`, `1e+16`, `1.2345678901234568e+17`); `inf`, `-inf`; `-0.0` keeps its sign. Property test: `repr(float)` over hypothesis-generated doubles including subnormals and integers ≥ 1e16.
- `DatetimeNs`: if every non-NaT value is at midnight write `%Y-%m-%d`, else `%Y-%m-%d %H:%M:%S`; sub-second values take the pandas fallback (they cannot arise from the supported `to_datetime` path unless the metadata has fractional seconds; then the whole export uses pandas, unchanged from today).
- `Str`: verbatim; quote (with `"` doubling) if the field contains `sep`, `"`, `\n` or `\r`. pandas' C writer additionally quotes fields that *start* with the quote character? No — QUOTE_MINIMAL quotes only on the characters above; an empty string `""` (non-missing) is written as `""`. Pinned by the oracle.
- `StrList`: Python `repr` of a list of str: `[]`, elements separated by `", "`, each element in single quotes unless it contains `'` and no `"` (then double quotes); backslash and non-printable escapes as Python's `str.__repr__` (`\\`, `\n`, `\t`, `\xNN` for other C0/DEL, `\uNNNN`/`\UNNNNNNNN` for non-printable code points; printable non-ASCII kept). Then the field-quoting rule applies (a list containing `"` gets the whole field quoted with doubled quotes — observed).
- `PyObject`/`Foreign` columns: `str(obj)` for object cells is not generally reproducible; the writer falls back to `DataFrame.to_csv` for the whole table if any such column is present (only a subclass can create one).

The writer lands last (phase 5) and is switched on only when it is byte-identical to pandas on every fixture and every CI platform; the CLI keeps `to_csv` until then.

## 7. Risks and verification

| # | Risk | Verification / mitigation |
|---|---|---|
| R1 | Float formatting deviates from Python `repr` | Hypothesis test `repr(x) == table_format(x)` over 1e6 doubles incl. edge classes; golden bytes of `*_stats.tsv`. |
| R2 | Float *parsing* differs from pandas' `xstrtod` in the last ulp | Differential test: parse every numeric cell of every fixture with both; hypothesis test over random decimal strings (`d+.d+`, exponents). If any mismatch: port `precise_xstrtod` (≈150 lines). |
| R3 | numpy tie order differs on some platform (SIMD dispatch, numpy change) | Oracle test `np.argsort(dates, kind="quicksort") == sort::numpy_argsort` on the fixtures, run on ubuntu x86_64 (CI) and in the release smoke matrix (Linux x86_64/aarch64, macOS Intel/ARM, Windows). Fallback: call `numpy.argsort` at the boundary (one call). |
| R4 | Variance rounding (FMA) differs per platform | Oracle test `groupby(Grouper).var()` vs Rust on fixtures and random groups per platform; `cfg(target_arch)` switch. Documented in README as a property the current implementation already has. |
| R5 | Date parsing: non-ISO formats, mixed formats, error messages | Fast path only for ISO shapes with a strict whole-column check; everything else goes to `pandas.to_datetime`, so messages are pandas'. Differential test on ISO variants, missing values, `2020-3-5`, `T` separator. |
| R6 | Regex dialect differences | Compile-failure fallback to Python `re`; `$`+newline rule; differential test with pandas `str.contains` on a corpus of patterns (`*` wildcards, alternation, dots, anchors, character classes, Unicode). |
| R7 | NaN handling in metadata columns (`object` with NaN, `float64` all-NaN, bool with NaN, `N count` float64 after an unmatched merge) | Round-trip tests (`equals`, dtypes, `to_csv` bytes) on read → Table → DataFrame for every fixture plus synthetic TSVs generated by hypothesis (random column types, random missing cells, quoted fields, CRLF, BOM). |
| R8 | Empty-dataset paths | Unit tests for: metadata with header only (`IndexError` from `load_reference`), filters that empty the data (`filter_by_position` early return, "dataset empty" message), `compute_stats` with no bin ≥ 2 rows, `-load` with all instruction cells empty. |
| R9 | `-load` path | Round trip `instance.data.to_csv → -load` equality tests already exist (`test_load_*`); add: a 0.1.x-style TSV with `N count` floats, a TSV with times in `date`, `-load` + `-dr` + `-recount` (exercises 4.6). |
| R10 | Index type/labels parity (`RangeIndex` vs `Index`) | Oracle compares `type(df.index)` and labels after each stage on test1. |
| R11 | Subclass state machine (identity, in-place edits) | Tests: `df = inst.data; inst.count_mutation_types(); df is inst.data and "number of mutations" in df`; `inst.data["x"] = 1; inst.filter_by_position(0,0); "x" in inst.data.columns`; `inst.data = df2; inst.data is df2`; `Bare` pattern; override with `self.data = self.data[mask]`. |
| R12 | Pinned pandas drifts (`pyproject` allows `>=2.2.2,<3`) | Oracle tests run against the locked version in CI; README states that ulp-level variance parity is guaranteed for the locked pandas and platform. |
| R13 | Compile time / wheel size creep | Measure `.so` size in CI (informational step, like the alignment throughput step); budget +1.5 MB. |

## 8. Test strategy

### 8.1 Goldens must be regenerated first

The committed `tests/data/test1/output/out.tsv` predates 0.2.0 (indel positions are 0-based: `d_29851_…` vs the current `d_29852_…`; identical ids, order and counts otherwise). The committed `out_stats.tsv` was produced with pandas that computed a different variance rounding: for the bin `[1, 1, 2, 1, 1, 1]` it holds `0.1666666666666666`, which neither the current arm64 pandas (`0.16666666666666669`) nor plain Welford nor numpy's two-pass produce; overall 47/51 variance cells match the current arm64 arithmetic and 46/51 match the x86-style arithmetic. Byte comparison against these files is therefore not a valid acceptance test for anything. Phase 0 regenerates `<out>.tsv`, `<out>_stats.tsv` and `<out>_regression_results.json` for test1 (default args and the documented `-dt 3D -k all` run), test4 S1/S2, test5 linear_01/powerlaw_01 (CI sample) and test3-ci UK/USA (fixed-seed figure ids) with the current 0.2.0 build **on the CI runner** (ubuntu x86_64), commits them, and adds a golden test that compares bytes. A second set generated on macOS arm64 is stored alongside for local runs (`*.arm64.tsv`), selected by `platform.machine()`; both sets are expected to differ only in variance ulps, and a test asserts exactly that (same rows, same columns, `numpy.isclose` everywhere, bytes equal outside `var` columns).

### 8.2 Same-machine pandas oracle

`tests/test_table_oracle.py` re-implements the current pipeline's pandas steps in ~60 lines (`read_csv`, `to_datetime`, `sort_values`, `str.contains` masks, date mask, left `merge`, `Grouper` `mean`/`var`/`size`, `to_csv`) and compares, stage by stage, with the Rust `Table` results on the fixtures and on hypothesis-generated inputs. It is platform-agnostic by construction (both sides run on the same numpy/pandas). It also drives R3, R4, R7, R10.

### 8.3 Property tests

Python-side with hypothesis (pytest, dev dependency): random TSVs (column type mix, NA tokens, quotes, CRLF, BOM, duplicate/empty headers, ragged rows) → `read_csv` vs Rust reader (values, dtypes, headers, errors); random `datetime64` arrays with heavy ties → argsort parity; random float groups → mean/var parity; random doubles → `repr` parity; random lists of strings (including quotes, backslashes, non-ASCII, control chars) → list `repr` parity; random DataFrames from the supported dtype set → `from_pandas → to_pandas` equality. Rust-side `cargo test` unit tests for the sort (against a naive stable sort where ties are absent), the modulo/bin arithmetic, the literal parser and the float formatter (against a table of known `repr` outputs).

### 8.4 Existing tests: impact

- `rust/tests/test_core_smoke.py`: all pass unchanged. `test_filter_by_position_window_rules` (bare subclass, `p.data = pd.DataFrame(...)`) exercises the setter. Add: state-machine tests (R11), `-load -dr -recount` (4.6), `N count` dtype after an unmatched merge (6.5).
- `rust/tests/test_parser_smoke.py`: unchanged; `test_parse_metadata_tsv` keeps working (`to_pandas()` of the sorted table with labels).
- `tests/test_core.py`, `tests/test_parser.py`: unchanged (they only use `.data` as a DataFrame and `to_csv`).
- `tests/test_UK_USA_dataset.py`, `tests/test_synthetic_datasets.py`, helpers: unchanged (`date_grouper` stays pandas).
- New: `rust/tests/test_table.py` (Table API, round trips), `tests/test_golden_outputs.py` (8.1), `tests/test_table_oracle.py` (8.2), hypothesis suites (8.3).
- CI (`.github/workflows/ci.yml`): add the golden/oracle tests to the default `pytest` run (they take seconds); add `pandas` to the release smoke-test install so the oracle runs in the wheel matrix; add an informational `.so` size step.

## 9. Phased implementation plan

Each phase is one or two commits and leaves `pytest` green; the CLI output is unchanged after every phase (byte-identical goldens are asserted from phase 0 on).

| Phase | Content | Commits | Effort |
|---|---|---|---|
| 0 | Regenerate goldens on CI and locally (8.1); golden byte test; oracle harness against the *current* code; CI wiring; README note on platform-dependent variance ulps | 2 | 1 day |
| 1 | `rust/src/table.rs`: `Column`, `DictStr`, `IndexKind`, `Table`, `from_pandas`/`to_pandas`, Python `Table` class, `__len__`/`columns`/`empty`/`dtypes`/`repr`; `data` property with the two-state ownership on `PyEvoMotionParser`; `_check_dataset_is_not_empty` accepts both; round-trip and state-machine tests. Pipeline methods still call pandas through the property (they behave exactly as before) | 2–3 | 3 days |
| 2 | `rust/src/csv_read.rs` (6.1), `rust/src/dates.rs` (6.2), `rust/src/sort.rs` (6.3) with unit + property + oracle tests; wire `parse_metadata_inner` and `parse_mutation_data` to build `Table`s; public staticmethods materialise | 3 | 3 days |
| 3 | Filters on `Table`: `filter_columns` (6.4, `regex` dependency), `filter_by_daterange`, `filter_by_position`, `drop_missing_instructions`, `parse_data` merge (6.5), `load_reference`; `count_mutation_types`, `get_lengths`, `length_filter`, `n_filter`; `PyEvoMotion.__init__` origin/counts on `Table`; override-detection helper for `get_differing_mutations` | 3 | 2.5 days |
| 4 | `compute_stats` on `Table` (6.6–6.7) with pandas fallback for non-Tick `DT`; `StatsTable`; `analysis` consumes it (6.8); oracle for bins/mean/var on fixtures and random groups; per-platform FMA check | 2 | 2.5 days |
| 5 | `rust/src/csv_write.rs` (6.9): float `repr`, list `repr`, datetime rule, quoting; `Table.to_tsv`; byte tests vs `to_csv` on fixtures and hypothesis; switch `cli.rs` data export to it (behind an env var first, default after one green release cycle) | 2 | 2 days |
| 6 | Docs: `rust/README.md` (module table, Table API, subclass guide), `CHANGELOG.md` (Fixed: count alignment; Internal: table; Notes: platform ulps), Sphinx API page for `Table` | 1 | 0.5 day |
| Total | | ~15 | 14–15 engineer-days |

Future (not in this plan, enabled by it): import pandas lazily so that `PyEvoMotion` works without pandas when `.data`/`analysis()` DataFrames are not requested; an opt-in "canonical" mode with a stable sort and platform-independent variance, announced as a behaviour change.

## 10. What changes for whom

Python users (library and CLI): nothing observable. Same constructor keywords, same `instance.data` DataFrame (columns, dtypes, row order, values, index), same `analysis()` return types and contents, same CLI files byte-for-byte on the same platform, same error messages on the documented error paths (`-load` errors, "dataset empty", date-range and position errors, missing `date` column). Construction and `-load` are faster; alignment time, which dominates, is unchanged. One correction: `-load` combined with `-dr`/`-f` and `-recount` (or a loaded file lacking the count columns) now yields correct integer counts where 0.2.0 produced NaN-holed float columns. The variance columns of `_stats.tsv` remain identical to what the same machine produced before; they were already different in the last digit between Apple Silicon and x86_64 machines and remain so (documented).

Subclass authors: every documented override keeps working, including reading and assigning `self.data` as a DataFrame, in-place edits and the bare-subclass pattern. New, optional: `PyEvoMotion.Table`, `Table.from_pandas`, `Table.to_pandas`, and `self.data = Table` for authors who want to stay in Rust-land. Visible differences: `PyEvoMotionParser.data` is a property (so `"data" in vars(instance)` is `False`); `parse_metadata` overrides still do not affect `__init__` (unchanged, now documented); `type(stats.index)`/`type(instance.data.index)` are pinned by tests to match pandas rather than guaranteed by construction.

## Appendix A. numpy `aquicksort_` port (reference)

```rust
const SMALL_QUICKSORT: usize = 15;
fn less(a: i64, b: i64) -> bool { a < b || (b == NAT && a != NAT) }   // NAT = i64::MIN
pub fn numpy_argsort(v: &[i64]) -> Vec<usize> {
    let n = v.len(); let mut t: Vec<usize> = (0..n).collect();
    if n == 0 { return t; }
    let (mut pl, mut pr) = (0usize, n - 1);
    let mut stack: Vec<(usize, usize)> = Vec::new(); let mut depth: Vec<i32> = Vec::new();
    let mut cdepth: i32 = (usize::BITS - 1 - n.leading_zeros()) as i32 * 2;   // 2 * floor(log2 n)
    loop {
        if cdepth < 0 { aheapsort(v, &mut t[pl..=pr]); }
        else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                if less(v[t[pm]], v[t[pl]]) { t.swap(pm, pl); }
                if less(v[t[pr]], v[t[pm]]) { t.swap(pr, pm); }
                if less(v[t[pm]], v[t[pl]]) { t.swap(pm, pl); }
                let vp = v[t[pm]]; let (mut pi, mut pj) = (pl, pr - 1); t.swap(pm, pj);
                loop {
                    pi += 1; while less(v[t[pi]], vp) { pi += 1; }
                    pj -= 1; while less(vp, v[t[pj]]) { pj -= 1; }
                    if pi >= pj { break; } t.swap(pi, pj);
                }
                t.swap(pi, pr - 1);
                if pi - pl < pr - pi { stack.push((pi + 1, pr)); pr = pi - 1; } else { stack.push((pl, pi - 1)); pl = pi + 1; }
                cdepth -= 1; depth.push(cdepth);
            }
            for pi in (pl + 1)..=pr {                       // insertion sort
                let vi = t[pi]; let vp = v[vi]; let mut pj = pi;
                while pj > pl && less(vp, v[t[pj - 1]]) { t[pj] = t[pj - 1]; pj -= 1; }
                t[pj] = vi;
            }
        }
        match stack.pop() { None => break, Some((l, r)) => { pl = l; pr = r; cdepth = depth.pop().unwrap(); } }
    }
    t
}
// aheapsort: numpy's 1-based sift-down heapsort on the index slice, using the same `less`.
```

Note `pi - 1` when `pi == 0` cannot occur because `pi >= pl + 1` after the first increment; `pr - 1` is safe because `pr - pl > 15`.

## Appendix B. Probe summary (this study, macOS arm64, CPython 3.12.8, pandas 2.3.3, numpy 2.4.6)

- Fixtures: test1 101 rows / 28 columns, dates all `YYYY-MM-DD`, `location` NaN ×98, `length` int64; test4 and test5 2001 rows / 2 columns; test3-ci 1001 rows / 8 columns with `Unnamed: 0`. No fixture row is dropped by `filter_by_position`.
- Row order of every export equals numpy quicksort order, not stable order; old and new test1 exports have identical order, ids and counts; only indel position strings differ.
- Introsort port: identical to numpy on all inputs listed in 6.3 once `SMALL_QUICKSORT = 15`.
- Variance: Welford+FMA = pandas 3000/3000 (ints), 1000/1000 (floats); plain Welford 2353/3000; committed golden matches neither (47/51 vs 46/51).
- Conversions at 10,000 × 30: 1.3 ms extract, 5.3 ms rebuild, 99 ms `to_csv`, 43 ms `read_csv`.
- Binary today: `PyEvoMotion.cpython-312-darwin.so` 1.52 MB, wheel 774 KB, ~100 crates.

---
