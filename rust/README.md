# PyEvoMotion Rust crate

This crate *is* the `PyEvoMotion` Python package: a single compiled
extension module built with PyO3 / maturin. There are no Python sources;
`import PyEvoMotion` loads the shared library, which defines

| Python name           | Source        | Role                                              |
|-----------------------|---------------|---------------------------------------------------|
| `PyEvoMotionBase`     | `src/base.rs`   | Math/statistics utilities (regressions, F-test, AIC, model selection) |
| `PyEvoMotionParser`   | `src/parser.rs` | Input parsing, in-process MAFFT alignment, mutation calling, filters |
| `_PyEvoMotionCore`    | `src/core.rs`   | Analysis, statistics over time windows, plotting |
| `PyEvoMotion`         | `src/core.rs`   | The public class users instantiate; constructor logic |
| `SequenceRecord`, `FastaReader` | `src/fasta.rs` | FASTA records and streaming reader (replaces Biopython) |
| `Table`               | `src/table.rs`  | Internal column store behind `PyEvoMotion.data`; `from_pandas`/`to_pandas`/`to_tsv` |
| (internal)            | `src/csv_read.rs`, `src/dates.rs`, `src/stats.rs`, `src/csv_write.rs` | TSV/CSV reading with pandas' inference rules, date parsing, window statistics, TSV writing |
| `_main`               | `src/cli.rs`    | Console entry point (`PyEvoMotion = "PyEvoMotion:_main"`) |

Inheritance is a single chain, `PyEvoMotion -> _PyEvoMotionCore ->
PyEvoMotionParser -> PyEvoMotionBase`, and every class is `subclass`-able
from Python. Methods that call other methods dispatch through Python
attribute lookup, so subclass overrides take effect.

Sequence alignment uses the pure-Rust [`mafft`](https://crates.io/crates/mafft)
crate (FFT-NS-2, C-compat scoring), so no external binary is needed.

## The data pipeline and pandas

Everything between reading the inputs and returning results runs on the
Rust `Table` (see `DESIGN_internal_table.md`). pandas appears only at the
boundary:

- `instance.data` is a property. Reading it materialises a DataFrame (once,
  cached until the next pipeline stage); assigning a DataFrame or a `Table`
  replaces the dataset. In-place edits of the DataFrame you were handed are
  picked up by the next stage, so the usual subclass idiom
  `self.data = self.data[mask]` works unchanged.
- `Table.from_pandas(df)` / `table.to_pandas()` round-trip losslessly for
  everything a TSV can hold (strings with missing values, ints, floats,
  bools, datetimes, the list-valued mutation column); other dtypes pass
  through untouched. `table.to_tsv(path)` writes exactly what
  `to_csv(sep="\t", index=False)` would.
- Row order is a stable sort by collection date and the per-window variance
  uses fixed-order arithmetic, so the data and statistics tables are
  byte-identical on every platform; `tests/test_golden_outputs.py` and the
  dataset tests compare them with the committed goldens under
  `tests/data/golden/` (regenerate with `PYEVOMOTION_UPDATE_GOLDENS=1`
  after an intentional change). Fitted models go through the platform's
  libm (exp/log, t and F distributions) and are compared numerically: linear
  fits agree to ~1e-12, the iterative power-law fits to ~1e-6 relative.

## Build & install (dev)

From the repository root, inside a Python 3.12 virtualenv:

```bash
uv pip install -e .          # maturin builds the extension in release mode
pytest rust/tests            # fast smoke tests against the installed build
```

`pyproject.toml` (section `[tool.maturin]`) points maturin at this crate and
sets `module-name = "PyEvoMotion"`.

## Build a wheel

```bash
maturin build --release --strip -m rust/Cargo.toml --interpreter 3.12
ls target/wheels/
```

Wheels are `cp312-cp312` (not abi3): `lib.rs` installs a `tp_init` slot on
the `PyEvoMotion` type via the full C API so that PyO3's `#[pymethods]
__init__` runs on direct construction and in Python subclasses.

## Notes on numerical parity with the old Python implementation

- **Power-law fit**: scipy's `curve_fit` used Trust Region Reflective with
  bounds; here it is unbounded Levenberg-Marquardt with `a = exp(t_a)` to
  keep the coefficient positive. Results agree to within tolerance on the
  bundled datasets.
- **Covariance**: estimated from `(JᵀJ)⁻¹ σ²` at the optimum, with the delta
  method for `var(a)`.
- **`_remove_nan`** returns 1-D arrays rather than `(n, 1)` arrays.
