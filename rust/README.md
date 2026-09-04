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
| `_main`               | `src/cli.rs`    | Console entry point (`PyEvoMotion = "PyEvoMotion:_main"`) |

Inheritance is a single chain, `PyEvoMotion -> _PyEvoMotionCore ->
PyEvoMotionParser -> PyEvoMotionBase`, and every class is `subclass`-able
from Python. Methods that call other methods dispatch through Python
attribute lookup, so subclass overrides take effect.

Sequence alignment uses the pure-Rust [`mafft`](https://crates.io/crates/mafft)
crate (FFT-NS-2, C-compat scoring), so no external binary is needed.

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
