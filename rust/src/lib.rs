use pyo3::ffi;
use pyo3::prelude::*;

mod base;
mod cli;
mod core;
mod parser;

/// PyEvoMotion — assess the evolution dynamics of related DNA sequences.
///
/// The entire public API is implemented in this compiled extension:
///
/// * ``PyEvoMotion`` — the main analysis class.
/// * ``PyEvoMotionBase`` — math/utility base.
/// * ``PyEvoMotionParser`` — input parsing (FASTA/metadata, alignment).
/// * ``_main`` — the command-line entry point.
#[pymodule]
fn PyEvoMotion(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<base::PyEvoMotionBase>()?;
    m.add_class::<base::LinearCallable>()?;
    m.add_class::<base::PowerLawCallable>()?;
    m.add_class::<core::PyEvoMotionCore>()?;
    m.add_class::<core::PyEvoMotion>()?;
    m.add_class::<parser::PyEvoMotionParser>()?;
    m.add_function(wrap_pyfunction!(cli::_main, m)?)?;

    // Declare __all__ so maturin's generated `from .PyEvoMotion import *`
    // re-export (and any star-import) includes the underscore-prefixed names
    // the CLI entry point and tests rely on (_main, _PyEvoMotionCore).
    m.add(
        "__all__",
        vec![
            "PyEvoMotion",
            "PyEvoMotionBase",
            "PyEvoMotionParser",
            "_PyEvoMotionCore",
            "_main",
        ],
    )?;

    // PyO3 does not wire a #[pymethods] __init__ to tp_init, so install the
    // trampoline manually on the PyEvoMotion type. This makes direct
    // construction (PyEvoMotion(...)) run the Rust __init__; Python subclasses
    // inherit this slot (or get CPython's normal wiring if they define their
    // own __init__), so subclassing keeps working with correct types.
    let ty = m.getattr("PyEvoMotion")?;
    unsafe {
        let ty_ptr = ty.as_ptr() as *mut ffi::PyTypeObject;
        (*ty_ptr).tp_init = Some(core::pyevomotion_tp_init);
        ffi::PyType_Modified(ty_ptr);
    }

    Ok(())
}
