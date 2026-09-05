"""Smoke tests for the Rust port of PyEvoMotion (rust/src/core.rs).

Exercises the methods in `_PyEvoMotionCore` end-to-end through the
public `PyEvoMotion` class so the wiring (Rust core mixin → Rust base)
is fully validated, including the `analysis()` flow that calls
`base::linear_regression` and `base::adjust_model` directly from Rust.
"""
import numpy as np
import pandas as pd
import pytest

from PyEvoMotion import PyEvoMotion, _PyEvoMotionCore, PyEvoMotionBase


@pytest.fixture(scope="module")
def instance():
    return PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv",
    )


def test_class_is_subclass_of_rust_core():
    assert issubclass(PyEvoMotion, _PyEvoMotionCore)
    assert issubclass(PyEvoMotion, PyEvoMotionBase)


def test_construction_populates_data(instance):
    df = instance.data
    assert len(df) > 0
    # count_mutation_types should have added these columns:
    for col in (
        "number of substitutions",
        "number of insertions",
        "number of deletions",
        "number of indels",
        "number of mutations",
    ):
        assert col in df.columns


def test_get_lengths_returns_series(instance):
    lengths = instance.get_lengths()
    assert isinstance(lengths, pd.Series)
    assert (lengths > 0).all()
    assert len(lengths) == len(instance.data)


def test_mutation_type_switch():
    assert PyEvoMotion._mutation_type_switch("all") == ["substitutions", "indels", "mutations"]
    assert PyEvoMotion._mutation_type_switch("total") == ["mutations"]
    assert PyEvoMotion._mutation_type_switch("substitutions") == ["substitutions"]
    assert PyEvoMotion._mutation_type_switch("indels") == ["indels"]
    with pytest.raises(ValueError):
        PyEvoMotion._mutation_type_switch("nope")


def test_apply_scaling_correction_linear(instance):
    model = {
        "model": object(),  # placeholder; gets replaced
        "parameters": {"m": 14.0, "b": 1.5},
        "confidence_intervals": {"m": (12.0, 16.0), "b": (1.0, 2.0)},
        "expression": "mx + b",
        "confidence_level": 0.95,
        "r2": 0.9,
    }
    instance._apply_scaling_correction_to_model(model)
    # dt_ratio for 7D vs 7D reference is 1.0, so values stay the same.
    assert model["parameters"]["m"] == pytest.approx(14.0 / instance.dt_ratio)
    assert model["parameters"]["b"] == pytest.approx(1.5)
    lo, hi = model["confidence_intervals"]["m"]
    assert lo == pytest.approx(12.0 / instance.dt_ratio)
    assert hi == pytest.approx(16.0 / instance.dt_ratio)


def test_apply_scaling_correction_power_law(instance):
    model = {
        "model": object(),
        "parameters": {"d": 4.0, "alpha": 1.7},
        "confidence_intervals": {"d": (3.0, 5.0), "alpha": (1.5, 1.9)},
        "expression": "d*x^alpha",
        "confidence_level": 0.95,
        "r2": 0.95,
    }
    instance._apply_scaling_correction_to_model(model)
    expected_scale = instance.dt_ratio ** 1.7
    assert model["parameters"]["d"] == pytest.approx(4.0 / expected_scale)
    assert model["parameters"]["alpha"] == pytest.approx(1.7)


def test_analysis_runs_end_to_end(instance):
    """The full analysis pipeline must run, returning stats and a
    populated regs dict whose keys mention the mutation types analysed.
    Math (linear_regression / adjust_model) is invoked from Rust core."""
    stats, regs = instance.analysis(length=0)

    assert isinstance(stats, pd.DataFrame)
    assert "date" in stats.columns
    assert "size" in stats.columns
    assert "dt_idx" in stats.columns

    # For each mutation kind in 'all' we expect a 'mean ... model' and a
    # 'scaled var ... model' (plus a *_full_results companion entry).
    has_mean = any(k.startswith("mean ") and k.endswith(" model") for k in regs)
    has_scaled_var = any(k.startswith("scaled var ") and k.endswith(" model") for k in regs)
    has_full = any(k.endswith("_full_results") for k in regs)
    assert has_mean
    assert has_scaled_var
    assert has_full

    # Each regression entry must be a dict with parameters + r2.
    for k, v in regs.items():
        if k.endswith("_full_results"):
            assert "selected_model" in v
            assert "linear_model" in v
            assert "power_law_model" in v
            assert "model_selection" in v
        else:
            assert "parameters" in v
            assert "r2" in v
            assert "model" in v
            # Model is callable (LinearCallable / PowerLawCallable) and
            # produces a numpy-coercible result.
            out = v["model"](np.arange(5))
            assert np.asarray(out).shape == (5,)


def test_analysis_substitutions_only_kind(instance):
    """`mutation_kind='substitutions'` should produce stats with just
    the substitution columns (mean + var) and matching reg entries."""
    stats, regs = instance.analysis(length=0, mutation_kind="substitutions")
    assert "mean number of substitutions" in stats.columns
    assert "var number of substitutions" in stats.columns
    # No 'mutations' or 'indels' columns
    assert "mean number of mutations" not in stats.columns
    assert "mean number of indels" not in stats.columns
    # regs only mentions substitutions
    assert all("substitutions" in k for k in regs.keys())


def test_subclass_preserves_behavior():
    """A user-defined subclass must inherit everything and still construct."""
    class MyAnalysis(PyEvoMotion):
        @classmethod
        def _mutation_type_switch(cls, kind):
            # Override a classmethod and confirm it's picked up.
            base = super()._mutation_type_switch(kind)
            base.append("__custom_marker__")
            return base

    assert MyAnalysis._mutation_type_switch("all")[-1] == "__custom_marker__"

    inst = MyAnalysis(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv",
    )
    assert isinstance(inst, PyEvoMotion)
    assert isinstance(inst, _PyEvoMotionCore)
    assert isinstance(inst, PyEvoMotionBase)
    assert hasattr(inst, "analysis")


# ─────────────────────── reloading a previous run (-load) ───────────────────────

COUNT_COLUMNS = (
    "number of substitutions",
    "number of indels",
    "number of insertions",
    "number of deletions",
    "number of mutations",
)


def test_load_mutation_instructions_reproduces_run(tmp_path, instance):
    """Constructing from the <out>.tsv of a previous run must give the same
    data (ids, instructions, counts) without re-aligning anything."""
    tsv = tmp_path / "out.tsv"
    instance.data.to_csv(tsv, sep="\t", index=False)

    loaded = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "does/not/matter.tsv",  # metadata is ignored when loading
        load_mutation_instructions=str(tsv),
    )
    assert loaded.reference.id == instance.reference.id
    assert loaded.data["id"].tolist() == instance.data["id"].tolist()
    assert loaded.data["mutation instructions"].tolist() == instance.data["mutation instructions"].tolist()
    for col in COUNT_COLUMNS:
        assert loaded.data[col].tolist() == instance.data[col].tolist()
    assert loaded.origin == instance.origin
    # The reloaded instance is fully functional.
    stats, regs = loaded.analysis(length=0)
    assert len(stats) > 0 and len(regs) > 0


def test_load_reuses_stored_counts_unless_recount(tmp_path, instance):
    df = instance.data.copy()
    df["number of mutations"] = -1  # deliberately wrong
    tsv = tmp_path / "out.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    args = ("tests/data/test1/test1.sequences.fasta", "ignored.tsv")

    kept = PyEvoMotion(*args, load_mutation_instructions=str(tsv))
    assert (kept.data["number of mutations"] == -1).all()

    recounted = PyEvoMotion(*args, load_mutation_instructions=str(tsv), recount_mutation_types=True)
    assert recounted.data["number of mutations"].tolist() == instance.data["number of mutations"].tolist()


def test_load_recounts_when_count_columns_are_absent(tmp_path, instance):
    df = instance.data.drop(columns=list(COUNT_COLUMNS))
    tsv = tmp_path / "out.tsv"
    df.to_csv(tsv, sep="\t", index=False)
    loaded = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta", "ignored.tsv",
        load_mutation_instructions=str(tsv),
    )
    for col in COUNT_COLUMNS:
        assert loaded.data[col].tolist() == instance.data[col].tolist()


def test_load_applies_filters(tmp_path, instance):
    tsv = tmp_path / "out.tsv"
    instance.data.to_csv(tsv, sep="\t", index=False)
    from datetime import datetime
    start = datetime(2020, 3, 1)
    loaded = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta", "ignored.tsv",
        load_mutation_instructions=str(tsv),
        date_range=(start, None),
    )
    assert (loaded.data["date"] >= start).all()
    assert 0 < len(loaded.data) < len(instance.data)


def test_load_rejects_file_without_instructions_column(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tdate\nA\t2020-01-01\n")
    with pytest.raises(ValueError, match="mutation instructions"):
        PyEvoMotion(
            "tests/data/test1/test1.sequences.fasta", "ignored.tsv",
            load_mutation_instructions=str(bad),
        )


# ─────────────────────── genome-window semantics ───────────────────────

def test_filter_by_position_window_rules():
    """Substitutions are kept for start <= pos < end, insertions/deletions
    for start < pos <= end. With the default (whole genome) window this
    excludes a deletion starting at the first reference base and keeps an
    insertion after the last one — the counting behaviour of every release
    before 0.2.0, preserved when positions became 1-based."""
    from PyEvoMotion import SequenceRecord

    class Bare(PyEvoMotion):
        def __init__(self):  # skip file parsing; we set the state by hand
            pass

    def run(instr, start=0, end=0):
        p = Bare()
        p.reference = SequenceRecord("ref", "A" * 10)
        p.data = pd.DataFrame({"id": ["x"], "mutation instructions": [instr]})
        p.filter_by_position(start, end)
        return p.data["mutation instructions"].iloc[0] if len(p.data) else None

    assert run(["d_1_AC", "s_5_T", "i_11_AAA", "s_10_G"]) == ["s_5_T", "i_11_AAA", "s_10_G"]
    assert run(["s_1_T"]) == ["s_1_T"]          # substitution at the first base counts
    assert run(["i_1_TT"]) is None              # insertion before the first base does not;
                                                # as the only mutation, the row is dropped
    assert run(["d_10_A"]) == ["d_10_A"]        # deletion of the last base counts
    # explicit window 3..6: substitutions in [3,6), indels in (3,6]
    assert run(["s_3_T", "s_6_T", "d_3_A", "d_6_A", "i_4_G"], 3, 6) == ["s_3_T", "d_6_A", "i_4_G"]
    # a sequence whose only mutations fall outside the window is dropped
    assert run(["s_8_T"], 1, 5) is None
