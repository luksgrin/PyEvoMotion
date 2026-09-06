"""Oracle tests for compute_stats on the internal table (rust/src/stats.rs).

The expected values come from pandas' own Grouper on the same data. Dates,
sizes and means must be identical; the sample variance is computed with
canonical (platform-independent) arithmetic and is compared within 1e-12.
"""
import numpy as np
import pandas as pd
import pytest

from PyEvoMotion import PyEvoMotion

META = "tests/data/test1/test1.metadata.tsv"
FASTA = "tests/data/test1/test1.sequences.fasta"
LEVELS = ["number of substitutions", "number of indels", "number of mutations"]


@pytest.fixture(scope="module")
def instance():
    return PyEvoMotion(FASTA, META)


def pandas_stats(df, dt, origin, levels=LEVELS):
    data = df.copy()
    if data.iloc[0]["date"] == origin and (data["date"] == origin).sum() == 1:
        data = pd.concat([data, pd.DataFrame([data.iloc[0]])], ignore_index=True)
        data = data.sort_values(by="date", kind="stable").reset_index(drop=True)
    g = data.groupby(pd.Grouper(key="date", freq=dt, origin=origin))
    filtered = g.filter(lambda x: len(x) >= 2)
    g = filtered.groupby(pd.Grouper(key="date", freq=dt, origin=origin))
    frames = [g[levels].mean().rename(columns=lambda c: "mean " + c),
              g[levels].var().rename(columns=lambda c: "var " + c),
              pd.DataFrame(g.size()).rename(columns=lambda c: "size")]
    return pd.concat(frames, axis=1).reset_index(level=["date"])


def _compare(got, exp):
    assert list(got.columns) == list(exp.columns)
    assert got["date"].tolist() == exp["date"].tolist()
    assert got["size"].tolist() == exp["size"].tolist()
    assert str(got["size"].dtype) == "int64" and str(got["date"].dtype) == "datetime64[ns]"
    mean_cols = [c for c in got.columns if c.startswith("mean")]
    var_cols = [c for c in got.columns if c.startswith("var")]
    np.testing.assert_array_equal(got[mean_cols].to_numpy(), exp[mean_cols].to_numpy())
    np.testing.assert_allclose(got[var_cols].to_numpy(), exp[var_cols].to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    assert type(got.index) is pd.RangeIndex


@pytest.mark.parametrize("dt", ["7D", "3D", "14D", "36h"])
def test_compute_stats_matches_pandas_grouper(instance, dt):
    origin = instance.origin
    got = instance.compute_stats(dt, origin)
    _compare(got, pandas_stats(instance.data, dt, origin))


def test_compute_stats_origin_later_than_data_and_subset_kind(instance):
    origin = instance.data["date"].min() + pd.Timedelta("4D")
    got = instance.compute_stats("7D", origin, "substitutions")
    _compare(got, pandas_stats(instance.data, "7D", origin, ["number of substitutions"]))


def test_compute_stats_keeps_empty_bins(instance):
    origin = instance.origin
    got = instance.compute_stats("3D", origin)
    assert (got["size"] == 0).any()
    assert got.loc[got["size"] == 0, "mean number of mutations"].isna().all()


def test_compute_stats_anchored_frequency_uses_pandas_path(instance):
    got = instance.compute_stats("W", instance.origin)   # weekly, anchored → pandas implementation
    assert isinstance(got, pd.DataFrame) and "size" in got.columns and len(got) > 0


def test_compute_stats_no_group_with_two_rows_raises():
    from PyEvoMotion import SequenceRecord
    class Bare(PyEvoMotion):
        def __init__(self):
            pass
    p = Bare()
    p.reference = SequenceRecord("r", "A" * 10)
    p.data = pd.DataFrame({"id": ["a", "b"], "date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
                           "mutation instructions": [["s_1_A"], []],
                           "number of substitutions": [1, 0], "number of indels": [0, 0], "number of mutations": [1, 0]})
    with pytest.raises(ValueError, match="at least 2 observations"):
        p.compute_stats("7D", pd.Timestamp("2019-12-01"))     # every window holds a single row
    # With the origin on the lone first row, that row is duplicated on purpose
    # (as before) so its window survives:
    got = p.compute_stats("7D", pd.Timestamp("2020-01-01"))
    assert got["size"].tolist()[0] == 2


def test_analysis_end_to_end_is_deterministic(instance):
    s1, r1 = instance.analysis(length=0)
    s2, r2 = instance.analysis(length=0)
    pd.testing.assert_frame_equal(s1, s2)
    assert r1["mean number of mutations model"]["parameters"] == r2["mean number of mutations model"]["parameters"]
