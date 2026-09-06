"""Tests for PyEvoMotion.Table (rust/src/table.rs) and the `data` property.

The Table is the internal column store that replaces pandas inside the
pipeline; pandas stays at the boundary. These tests pin (a) the lossless
round trip for every dtype PyEvoMotion can produce and (b) the two-state
ownership semantics of `instance.data` described in
rust/DESIGN_internal_table.md §4.
"""
import numpy as np
import pandas as pd
import pytest

from PyEvoMotion import PyEvoMotion, PyEvoMotionParser, Table, SequenceRecord

META = "tests/data/test1/test1.metadata.tsv"
FASTA = "tests/data/test1/test1.sequences.fasta"


def _roundtrip(df):
    t = Table.from_pandas(df)
    back = t.to_pandas()
    pd.testing.assert_frame_equal(back, df, check_index_type=True, check_column_type=True)
    assert type(back.index) is type(df.index)
    assert list(back.dtypes.astype(str)) == list(df.dtypes.astype(str))
    return t


# ─────────────────────── round trips ───────────────────────

def test_roundtrip_metadata_as_read_csv():
    df = pd.read_csv(META, sep="\t")            # 27 object + 1 int64 column, NaNs in `location`
    t = _roundtrip(df)
    assert len(t) == len(df) and t.columns == list(df.columns)
    assert t.dtypes["length"] == "int64" and t.dtypes["location"] == "object"
    assert not t.empty


def test_roundtrip_after_to_datetime_and_unstable_sort():
    df = pd.read_csv(META, sep="\t")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")               # int64 Index with permuted labels
    t = _roundtrip(df)
    assert t.dtypes["date"] == "datetime64[ns]"
    assert type(t.to_pandas().index) is pd.Index


def test_roundtrip_full_instance_data():
    inst = PyEvoMotion(FASTA, META)
    df = inst.data                                # lists, int counts, datetimes, NaNs
    t = _roundtrip(df)
    assert t.dtypes["mutation instructions"] == "object"
    assert t.column("mutation instructions")[1] == df["mutation instructions"].iloc[1]
    assert t.dtypes["number of mutations"] == "int64"


def test_roundtrip_edge_dtypes():
    df = pd.DataFrame({
        "i": np.array([1, 2, 3], dtype="int64"),
        "u": np.array([1, 2, 2**63], dtype="uint64"),
        "f": [1.5, np.nan, -0.0],
        "b": [True, False, True],
        "s": ["a", np.nan, "c"],
        "l": [["s_1_A"], [], ["d_2_C", "i_3_GG"]],
        "d": pd.to_datetime(["2020-01-01 00:00:00", None, "2021-06-30 12:00:00"]),
        "o": [1, "x", None],                     # mixed → opaque objects
        "cat": pd.Categorical(["x", "y", "x"]),  # foreign dtype
    })
    t = _roundtrip(df)
    assert t.dtypes["o"] == "object" and t.dtypes["cat"] == "category"
    assert t.dtypes["d"] == "datetime64[ns]" and t.dtypes["u"] == "uint64"


def test_roundtrip_datetime_second_unit_and_empty_frame():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"]).astype("datetime64[s]")})
    assert Table.from_pandas(df).dtypes["d"] == "datetime64[s]"
    _roundtrip(df)
    empty = pd.DataFrame({"id": pd.Series([], dtype="object"), "date": pd.Series([], dtype="datetime64[ns]")})
    t = _roundtrip(empty)
    assert t.empty and len(t) == 0


def test_from_pandas_rejects_non_dataframe():
    with pytest.raises(TypeError):
        Table.from_pandas([1, 2, 3])


def test_table_api_surface():
    t = Table.from_pandas(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert "a" in t and "z" not in t
    assert t.column("b") == ["x", "y"]
    with pytest.raises(KeyError):
        t.column("z")
    assert repr(t) == "Table(2 rows x 2 columns: a: int64, b: object)"


# ─────────────────────── `data` property semantics ───────────────────────

class Bare(PyEvoMotion):
    def __init__(self):  # no parsing; state is set by the test
        pass


def test_unset_data_raises_attribute_error_like_before():
    with pytest.raises(AttributeError, match="data"):
        Bare().data


def test_setter_accepts_dataframe_and_keeps_identity():
    p = Bare()
    df = pd.DataFrame({"id": ["x"], "mutation instructions": [["s_1_A"]]})
    p.data = df
    assert p.data is df                          # pandas-visible state: same object
    p.data["extra"] = 1                          # in-place edit is visible
    assert "extra" in p.data.columns


def test_setter_accepts_table_and_materialises_once():
    p = Bare()
    df = pd.DataFrame({"id": ["x", "y"], "n": [1, 2]})
    p.data = Table.from_pandas(df)
    first = p.data                               # materialised on first access
    assert first is p.data                       # cached until the next stage
    pd.testing.assert_frame_equal(first, df)


def test_setter_rejects_other_types():
    p = Bare()
    with pytest.raises((TypeError, ValueError)):
        p.data = {"id": ["x"]}


def test_override_idiom_reassigning_data_works(tmp_path):
    """The classic subclass pattern: read self.data, filter with pandas, assign back."""
    class OnlyEngland(PyEvoMotion):
        def filter_columns(self, filters):
            super().filter_columns(filters)
            self.data = self.data[self.data["division"] == "England"]

    inst = OnlyEngland(FASTA, META)
    assert set(inst.data["division"]) == {"England"}
    assert len(inst.data) > 10
    assert "number of mutations" in inst.data.columns


def test_in_place_mutation_seen_by_following_stage():
    p = Bare()
    p.reference = SequenceRecord("ref", "A" * 10)
    p.data = pd.DataFrame({"id": ["a", "b"], "mutation instructions": [["s_2_T"], ["s_9_G"]]})
    p.data["marker"] = [1, 2]                    # edit the handed-out frame in place
    p.filter_by_position(1, 5)                   # keeps only the first row
    assert p.data["marker"].tolist() == [1]
    assert p.data["mutation instructions"].tolist() == [["s_2_T"]]


# ─────────────────────── pipeline stages on the table (phase 3) ───────────────────────

def _bare_with(df, ref_len=10):
    p = Bare()
    p.reference = SequenceRecord("ref", "A" * ref_len)
    p.data = df
    return p


def test_filter_columns_matches_pandas_str_contains():
    df = pd.DataFrame({"id": list("abcdef"), "country": ["Spain", "spain", "Portugal", "France", "Italy", "Spain "],
                       "mutation instructions": [[]] * 6})
    for filters in ({"country": "Spa*"}, {"country": ["Spain", "Fra*"]}, {"country": "^Spain$"}, {"nope": "x"}):
        p = _bare_with(df.copy())
        p.filter_columns(filters)
        exp = df.copy()
        for k, v in filters.items():
            if k not in exp.columns:
                continue
            vals = [v] if isinstance(v, str) else v
            pattern = "|".join(x.replace("*", ".*") for x in vals)
            exp = exp[exp[k].str.contains(pattern, regex=True)]
        pd.testing.assert_frame_equal(p.data, exp)


def test_filter_columns_regex_fallback_for_lookahead():
    df = pd.DataFrame({"id": list("abc"), "x": ["foo1", "foo2", "bar1"], "mutation instructions": [[]] * 3})
    p = _bare_with(df.copy())
    p.filter_columns({"x": "foo(?=2)"})           # lookahead: not supported by the regex crate → Python re
    assert p.data["id"].tolist() == ["b"]


def test_filter_columns_errors_like_pandas():
    p = _bare_with(pd.DataFrame({"id": ["a", "b"], "c": ["x", np.nan], "mutation instructions": [[], []]}))
    with pytest.raises(ValueError, match="NA / NaN"):
        p.filter_columns({"c": "x"})
    p = _bare_with(pd.DataFrame({"id": ["a", "b"], "n": [1, 2], "mutation instructions": [[], []]}))
    with pytest.raises(AttributeError, match=".str accessor"):
        p.filter_columns({"n": "1"})


def test_filter_by_daterange_clamps_to_data_and_errors():
    from datetime import datetime
    df = pd.DataFrame({"id": list("abcd"), "date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-10", None]),
                       "mutation instructions": [[]] * 4})
    p = _bare_with(df.copy())
    p.filter_by_daterange(datetime(2019, 1, 1), datetime(2020, 1, 6))   # start clamped to data min
    assert p.data["id"].tolist() == ["a", "b"]
    p = _bare_with(df.copy())
    p.filter_by_daterange(None, None)
    assert p.data["id"].tolist() == ["a", "b", "c"]                      # NaT never inside the range
    p = _bare_with(df.copy())
    with pytest.raises(ValueError, match="Start date must be smaller"):
        p.filter_by_daterange(datetime(2020, 1, 8), datetime(2020, 1, 2))
    p = _bare_with(df.iloc[:0].copy())
    p.filter_by_daterange(datetime(2020, 1, 1), None)                    # empty data: no error, stays empty
    assert len(p.data) == 0


def test_load_with_filters_and_recount_gives_integer_counts(tmp_path):
    """-load + -dr + -recount: counts must be recomputed row-wise (int64, no
    NaN) even though the loaded table was filtered and its index has gaps."""
    from datetime import datetime
    inst = PyEvoMotion(FASTA, META)
    tsv = tmp_path / "out.tsv"
    inst.data.to_csv(tsv, sep="\t", index=False)
    loaded = PyEvoMotion(FASTA, "ignored.tsv", load_mutation_instructions=str(tsv),
                         date_range=(datetime(2020, 3, 1), None), recount_mutation_types=True)
    assert str(loaded.data["number of mutations"].dtype) == "int64"
    assert loaded.data["number of mutations"].tolist() == [len(m) for m in loaded.data["mutation instructions"]]
    assert 0 < len(loaded.data) < len(inst.data)


def test_get_lengths_and_noop_filters_reset_index():
    df = pd.DataFrame({"id": ["a", "b", "c"], "N count": [0, 1, 0],
                       "mutation instructions": [["s_2_T"], ["d_3_AA"], ["i_4_GGG"]]})
    p = _bare_with(df.copy())
    lengths = p.get_lengths()
    assert isinstance(lengths, pd.Series) and lengths.tolist() == [10, 8, 13]
    p.data = p.data.iloc[[0, 2]]                  # gapped index, assigned as a DataFrame
    p.filter_by_position(1, 0)                    # Rust stage: drops nothing, keeps labels [0, 2]
    p.length_filter(0)                            # deliberate no-op except reset_index (Rust-owned path)
    assert p.data.index.tolist() == [0, 1]
    p.n_filter(0.5, "lt")
    assert p.data.index.tolist() == [0, 1]
    with pytest.raises(ValueError, match="not recognized"):
        p.n_filter(0.5, "nope")


# ─────────────────────── TSV writer (phase 5) ───────────────────────

def _same_bytes(df, tmp_path, name):
    a = tmp_path / f"{name}_pandas.tsv"
    b = tmp_path / f"{name}_rust.tsv"
    df.to_csv(a, sep="\t", index=False)
    Table.from_pandas(df).to_tsv(str(b))
    assert a.read_bytes() == b.read_bytes(), f"{name}: writer output differs from pandas"


def test_to_tsv_matches_pandas_on_data_and_stats(tmp_path):
    inst = PyEvoMotion(FASTA, META)
    _same_bytes(inst.data, tmp_path, "data")
    stats, _ = inst.analysis(length=0)
    _same_bytes(stats, tmp_path, "stats")
    _same_bytes(pd.read_csv(META, sep="\t"), tmp_path, "metadata")


def test_to_tsv_matches_pandas_on_edge_values(tmp_path):
    df = pd.DataFrame({
        "f": [1e-5, 1e16, 0.30000000000000004, 1.2345678901234568e17, np.nan, -0.0, 1e15, 6.249999999999999, 100.0],
        "i": np.arange(9, dtype="int64"),
        "s": ["plain", "with\ttab", 'say "hi"', "", "multi\nline", "it's", "trail ", "NA", "None"],
        "l": [["s_1_A", "d_2_C"], [], ["it's"], ["a\\b"], ['q"q'], ["x"] * 3, ["é"], ["tab\t"], ["new\nline"]],
        "d": pd.to_datetime(["2020-01-01"] * 8 + [None]),
        "b": [True, False] * 4 + [True],
    })
    _same_bytes(df, tmp_path, "edge")
    dt = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 13:05:09"])})
    _same_bytes(dt, tmp_path, "times")
    empty_str = pd.DataFrame({"s": ["", np.nan, "x"]})   # "" and NaN both empty fields
    _same_bytes(empty_str, tmp_path, "empties")


def test_to_tsv_falls_back_to_pandas_for_opaque_objects(tmp_path):
    df = pd.DataFrame({"o": [1, "x", None], "c": pd.Categorical(["a", "b", "a"])})
    _same_bytes(df, tmp_path, "opaque")
