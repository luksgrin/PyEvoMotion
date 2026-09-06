"""Oracle tests for the Rust readers (rust/src/csv_read.rs, rust/src/dates.rs).

Each test builds the expected result with pandas itself on the same input and
compares with assert_frame_equal, so the tests are platform-agnostic. The
canonical row order is the *stable* sort by date (design doc, decision box).
"""
import ast
import os
import pandas as pd
import pytest

from PyEvoMotion import PyEvoMotion, PyEvoMotionParser

META = "tests/data/test1/test1.metadata.tsv"
FASTA = "tests/data/test1/test1.sequences.fasta"


def pandas_metadata(path, sep):
    df = pd.read_csv(path, sep=sep)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(by="date", kind="stable")


@pytest.mark.parametrize("path,sep", [
    (META, "\t"),
    ("tests/data/test4/S1.tsv", "\t"),
    pytest.param("tests/data/test3/ci/test3UK.tsv", "\t",
                 marks=pytest.mark.skipif(not os.path.exists("tests/data/test3/ci/test3UK.tsv"), reason="CI subset not downloaded")),
])
def test_parse_metadata_matches_pandas(path, sep):
    got = PyEvoMotionParser.parse_metadata(path)
    exp = pandas_metadata(path, sep)
    pd.testing.assert_frame_equal(got, exp)
    assert list(got.dtypes.astype(str)) == list(exp.dtypes.astype(str))
    assert got.index.tolist() == exp.index.tolist()      # labels kept, stable order


def test_parse_metadata_csv_extension(tmp_path):
    csv = tmp_path / "m.csv"
    pd.read_csv(META, sep="\t").head(5).to_csv(csv, index=False)
    got = PyEvoMotionParser.parse_metadata(str(csv))
    pd.testing.assert_frame_equal(got, pandas_metadata(str(csv), ","))


def test_parse_metadata_errors(tmp_path):
    with pytest.raises(ValueError, match="Unsupported metadata extension"):
        PyEvoMotionParser.parse_metadata(str(tmp_path / "m.txt"))
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tname\nA\tfoo\n")
    with pytest.raises(ValueError, match='"date" column'):
        PyEvoMotionParser.parse_metadata(str(bad))
    with pytest.raises(FileNotFoundError):
        PyEvoMotionParser.parse_metadata(str(tmp_path / "missing.tsv"))


def _oracle_read(text, tmp_path, sep="\t", name="t.tsv"):
    """Compare the Rust reader with pandas on the same file. When pandas
    itself raises, the Rust reader must raise the same exception type."""
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    try:
        df = pd.read_csv(p, sep=sep)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:  # pandas refuses: so must we
        with pytest.raises(type(e)):
            PyEvoMotionParser.parse_metadata(str(p))
        return None
    exp = df.sort_values(by="date", kind="stable")
    got = PyEvoMotionParser.parse_metadata(str(p))
    pd.testing.assert_frame_equal(got, exp)
    return got


def test_reader_inference_edge_cases(tmp_path):
    text = (
        "﻿id\tdate\tn\tf\tu\tb\tbn\tempty\t\tid\tmixed\n"
        "a\t2020-01-03\t1\t1.5\t9223372036854775808\tTrue\tTrue\t\tx\tdup\t 7\n"
        "b\t2020-01-01\t2\tNaN\t1\tfalse\t\t\ty\tdup2\tseven\n"
        "c\t2020-01-03\t-3\t1e3\t5\tTRUE\tFALSE\t\tz\tdup3\t8 \n"
    )
    got = _oracle_read(text, tmp_path)
    assert list(got.columns) == ["id", "date", "n", "f", "u", "b", "bn", "empty", "Unnamed: 8", "id.1", "mixed"]
    assert str(got["u"].dtype) == "uint64" and str(got["b"].dtype) == "bool"
    assert str(got["bn"].dtype) == "object" and str(got["empty"].dtype) == "float64"
    assert got["id"].tolist() == ["b", "a", "c"]          # stable: a before c


def test_reader_quotes_crlf_short_rows(tmp_path):
    text = 'id\tdate\tnote\r\n"q ""x"" y"\t2020-01-02\t"tab\there"\r\nshort\t2020-01-01\r\n\r\nz\t2020-01-02\tplain\r\n'
    got = _oracle_read(text, tmp_path)
    assert 'tab\there' in got["note"].tolist()
    assert got["note"].isna().sum() == 1                  # the short row was NaN-padded


def test_reader_long_row_is_parser_error(tmp_path):
    p = tmp_path / "long.tsv"
    p.write_text("id\tdate\na\t2020-01-01\textra\n")
    with pytest.raises(pd.errors.ParserError, match="Expected 2 fields in line 2, saw 3"):
        PyEvoMotionParser.parse_metadata(str(p))


def test_reader_int_with_missing_becomes_float(tmp_path):
    got = _oracle_read("id\tdate\tn\na\t2020-01-01\t1\nb\t2020-01-02\t\n", tmp_path)
    assert str(got["n"].dtype) == "float64"


def test_dates_iso_variants_and_fallback(tmp_path):
    # 1-digit month/day, compact, year-month, datetime, and a dd/mm/yyyy fallback
    _oracle_read("id\tdate\na\t2020-3-5\nb\t2020-03-04\n", tmp_path, name="a.tsv")
    _oracle_read("id\tdate\na\t20200305\nb\t20200304\n", tmp_path, name="b.tsv")
    _oracle_read("id\tdate\na\t2020-03\nb\t2020-02\n", tmp_path, name="c.tsv")
    _oracle_read("id\tdate\na\t2020-03-05 10:00:00\nb\t2020-03-05T09:00:00\n", tmp_path, name="d.tsv")
    _oracle_read("id\tdate\na\t2020-03-05 10:00:00\nb\t2020-03-05 09:00:00.5\n", tmp_path, name="d2.tsv")  # pandas decides
    _oracle_read("id\tdate\na\t2020-03-05 10:00:00.25\nb\t2020-03-05 09:00:00.5\n", tmp_path, name="d3.tsv")
    _oracle_read("id\tdate\na\t05/03/2020\nb\t04/03/2020\n", tmp_path, name="e.tsv")   # pandas path
    _oracle_read("id\tdate\na\t2020-03-05\nb\t\n", tmp_path, name="f.tsv")            # NaT last


def test_dates_mixed_formats_raise_like_pandas(tmp_path):
    p = tmp_path / "mixed.tsv"
    p.write_text("id\tdate\na\t2020-03-05\nb\t2020/03/06\n")
    with pytest.raises(ValueError):
        pd.to_datetime(pd.read_csv(p, sep="\t")["date"])          # pandas itself refuses
    with pytest.raises(ValueError):
        PyEvoMotionParser.parse_metadata(str(p))


def test_parse_mutation_data_matches_pandas(tmp_path):
    inst = PyEvoMotion(FASTA, META)
    tsv = tmp_path / "out.tsv"
    inst.data.to_csv(tsv, sep="\t", index=False)
    got = PyEvoMotionParser.parse_mutation_data(str(tsv))
    exp = pd.read_csv(tsv, sep="\t")
    exp["mutation instructions"] = exp["mutation instructions"].apply(ast.literal_eval)
    exp["date"] = pd.to_datetime(exp["date"])
    exp = exp.sort_values(by="date", kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, exp)
    # and it reproduces the instance's data exactly
    pd.testing.assert_frame_equal(got, inst.data.reset_index(drop=True))


def test_parse_mutation_data_errors(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tdate\nA\t2020-01-01\n")
    with pytest.raises(ValueError, match="mutation instructions"):
        PyEvoMotionParser.parse_mutation_data(str(bad))
    broken = tmp_path / "broken.tsv"
    broken.write_text("id\tdate\tmutation instructions\nA\t2020-01-01\t[not a literal\n")
    with pytest.raises(ValueError, match="Could not parse the mutation instructions of 'A'"):
        PyEvoMotionParser.parse_mutation_data(str(broken))
