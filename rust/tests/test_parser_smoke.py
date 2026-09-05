"""Smoke tests for PyEvoMotion.PyEvoMotionParser (rust/src/parser.rs).

Covers the algorithmic helpers (_column_decision, _get_consecutives,
create_modifs) directly, plus the I/O methods (parse_metadata,
parse_sequence_by_id, _run_mafft, generate_alignment) and the orchestrating
__init__ on real test1 fixtures.
"""

import numpy as np
import pandas as pd
import pytest

from PyEvoMotion import PyEvoMotionParser, SequenceRecord, FastaReader


# ─────────────────────── pure algorithm ───────────────────────

def test_get_consecutives_basic():
    assert PyEvoMotionParser._get_consecutives([]) == []
    assert PyEvoMotionParser._get_consecutives([5]) == [[5]]
    assert PyEvoMotionParser._get_consecutives([1, 2, 3]) == [[1, 2, 3]]
    assert PyEvoMotionParser._get_consecutives([1, 2, 3, 5, 6, 8]) == [[1, 2, 3], [5, 6], [8]]
    assert PyEvoMotionParser._get_consecutives([10, 11, 12, 14]) == [[10, 11, 12], [14]]


def test_column_decision_match():
    assert PyEvoMotionParser._column_decision(np.array(["A", "A"])) == 0
    assert PyEvoMotionParser._column_decision(np.array(["G", "G"])) == 0


def test_column_decision_n_treated_as_match():
    assert PyEvoMotionParser._column_decision(np.array(["A", "N"])) == 0
    assert PyEvoMotionParser._column_decision(np.array(["N", "T"])) == 0


def test_column_decision_substitution():
    assert PyEvoMotionParser._column_decision(np.array(["A", "G"])) == 1


def test_column_decision_insertion():
    # Gap in reference → insertion in target
    assert PyEvoMotionParser._column_decision(np.array(["-", "A"])) == 2


def test_column_decision_deletion():
    # Gap in target → deletion from reference
    assert PyEvoMotionParser._column_decision(np.array(["A", "-"])) == 3


def _alignment_from_strings(ref: str, target: str):
    """Build the [reference, target] pair generate_alignment returns."""
    return [SequenceRecord("ref", ref), SequenceRecord("target", target)]


def test_create_modifs_substitution_only():
    aln = _alignment_from_strings("AAAAA", "AAGAA")
    out = PyEvoMotionParser.create_modifs(aln)
    assert out == ["s_3_G"]


def test_create_modifs_deletion_only():
    aln = _alignment_from_strings("ACCCT", "A---T")
    out = PyEvoMotionParser.create_modifs(aln)
    # Gaps in the target at columns 1,2,3 (0-based) → the deleted bases
    # start at 1-based reference position 2. Value = reference bases.
    assert out == ["d_2_CCC"]


def test_create_modifs_insertion_only():
    aln = _alignment_from_strings("A--T", "AGGT")
    out = PyEvoMotionParser.create_modifs(aln)
    # Insertion at columns 1,2 (gap in reference, bases GG in target): the
    # first inserted base sits at 1-based position 2. After re-indexing
    # later mutations to pre-insertion coordinates the offsets shift; here
    # there are no later mutations, so just the insertion itself appears.
    assert out == ["i_2_GG"]


def test_create_modifs_combined():
    # ref: A C C C T
    # tgt: A C - G T  (column 2 is deletion, column 3 is substitution C->G)
    aln = _alignment_from_strings("ACCCT", "AC-GT")
    out = PyEvoMotionParser.create_modifs(aln)
    assert "d_3_C" in out
    assert "s_4_G" in out


def test_create_modifs_positions_are_1_based_for_every_kind():
    # ref: A C G T A C G T
    # tgt: A T G - A C G T   (s at col 1, d at col 3)  → same convention
    aln = _alignment_from_strings("ACGTACGT", "ATG-ACGT")
    assert PyEvoMotionParser.create_modifs(aln) == ["s_2_T", "d_4_T"]
    # Insertion before the very first base is position 1.
    aln = _alignment_from_strings("--ACG", "TTACG")
    assert PyEvoMotionParser.create_modifs(aln) == ["i_1_TT"]


def test_create_modifs_lowercase_input_is_uppercased():
    aln = _alignment_from_strings("aaaaa", "aagaa")
    out = PyEvoMotionParser.create_modifs(aln)
    assert out == ["s_3_G"]


# ─────────────────────── I/O methods ───────────────────────

def test_parse_metadata_tsv():
    df = PyEvoMotionParser.parse_metadata("tests/data/test1/test1.metadata.tsv")
    assert "date" in df.columns
    # parse_metadata sorts by date
    assert df["date"].is_monotonic_increasing


def test_parse_metadata_missing_date_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tname\nA\tfoo\nB\tbar\n")
    with pytest.raises(ValueError, match="date"):
        PyEvoMotionParser.parse_metadata(str(bad))


def test_parse_sequence_by_id_found():
    rec = PyEvoMotionParser.parse_sequence_by_id(
        "tests/data/test1/test1.sequences.fasta",
        "hCoV-19/Wuhan/IVDC-HB-01/2019",
    )
    assert rec is not None
    assert rec.id == "hCoV-19/Wuhan/IVDC-HB-01/2019"


def test_parse_sequence_by_id_missing_returns_none():
    rec = PyEvoMotionParser.parse_sequence_by_id(
        "tests/data/test1/test1.sequences.fasta",
        "DOES_NOT_EXIST",
    )
    assert rec is None


# ─────────────────────── full __init__ via PyEvoMotion ───────────────────────

def test_parser_init_through_pyevomotion():
    """Running PyEvoMotion(...) drives PyEvoMotionParser.__init__ through
    multi-inheritance; should leave self.data and self.reference populated."""
    from PyEvoMotion import PyEvoMotion
    inst = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv",
    )
    assert isinstance(inst.data, pd.DataFrame)
    assert "id" in inst.data.columns
    assert "mutation instructions" in inst.data.columns
    assert "N count" in inst.data.columns
    assert inst.reference is not None
    assert hasattr(inst.reference, "seq")


# ─────────────────────── explicit reference (refseq) ───────────────────────

def test_refseq_uses_first_record_of_given_fasta(tmp_path):
    """With refseq, the reference is the first record of that file rather
    than the earliest-dated metadata entry."""
    from PyEvoMotion import PyEvoMotion
    records = list(PyEvoMotionParser.read_fasta("tests/data/test1/test1.sequences.fasta"))
    # Pick a record that is not the default reference and write it alone.
    default = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv",
    ).reference.id
    chosen = next(r for r in records if r.id != default)
    ref_fasta = tmp_path / "ref.fasta"
    ref_fasta.write_text(chosen.format())

    inst = PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv",
        refseq=str(ref_fasta),
    )
    assert inst.reference.id == chosen.id
    # The chosen sequence aligned against itself carries no mutations.
    own = inst.data[inst.data["id"] == chosen.id]["mutation instructions"].iloc[0]
    assert list(own) == []


def test_refseq_empty_fasta_raises(tmp_path):
    from PyEvoMotion import PyEvoMotion
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    with pytest.raises(ValueError, match="no sequences"):
        PyEvoMotion(
            "tests/data/test1/test1.sequences.fasta",
            "tests/data/test1/test1.metadata.tsv",
            refseq=str(empty),
        )


# ─────────────────────── ids missing from the FASTA ───────────────────────

def test_metadata_ids_missing_from_fasta_are_dropped_with_warning(tmp_path, capfd):
    """A metadata row whose id has no FASTA record must not crash the run;
    it is dropped and reported on stderr."""
    from PyEvoMotion import PyEvoMotion
    meta = pd.read_csv("tests/data/test1/test1.metadata.tsv", sep="\t")
    ghost = meta.iloc[[0]].copy()
    ghost["id"] = "NOT_IN_FASTA/2020"
    ghost["date"] = "2020-06-01"
    meta_path = tmp_path / "meta.tsv"
    pd.concat([meta, ghost]).to_csv(meta_path, sep="\t", index=False)

    inst = PyEvoMotion("tests/data/test1/test1.sequences.fasta", str(meta_path))
    err = capfd.readouterr().err
    assert "1 sequence(s) have no mutation instructions" in err
    assert "NOT_IN_FASTA/2020" in err
    assert "NOT_IN_FASTA/2020" not in set(inst.data["id"])
    assert len(inst.data) == len(meta)
    assert inst.data["mutation instructions"].isna().sum() == 0
    # The index is contiguous again after the drop.
    assert list(inst.data.index) == list(range(len(inst.data)))


# ─────────────────────── FASTA reader / SequenceRecord ───────────────────────

FASTA = "tests/data/test1/test1.sequences.fasta"


def test_read_fasta_streams_all_records():
    reader = PyEvoMotionParser.read_fasta(FASTA)
    assert isinstance(reader, FastaReader)
    records = list(reader)
    assert len(records) == 101
    assert all(isinstance(r, SequenceRecord) for r in records)
    ids = [r.id for r in records]
    assert len(set(ids)) == 101                       # unique ids
    assert "hCoV-19/Wuhan/IVDC-HB-01/2019" in ids     # the reference is among them
    assert len(records[0]) == len(records[0].seq) > 29000
    assert set(records[0].seq.upper()) <= set("ACGTNRYKMSWBDHV-")


def test_read_fasta_matches_parse_sequence_by_id():
    wanted = "hCoV-19/Wuhan/IVDC-HB-01/2019"
    rec = PyEvoMotionParser.parse_sequence_by_id(FASTA, wanted)
    streamed = next(r for r in PyEvoMotionParser.read_fasta(FASTA) if r.id == wanted)
    assert rec == streamed and rec.seq == streamed.seq and rec.description == streamed.description


def test_fasta_parser_header_multiline_and_crlf(tmp_path):
    fa = tmp_path / "x.fasta"
    fa.write_bytes(b"junk before first header\r\n>seq1 some description here\r\nACGT\r\nac gt\r\n\r\n>seq2\nTT\n")
    recs = list(PyEvoMotionParser.read_fasta(str(fa)))
    assert [r.id for r in recs] == ["seq1", "seq2"]
    assert recs[0].description == "seq1 some description here"
    assert recs[0].seq == "ACGTacgt"          # whitespace dropped, case kept
    assert recs[1].seq == "TT" and recs[1].description == "seq2"
    assert recs[0].format() == ">seq1 some description here\nACGTacgt\n"
    assert recs[0].upper().seq == "ACGTACGT" and recs[0].lower().seq == "acgtacgt"
    assert repr(recs[1]) == "SequenceRecord(id='seq2', length=2)"


def test_read_fasta_missing_file_raises():
    with pytest.raises(IOError):
        PyEvoMotionParser.read_fasta("does/not/exist.fasta")


def test_generate_alignment_returns_two_records():
    ref = SequenceRecord("r", "ACGTACGTAC")
    tgt = SequenceRecord("t", "ACGTCGTAC")   # one deletion
    aln = PyEvoMotionParser.generate_alignment(ref, tgt)
    assert isinstance(aln, list) and len(aln) == 2
    assert all(isinstance(r, SequenceRecord) for r in aln)
    assert aln[0].id == "r" and aln[1].id == "t"
    assert len(aln[0].seq) == len(aln[1].seq)
    assert aln[0].seq.replace("-", "").upper() == "ACGTACGTAC"
    assert "-" in aln[1].seq
    assert PyEvoMotionParser.create_modifs(aln) == ["d_5_A"] or PyEvoMotionParser.create_modifs(aln)[0].startswith("d_")
