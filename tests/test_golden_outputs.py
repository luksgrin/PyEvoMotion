"""Byte-for-byte golden tests of the CLI on the bundled test1 dataset.

Outputs are platform-independent by design (stable sort by date, canonical
window statistics), so these files must be identical on every machine and
CI runner. See tests/helpers/golden.py for how to update them.
"""
import os
import subprocess
import pytest

from .helpers.golden import check_golden, ALL_FILES

SEQS = "tests/data/test1/test1.sequences.fasta"
META = "tests/data/test1/test1.metadata.tsv"


def _run(out_prefix, *extra):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    result = subprocess.run(["PyEvoMotion", SEQS, META, out_prefix, *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,args", [
    ("test1_default", []),
    ("test1_3D_filtered", ["-dt", "3D", "-k", "all", "-l", "25000", "-f", "country", "[China", "United Kingdom]", "-dr", "2020-01-01..2020-12-31"]),
    ("test1_substitutions_window", ["-k", "substitutions", "-gp", "1000..28000", "-cl", "0.9"]),
])
def test_cli_outputs_match_goldens(tmp_path, name, args):
    prefix = str(tmp_path / name / "out")
    _run(prefix, *args)
    check_golden(prefix, name, ALL_FILES)


def test_reload_reproduces_golden(tmp_path):
    """-load on the golden data table must give the golden statistics."""
    golden_tsv = "tests/data/golden/test1_default/out.tsv"
    if not os.path.exists(golden_tsv):
        pytest.skip("goldens not generated yet")
    prefix = str(tmp_path / "reload" / "out")
    _run(prefix, "-load", golden_tsv)
    check_golden(prefix, "test1_default")
