import os
import pytest
import subprocess
from datetime import datetime

# Setup
@pytest.fixture
def setup():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def run_synthetic_test(setup, seq_file, meta_file, output_prefix):
    """Abstracted logic to test PyEvoMotion on synthetic datasets."""
    
    _date = setup
    os.makedirs(f"tests/data/synthetic/output/{_date}", exist_ok=True)

    # Invoke PyEvoMotion as if it were a command line tool
    result = subprocess.run(
        [
            "PyEvoMotion",
            seq_file,
            meta_file,
            f"tests/data/synthetic/output/{_date}/{output_prefix}",
            "-ep"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check for errors
    if result.stderr:
        print(result.stdout)
        print(result.stderr)
        pytest.fail(f"PyEvoMotion failed with error: {result.stderr}")

    assert os.path.exists(f"tests/data/synthetic/output/{_date}/{output_prefix}_plots.pdf")

def test_S1_dataset(setup):
    """Tests that PyEvoMotion can process the S1 synthetic dataset correctly."""
    run_synthetic_test(
        setup,
        "S1.fasta",
        "S1.tsv",
        "synthdata1_out"
    )

def test_S2_dataset(setup):
    """Tests that PyEvoMotion can process the S2 synthetic dataset correctly."""
    run_synthetic_test(
        setup,
        "S2.fasta",
        "S2.tsv",
        "synthdata2_out"
    ) 