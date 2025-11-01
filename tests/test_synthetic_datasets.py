import os
import pytest
import subprocess
from datetime import datetime

# Setup
@pytest.fixture
def setup():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def run_synthetic_test(setup, seq_file, meta_file, output_prefix, output_dir="test4"):
    """Abstracted logic to test PyEvoMotion on synthetic datasets."""
    
    _date = setup
    os.makedirs(f"tests/data/{output_dir}/output/{_date}", exist_ok=True)

    # Invoke PyEvoMotion as if it were a command line tool
    result = subprocess.run(
        [
            "PyEvoMotion",
            seq_file,
            meta_file,
            f"tests/data/{output_dir}/output/{_date}/{output_prefix}",
            "-ep",
            "-k", "substitutions"
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

    assert os.path.exists(f"tests/data/{output_dir}/output/{_date}/{output_prefix}_plots.pdf")

def test_S1_dataset(setup):
    """Tests that PyEvoMotion can process the S1 synthetic dataset correctly."""
    run_synthetic_test(
        setup,
        "tests/data/test4/S1.fasta",
        "tests/data/test4/S1.tsv",
        "synthdata1_out"
    )

def test_S2_dataset(setup):
    """Tests that PyEvoMotion can process the S2 synthetic dataset correctly."""
    run_synthetic_test(
        setup,
        "tests/data/test4/S2.fasta",
        "tests/data/test4/S2.tsv",
        "synthdata2_out"
    )

@pytest.mark.parametrize("dataset_num", [f"{i:02d}" for i in range(1, 2)]) # Run only 1 dataset to avoid github actions timeout
def test_linear_datasets(setup, dataset_num):
    """Tests that PyEvoMotion can process all linear synthetic datasets correctly."""
    run_synthetic_test(
        setup,
        f"tests/data/test5/linear/synthdata_linear_{dataset_num}.fasta",
        f"tests/data/test5/linear/synthdata_linear_{dataset_num}.tsv",
        f"linear_{dataset_num}_out",
        "test5/linear"
    )

@pytest.mark.parametrize("dataset_num", [f"{i:02d}" for i in range(1, 2)]) # Run only 1 dataset to avoid github actions timeout
def test_powerlaw_datasets(setup, dataset_num):
    """Tests that PyEvoMotion can process all powerlaw synthetic datasets correctly."""
    run_synthetic_test(
        setup,
        f"tests/data/test5/powerlaw/synthdata_powerlaw_{dataset_num}.fasta",
        f"tests/data/test5/powerlaw/synthdata_powerlaw_{dataset_num}.tsv",
        f"powerlaw_{dataset_num}_out",
        "test5/powerlaw"
    )

