import os
import pytest
import subprocess
import pandas as pd
from PyEvoMotion import PyEvoMotionParser
from datetime import datetime
from .helpers.test_UK_USA_dataset_helpers import SAMPLE_SEED, use_full_data

# Each synthetic dataset has 2,001 sequences. Aligning all of them takes
# ~30 min per dataset on a GitHub runner, so by default the tests run on a
# fixed-seed sample (reference + CI_SAMPLE_SIZE sequences); set
# PYEVOMOTION_FULL_TEST_DATA=1 to use every sequence.
CI_SAMPLE_SIZE = 400

# Setup
@pytest.fixture
def setup():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def subsample_dataset(seq_file, meta_file, out_dir, n=CI_SAMPLE_SIZE, seed=SAMPLE_SEED):
    """Write a fixed-seed subset (reference + n sequences) of a synthetic
    dataset into ``out_dir`` and return the new (fasta, tsv) paths."""
    meta = pd.read_csv(meta_file, sep="\t", parse_dates=["date"])
    sample = meta.iloc[1:].sample(n=n, random_state=seed).sort_values("date", kind="stable")
    subset = pd.concat([meta.iloc[[0]], sample]).reset_index(drop=True)
    stem = os.path.splitext(os.path.basename(seq_file))[0]
    sub_tsv = f"{out_dir}/sample_{stem}.tsv"
    sub_fasta = f"{out_dir}/sample_{stem}.fasta"
    subset.to_csv(sub_tsv, sep="\t", index=False)
    wanted = set(subset["id"])
    with open(sub_fasta, "w") as out:
        for rec in PyEvoMotionParser.read_fasta(seq_file):
            if rec.id in wanted:
                out.write(rec.format())
    return sub_fasta, sub_tsv

def run_synthetic_test(setup, seq_file, meta_file, output_prefix, output_dir="test4"):
    """Abstracted logic to test PyEvoMotion on synthetic datasets."""
    
    _date = setup
    os.makedirs(f"tests/data/{output_dir}/output/{_date}", exist_ok=True)
    if not use_full_data():
        seq_file, meta_file = subsample_dataset(seq_file, meta_file, f"tests/data/{output_dir}/output/{_date}")

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

