import os
import pytest
from datetime import datetime
from .helpers.test_UK_USA_dataset_helpers import check_data_exists, download_data_zip, extract_data_zip, generate_sampled_df, generate_figure_df

# Setup
@pytest.fixture
def setup():
    if not check_data_exists():
        download_data_zip()
        extract_data_zip()
    return datetime.now().strftime('%Y%m%d%H%M%S')


# General helper function for testing dataset parsing
def run_dataset_test(setup, meta_file_path, seq_file_path, output_prefix):
    """Abstracted logic to test PyEvoMotion on a dataset."""
    
    _date = setup
    _dt = "7D"
    _size = 100  # Feel free to change this value
    os.makedirs(f"tests/data/test3/output/{_date}", exist_ok=True)

    _filename = generate_sampled_df(
        meta_file_path,
        _date,
        _dt,
        _size
    )

    # Invoke PyEvoMotion as if it were a command line tool
    os.system(" ".join([
        f"poetry",
        "run",
        "PyEvoMotion",
        seq_file_path,
        _filename,
        f"tests/data/test3/output/{_date}/{output_prefix}",
        "-k", "total",
        "-n", "5",
        "-dt", _dt,
        "-dr", "2020-10-01..2021-08-01",
        "-ep",
        "-xj",
    ]))
    assert os.path.exists(f"tests/data/test3/output/{_date}/{output_prefix}_plots.pdf")

def run_fig_test(setup, set, meta_file_path, seq_file_path, output_prefix):
    _date = setup
    _dt = "7D"
    os.makedirs(f"tests/data/test3/output/{_date}", exist_ok=True)

    _filename = generate_figure_df(
        meta_file_path,
        _date,
        set
    )

    # Invoke PyEvoMotion as if it were a command line tool
    os.system(" ".join([
        f"poetry",
        "run",
        "PyEvoMotion",
        seq_file_path,
        _filename,
        f"tests/data/test3/output/{_date}/{output_prefix}",
        "-k", "total",
        "-n", "5",
        "-dt", _dt,
        "-dr", "2020-10-01..2021-08-01",
        "-ep",
        "-xj",
    ]))
    assert os.path.exists(f"tests/data/test3/output/{_date}/{output_prefix}_plots.pdf")

def test_UK_dataset(setup):
    """Tests that PyEvoMotion can parse the UK dataset correctly.
    """
    run_dataset_test(
        setup,
        "tests/data/test3/test3UK.tsv",
        "tests/data/test3/test3UK.fasta",
        "UKout"
    )

def test_USA_dataset(setup):
    """Tests that PyEvoMotion can parse the USA dataset correctly.
    """
    run_dataset_test(
        setup,
        "tests/data/test3/test3USA.tsv",
        "tests/data/test3/testUSA.fasta",
        "USAout"
    )

def test_UK_figure_dataset(setup):
    """Tests that PyEvoMotion can generate the UK corresponding figure from the manuscript.
    """
    run_fig_test(
        setup,
        "UK",
        "tests/data/test3/test3UK.tsv",
        "tests/data/test3/test3UK.fasta",
        "UKout_fig"
    )

def test_USA_figure_dataset(setup):
    """Tests that PyEvoMotion can generate the USA corresponding figure from the manuscript.
    """
    run_fig_test(
        setup,
        "USA",
        "tests/data/test3/test3USA.tsv",
        "tests/data/test3/testUSA.fasta",
        "USAout_fig"
    )