import os
import pytest
import subprocess
from datetime import datetime
from .helpers.test_UK_USA_dataset_helpers import ensure_data, data_paths, generate_sampled_df, generate_figure_df

# These tests run against the UK/USA dataset. By default they use the small
# fixed-seed CI subset (downloaded from the GitHub release `ci-data-<version>`
# when absent); set PYEVOMOTION_FULL_TEST_DATA=1 to run them on the full 11 GB
# dataset instead. See tests/helpers/test_UK_USA_dataset_helpers.py.

# Setup
@pytest.fixture
def setup():
    ensure_data()
    return datetime.now().strftime('%Y%m%d%H%M%S')


# General helper function for testing dataset parsing
def run_dataset_test(setup, set, output_prefix):
    """Abstracted logic to test PyEvoMotion on a dataset."""
    meta_file_path, seq_file_path = data_paths(set)

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
    result = subprocess.run(
        [
            "PyEvoMotion",
            seq_file_path,
            _filename,
            f"tests/data/test3/output/{_date}/{output_prefix}",
            "-k", "total",
            "-dt", _dt,
            "-dr", "2020-10-01..2021-08-01",
            "-ep",
            "-xj",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check for known errors that happen when random sampling is defficient
    if ("ValueError: No groups with at least 2 observations" in result.stderr) or ("ValueError: The dataset is (almost) empty at this point of the analysis." in result.stderr):
        pytest.skip("Skipped due to insufficient observations in random input. Consider re-running this particular test.")

    assert os.path.exists(f"tests/data/test3/output/{_date}/{output_prefix}_plots.pdf")

def run_fig_test(setup, set, output_prefix, dt="7D", load_from=None):
    """Run the figure-style analysis for ``set``.

    With ``load_from`` (the ``<out>.tsv`` of a previous run) the alignment
    step is skipped via ``-load`` and only the time-window analysis is redone.
    Returns the output prefix of this run."""
    meta_file_path, seq_file_path = data_paths(set)
    _date = setup
    _dt = dt
    os.makedirs(f"tests/data/test3/output/{_date}", exist_ok=True)

    _filename = load_from or generate_figure_df(
        meta_file_path,
        _date,
        set
    )

    # Invoke PyEvoMotion as if it were a command line tool
    result = subprocess.run(
        [
            "PyEvoMotion",
            seq_file_path,
            _filename if load_from is None else meta_file_path,
            f"tests/data/test3/output/{_date}/{output_prefix}",
            "-k", "total",
            "-dt", _dt,
            "-dr", "2020-10-01..2021-08-01",
            "-ep",
            "-xj",
        ] + (["-load", load_from] if load_from else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check for known errors that happen when random sampling is defficient
    if ("ValueError: No groups with at least 2 observations" in result.stderr) or ("ValueError: The dataset is (almost) empty at this point of the analysis." in result.stderr):
        pytest.skip("Skipped due to insufficient observations in random input. Consider re-running this particular test.")
    if result.stderr:
        print(result.stdout)
        print(result.stderr)
    assert os.path.exists(f"tests/data/test3/output/{_date}/{output_prefix}_plots.pdf")
    return f"tests/data/test3/output/{_date}/{output_prefix}"


@pytest.fixture(scope="module")
def uk_figure_run():
    """The UK figure analysis, run once per module. Aligning the sequences is
    the expensive step; the window-size tests below reuse this run's data
    table through ``-load`` instead of aligning again."""
    ensure_data()
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    return run_fig_test(date, "UK", "UKout_fig")

def test_UK_dataset(setup):
    """Tests that PyEvoMotion can parse the UK dataset correctly.
    """
    run_dataset_test(
        setup,
        "UK",
        "UKout"
    )

def test_USA_dataset(setup):
    """Tests that PyEvoMotion can parse the USA dataset correctly.
    """
    run_dataset_test(
        setup,
        "USA",
        "USAout"
    )

def test_UK_figure_dataset(uk_figure_run):
    """Tests that PyEvoMotion can generate the UK corresponding figure from the manuscript.
    """
    for suffix in (".tsv", "_stats.tsv", "_regression_results.json", "_plots.pdf"):
        assert os.path.exists(uk_figure_run + suffix)

def test_USA_figure_dataset(setup):
    """Tests that PyEvoMotion can generate the USA corresponding figure from the manuscript.
    """
    run_fig_test(
        setup,
        "USA",
        "USAout_fig"
    )

def test_UK_5D_dataset(setup, uk_figure_run):
    """Tests that PyEvoMotion runs correctly with a 5D time-window, reusing
    the UK figure run's mutation instructions via -load.
    """
    run_fig_test(
        setup,
        "UK",
        "UKout_5D",
        dt="5D",
        load_from=uk_figure_run + ".tsv",
    )

def test_UK_10D_dataset(setup, uk_figure_run):
    """Tests that PyEvoMotion runs correctly with a 10D time-window, reusing
    the UK figure run's mutation instructions via -load.
    """
    run_fig_test(
        setup,
        "UK",
        "UKout_10D",
        dt="10D",
        load_from=uk_figure_run + ".tsv",
    )

def test_UK_14D_dataset(setup, uk_figure_run):
    """Tests that PyEvoMotion runs correctly with a 14D time-window, reusing
    the UK figure run's mutation instructions via -load.
    """
    run_fig_test(
        setup,
        "UK",
        "UKout_14D",
        dt="14D",
        load_from=uk_figure_run + ".tsv",
    )