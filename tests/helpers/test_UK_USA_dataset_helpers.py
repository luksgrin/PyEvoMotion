import os
import json
import tarfile
import zipfile
import warnings
import numpy as np
import pandas as pd
import urllib.request
from PyEvoMotion import PyEvoMotion

# ─────────────────────── data location ───────────────────────
#
# Two flavours of the UK/USA dataset exist:
#
# * the CI subset (default): a fixed-seed, date-stratified sample of
#   N sequences per set drawn from the manuscript sample, ~20 MB, hosted as
#   an asset of the GitHub release `ci-data-<version>` and extracted into
#   tests/data/test3/ci/. Built by tests/helpers/make_ci_subset.py.
# * the full dataset (~11 GB): downloaded from SourceForge into
#   tests/data/test3/ when PYEVOMOTION_FULL_TEST_DATA is set.

ROOT = "tests/data/test3"
CI_DIR = f"{ROOT}/ci"
CI_DATA_VERSION = "v1"
CI_DATA_URL = (
    "https://github.com/luksgrin/PyEvoMotion/releases/download/"
    f"ci-data-{CI_DATA_VERSION}/test3-ci-subset-{CI_DATA_VERSION}.tar.gz"
)
FULL_DATA_URL = "https://sourceforge.net/projects/pyevomotion/files/test_data.zip/download"
FULL_DATA_ENV = "PYEVOMOTION_FULL_TEST_DATA"

# Seed for every random sample the tests draw, so a run is reproducible.
SAMPLE_SEED = 20260905

_FILES = ["test3UK.fasta", "test3USA.fasta", "test3UK.tsv", "test3USA.tsv"]


def use_full_data() -> bool:
    return os.environ.get(FULL_DATA_ENV, "").lower() not in ("", "0", "false", "no")


def data_dir() -> str:
    return ROOT if use_full_data() else CI_DIR


def data_paths(set_: str) -> tuple[str, str]:
    """(metadata TSV, FASTA) for ``set_`` in {"UK", "USA"} in the active flavour."""
    d = data_dir()
    return f"{d}/test3{set_}.tsv", f"{d}/test3{set_}.fasta"


def figure_ids_path() -> str:
    return f"{ROOT}/ids_sampled_for_figure.json" if use_full_data() else f"{ROOT}/ids_sampled_for_ci.json"


def check_data_exists() -> bool:
    """Check that the active flavour of the UK/USA dataset is present."""
    return all(os.path.exists(os.path.join(data_dir(), f)) for f in _FILES)


def download_data_zip() -> None:
    """Download the full UK/USA dataset (~11 GB) from SourceForge."""
    warnings.warn(f"""
The full UK/USA dataset is not present.
Downloading it from
    {FULL_DATA_URL}
into
    {ROOT}/test_data.zip
This may take a while.
""")
    urllib.request.urlretrieve(FULL_DATA_URL, f"{ROOT}/test_data.zip")


def extract_data_zip() -> None:
    """Extract the full UK/USA dataset and remove the archive."""
    with zipfile.ZipFile(f"{ROOT}/test_data.zip", "r") as zip_ref:
        zip_ref.extractall(f"{ROOT}/")
    os.remove(f"{ROOT}/test_data.zip")


def download_ci_subset() -> None:
    """Download and extract the CI subset from the GitHub release."""
    os.makedirs(CI_DIR, exist_ok=True)
    archive = f"{CI_DIR}/subset.tar.gz"
    warnings.warn(f"Downloading the CI subset of the UK/USA dataset from {CI_DATA_URL}")
    urllib.request.urlretrieve(CI_DATA_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(CI_DIR)
    os.remove(archive)


def ensure_data() -> None:
    """Make sure the active flavour of the dataset is on disk."""
    if check_data_exists():
        return
    if use_full_data():
        download_data_zip()
        extract_data_zip()
    else:
        download_ci_subset()


# ─────────────────────── sampling ───────────────────────

def date_grouper(df: pd.DataFrame, DT: str, origin: str) -> pd.core.groupby.generic.DataFrameGroupBy:
    return PyEvoMotion.date_grouper(df, DT, origin)

def equal_date_distribution_sample(
    df: pd.DataFrame,
    DT: str,
    origin: str,
    n: int,
    random_state: int | None = SAMPLE_SEED,
) -> pd.DataFrame:
    """
    Sample the input DataFrame with equal distribution of dates to minimize sampling bias.
    """
    gb = date_grouper(df, DT, origin)
    group_sizes = gb.size().reset_index()
    group_sizes.columns = ["date", "size"]

    # Assign name to each group
    group_map = {key:f"group {idx}" for idx, (key, _) in enumerate(gb)}

    # Apply group name to each group
    group_sizes["group"] = group_sizes["date"].map(group_map)

    # Calculate weights
    group_sizes["size"] = 1/group_sizes["size"]
    # Handle the divisions by zero
    group_sizes.replace({"size": {np.inf: 0}}, inplace=True)
    # Normalize the weights
    group_sizes["size"] /= group_sizes["size"].sum()

    weigths_map = dict(zip(
        group_sizes["group"].to_list(),
        group_sizes["size"].to_list()
    ))

    weights = gb["date"].transform(
        lambda x: weigths_map[group_map[x.name]]
    )

    return df.sample(n=n, weights=weights, random_state=random_state)

def generate_sampled_df(
    file_path: str,
    date: str,
    DT: str,
    size: int = 100
) -> str:
    print(f"Generating sampled DataFrame for {file_path}")
    df = (
        pd.read_csv(
            file_path,
            sep="\t",
            index_col=0,
            parse_dates=["date"],
        )
    )
    _origin = df["date"].min()

    _filename = f"{ROOT}/output/{date}/sample_{os.path.basename(file_path).split('.')[0]}.tsv"
    (
        pd.concat([
            df.iloc[[0]],
            equal_date_distribution_sample(df.iloc[1:,:], DT, _origin, size)
        ])
        .reset_index(drop=True)
        .to_csv(_filename, sep="\t")
    )
    return _filename

def generate_figure_df(
    file_path: str,
    date: str,
    set: str,
) -> str:
    print(f"Generating figure DataFrame for {file_path}")

    with open(figure_ids_path()) as f:
        ids = json.load(f)[set]

    df = (
        pd.read_csv(
            file_path,
            sep="\t",
            index_col=0,
            parse_dates=["date"],
        )
    )

    _filename = f"{ROOT}/output/{date}/sample_{os.path.basename(file_path).split('.')[0]}.tsv"

    (
        df[df["id"].isin(ids)]
        .reset_index(drop=True)
        .to_csv(_filename, sep="\t")
    )
    return _filename
