import os
import json
import zipfile
import warnings
import numpy as np
import pandas as pd
import urllib.request
from PyEvoMotion import PyEvoMotion

def check_data_exists() -> bool:
    """
    Check if the UK-USA dataset has been downloaded.
    """

    _files = [
        "test3UK.fasta",
        "testUSA.fasta",
        "test3UK.tsv",
        "test3USA.tsv"
    ]

    _parent_path = "tests/data/test3/"

    for file in _files:
        if not os.path.exists(os.path.join(_parent_path, file)):
            return False

    return True

def download_data_zip() -> None:
    """
    Download the UK-USA dataset from the repository.
    """
    warnings.warn("""
The necessary data for testing is not present.
Downloading the UK-USA dataset from
    https://sourceforge.net/projects/pyevomotion/files/test_data.zip
into
    tests/data/test3/test_data.zip
This may take a while.
"""
)
    urllib.request.urlretrieve(
        "https://sourceforge.net/projects/pyevomotion/files/test_data.zip/download",
        "tests/data/test3/test_data.zip"
    )

def extract_data_zip() -> None:
    """
    Extract the UK-USA dataset.
    """
    with zipfile.ZipFile("tests/data/test3/test_data.zip", "r") as zip_ref:
        zip_ref.extractall("tests/data/test3/")
    os.remove("tests/data/test3/test_data.zip")

def date_grouper(df: pd.DataFrame, DT: str, origin: str) -> pd.core.groupby.generic.DataFrameGroupBy:
    return PyEvoMotion.date_grouper(df, DT, origin)

def equal_date_distribution_sample(df: pd.DataFrame, DT: str, origin: str, n: int) -> pd.DataFrame:
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
    # group_sizes["size"].replace(np.inf, 0, inplace=True)
    # Normalize the weights
    group_sizes["size"] /= group_sizes["size"].sum()

    weigths_map = dict(zip(
        group_sizes["group"].to_list(),
        group_sizes["size"].to_list()
    ))


    weights = gb["date"].transform(
        lambda x: weigths_map[group_map[x.name]]
    )

    return df.sample(n=n, weights=weights)

def generate_sampled_df(
    file_path: str,
    date: str,
    DT: str,
    size: int = 100
) -> None:
    print(f"Generating sampled DataFrame for {file_path}")
    df = (
        pd.read_csv(
            file_path,
            sep="\t",
            index_col=0,
            parse_dates=["Date"],
        )
        .rename(columns={"ID": "id", "Date": "date"})
    )
    _origin = df["date"].min()

    _filename = f"tests/data/test3/output/{date}/sample_{os.path.basename(file_path).split(".")[0]}.tsv"
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
) -> None:
    print(f"Generating figure DataFrame for {file_path}")

    with open("tests/data/test3/ids_sampled_for_figure.json") as f:
        ids = json.load(f)[set]

    df = (
        pd.read_csv(
            file_path,
            sep="\t",
            index_col=0,
            parse_dates=["Date"],
        )
        .rename(columns={"ID": "id", "Date": "date"})
    )

    _filename = f"tests/data/test3/output/{date}/sample_{os.path.basename(file_path).split(".")[0]}.tsv"

    (
        df[df["id"].isin(ids)]
        .reset_index(drop=True)
        .to_csv(_filename, sep="\t")
    )
    return _filename