import os
import sys
import json
import zipfile
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                         FUNCTIONS                          #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

def set_matplotlib_global_params() -> None:
    mpl_params = {
        "font.sans-serif": "Helvetica",
        "axes.linewidth": 2,
        "axes.labelsize": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 20,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.frameon": False,
    }
    for k, v in mpl_params.items(): mpl.rcParams[k] = v

def check_test_data_exists() -> bool:
    """
    Check if the UK-USA dataset has been downloaded.
    """

    _files = [
        "test3UK.fasta",
        "test3USA.fasta",
        "test3UK.tsv",
        "test3USA.tsv"
    ]

    _parent_path = "tests/data/test3/"

    for file in _files:
        if not os.path.exists(os.path.join(_parent_path, file)):
            return False

    return True

def download_test_data_zip() -> None:
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

def extract_test_data_zip() -> None:
    """
    Extract the UK-USA dataset.
    """
    with zipfile.ZipFile("tests/data/test3/test_data.zip", "r") as zip_ref:
        zip_ref.extractall("tests/data/test3/")
    os.remove("tests/data/test3/test_data.zip")

def check_fig_data_exists() -> bool:
    """
    Check if the figure data files exist.
    """
    _files = [
        "share/figdataUK.tsv",
        "share/figdataUSA.tsv"
    ]

    for file in _files:
        if not os.path.exists(file):
            return False

    return True

def create_fig_data() -> None:
    print("Creating figure data files for the manuscript...")
    with open("tests/data/test3/ids_sampled_for_figure.json") as f:
        ids = json.load(f)

    if not check_test_data_exists():
        print("The necessary data for testing is not present. Downloading it now...")
        download_test_data_zip()
        extract_test_data_zip()

    for country in ["UK", "USA"]:
        df = (
            pd.read_csv(
                f"tests/data/test3/test3{country}.tsv",
                sep="\t",
                index_col=0,
                parse_dates=["date"],
            )
        )
        (
            df[df["id"].isin(ids[country])]
            .reset_index(drop=True)
            .to_csv(f"share/figdata{country}.tsv", sep="\t")
        )

def check_final_data_and_models_exist() -> bool:
    """
    Check if the final data files and models exist.
    """
    _files = [
        "share/figUSA_stats.tsv",
        "share/figUK_stats.tsv",
        "share/figUSA_regression_results.json",
        "share/figUK_regression_results.json"
    ]

    for file in _files:
        if not os.path.exists(file):
            return False

    return True

def load_final_data_df() -> pd.DataFrame:
    return pd.read_csv(
        "share/figUSA_stats.tsv",
        sep="\t",
    ).merge(
        pd.read_csv(
            "share/figUK_stats.tsv",
            sep="\t",
        ),
        on="date",
        how="outer",
        suffixes=(" USA", " UK"),
    )

def load_models() -> dict[str, dict[str, callable]]:
    _kinds = ("USA", "UK")
    _file = "share/fig{}_regression_results.json"

    _contents = {}

    for k in _kinds:
        with open(_file.format(k)) as f:
            _contents[k] = json.load(f)

    return {
        "USA": {
            "mean": [
                lambda x: (
                    _contents["USA"]["mean number of mutations per 7D model"]["parameters"]["m"]*x
                    + _contents["USA"]["mean number of mutations per 7D model"]["parameters"]["b"]
                ),
                _contents["USA"]["mean number of mutations per 7D model"]["r2"],
            ],
            "var": [
                lambda x: (
                    _contents["USA"]["scaled var number of mutations per 7D model"]["parameters"]["d"]
                    *(x**_contents["USA"]["scaled var number of mutations per 7D model"]["parameters"]["alpha"])
                ),
                _contents["USA"]["scaled var number of mutations per 7D model"]["r2"],
            ]
        },
        "UK": {
            "mean": [
                lambda x: (
                    _contents["UK"]["mean number of mutations per 7D model"]["parameters"]["m"]*x
                    + _contents["UK"]["mean number of mutations per 7D model"]["parameters"]["b"]
                ),
                _contents["UK"]["mean number of mutations per 7D model"]["r2"],
            ],
            "var": [
                lambda x: (
                    _contents["UK"]["scaled var number of mutations per 7D model"]["parameters"]["d"]
                    *(x**_contents["UK"]["scaled var number of mutations per 7D model"]["parameters"]["alpha"])
                ),
                _contents["UK"]["scaled var number of mutations per 7D model"]["r2"],
            ]
        },
    }

def safe_map(f: callable, x: list[int | float]) -> list[int | float]:
    _results = []
    for el in x:
        try: _results.append(f(el))
        except Exception as e:
            print(f"WARNING: {e}")
            _results.append(None)
    return _results

def plot(df: pd.DataFrame, models: dict[str, any], export: bool = False, show: bool = True) -> None:
    set_matplotlib_global_params()
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    colors = {
        "UK": "#76d6ff",
        "USA": "#FF6346",
    }

    for idx, case in enumerate(("mean", "var")):
        for col in (f"{case} number of mutations USA", f"{case} number of mutations UK"):

            _country = col.split()[-1].upper()

            ax[idx].scatter(
                df.index,
                df[col] - (df[col].min() if idx == 1 else 0),
                color=colors[_country],
                edgecolor="k",
                zorder=2,   
            )
            
            _x = np.arange(-10, 50, 0.5) 
            ax[idx].plot(
                _x + (8 if _country == "USA" else 0),
                safe_map(models[_country][case][0], _x),
                color=colors[_country],
                label=rf"{_country} ($R^2 = {round(models[_country][case][1], 2):.2f})$",
                linewidth=3,
                zorder=1,
            )

            # Styling
            ax[idx].set_xlim(-0.5, 40.5)
            ax[idx].set_ylim(30, 50) if idx == 0 else ax[idx].set_ylim(0, 16)

            ax[idx].set_xlabel("time (wk)")

            if case == "mean":
                ax[idx].set_ylabel(f"{case}  (# mutations)")
            elif case == "var":
                ax[idx].set_ylabel(f"{case}iance  (# mutations)")
            
            ax[idx].set_xticks(np.arange(0, 41, 10))
            ax[idx].set_yticks(np.arange(30, 51, 5)) if idx == 0 else ax[idx].set_yticks(np.arange(0, 17, 4))

        ax[idx].legend(
            fontsize=16,
            loc="upper left",
        )

    fig.suptitle(" ", fontsize=1) # To get some space on top
    fig.tight_layout()
    plt.annotate("a", (0.02, 0.94), xycoords="figure fraction", fontsize=28, fontweight="bold")
    plt.annotate("b", (0.02, 0.47), xycoords="figure fraction", fontsize=28, fontweight="bold")

    if export:
        fig.savefig(
            "share/figure.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        print("Figure saved as share/figure.pdf")

    if show: plt.show()

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                           MAIN                             #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

def main(export: bool = False) -> None:

    if not check_final_data_and_models_exist():
        print("Final data files do not exist. Creating them...")

        if not check_fig_data_exists():
            print("Figure data files do not exist. Creating them...")
            create_fig_data()

        for country in ["UK", "USA"]:
            # Invoke PyEvoMotion as if it were a command line tool
            print(f"Running PyEvoMotion for {country}...")
            os.system(" ".join([
                "PyEvoMotion",
                f"tests/data/test3/test3{country}.fasta",
                f"share/figdata{country}.tsv",
                f"share/fig{country}",
                "-k", "total",
                "-n", "5",
                "-dt", "7D",
                "-dr", "2020-10-01..2021-08-01",
                "-ep",
                "-xj",
            ]))

    # Load plot data & models
    df = load_final_data_df()
    models = load_models()

    # Plot
    plot(df, models, export=export)


if __name__ == "__main__":

    # Doing this way to not raise an out of bounds error when running the script without arguments
    _export_flag = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            _export_flag = True

    main(export=_export_flag)