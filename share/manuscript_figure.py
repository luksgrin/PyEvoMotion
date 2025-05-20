import os
import sys
import json
import zipfile
import warnings
import urllib.request
import subprocess

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                         CONSTANTS                          #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

COLORS = {
    "UK": "#76d6ff",
    "USA": "#FF6346",
}

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

def plot_main_figure(df: pd.DataFrame, models: dict[str, any], export: bool = False, show: bool = True) -> None:
    set_matplotlib_global_params()
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    for idx, case in enumerate(("mean", "var")):
        for col in (f"{case} number of mutations USA", f"{case} number of mutations UK"):

            _country = col.split()[-1].upper()

            ax[idx].scatter(
                df.index,
                df[col] - (df[col].min() if idx == 1 else 0),
                color=COLORS[_country],
                edgecolor="k",
                zorder=2,   
            )
            
            _x = np.arange(-10, 60, 0.5) 
            ax[idx].plot(
                _x + (8 if _country == "USA" else 0),
                safe_map(models[_country][case][0], _x),
                color=COLORS[_country],
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

def size_plot(df: pd.DataFrame, export: bool = False, show: bool = True) -> None:
    set_matplotlib_global_params()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Plot UK first
    markerline, stemlines, baseline = ax.stem(df.index, df[f"size UK"], label="UK")
    plt.setp(stemlines, color=COLORS["UK"])
    plt.setp(markerline, color=COLORS["UK"], markeredgecolor="k")
    plt.setp(baseline, color="#ffffff")

    # Plot USA
    markerline, stemlines, baseline = ax.stem(df.index, df[f"size USA"], label="USA")
    plt.setp(stemlines, color=COLORS["USA"])
    plt.setp(markerline, color=COLORS["USA"], markeredgecolor="k")
    plt.setp(baseline, color="#ffffff")

    # Plot UK again but with slight transparency on the stem
    markerline, stemlines, baseline = ax.stem(df.index, df[f"size UK"])
    plt.setp(stemlines, color=COLORS["UK"], alpha=0.5)
    plt.setp(markerline, color=COLORS["UK"], markeredgecolor="#000000")
    plt.setp(baseline, color="#ffffff")

    ax.set_ylim(0, 405)
    ax.set_xlim(-0.5, 40.5)

    ax.set_xlabel("time (wk)")
    ax.set_ylabel("Number of sequences")
    
    ax.legend(
        fontsize=16,
        loc="upper right",
        bbox_to_anchor=(1.08, 1.08)
    )

    if export:
        fig.savefig(
            "share/weekly_size.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        print("Figure saved as share/weekly_size.pdf")

    if show: plt.show()

def anomalous_diffusion_plot(export: bool = False, show: bool = True) -> None:
    set_matplotlib_global_params()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    x = np.linspace(0, 10, 100)

    plt.plot(x, x**0.8, label=r"$\alpha = 0.8$" + "\n(subdiffusion)", color=COLORS["UK"], linewidth=3)
    plt.plot(x, x**1, label=r"$\alpha = 1$" + "\n(normal diffusion)", color="#000000", linewidth=3)
    plt.plot(x, x**1.2, label=r"$\alpha = 1.2$" + "\n(superdiffusion)", color=COLORS["USA"], linewidth=3)
    
    plt.legend(
        fontsize=13,
        loc="upper left",
        title=r"variance $\propto \text{time}^\alpha$",
        title_fontsize=15
    )
    
    plt.xlabel("time")
    plt.ylabel("variance")

    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    if export:
        fig.savefig(
            "share/anomalous_diffusion.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        print("Figure saved as share/anomalous_diffusion.pdf")

    if show: plt.show()

def check_synthetic_data_exists() -> bool:
    """
    Check if the synthetic data output files exist.
    """
    _files = [
        "tests/data/test4/synthdata1_out_stats.tsv",
        "tests/data/test4/synthdata2_out_stats.tsv",
        "tests/data/test4/synthdata1_out_regression_results.json",
        "tests/data/test4/synthdata2_out_regression_results.json"
    ]

    for file in _files:
        if not os.path.exists(file):
            return False

    return True

def run_synthetic_data_tests() -> None:
    """
    Run the synthetic data tests to generate the required files.
    """
    print("Running synthetic data tests to generate required files...")
    
    # Create output directory
    os.makedirs("tests/data/test4", exist_ok=True)
    
    # Run tests for S1 dataset
    result1 = subprocess.run(
        [
            "PyEvoMotion",
            "S1.fasta",
            "S1.tsv",
            "tests/data/test4/synthdata1_out",
            "-ep"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result1.stderr:
        print(result1.stdout)
        print(result1.stderr)
        raise RuntimeError("Failed to process S1 dataset")
    
    # Run tests for S2 dataset
    result2 = subprocess.run(
        [
            "PyEvoMotion",
            "S2.fasta",
            "S2.tsv",
            "tests/data/test4/synthdata2_out",
            "-ep"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result2.stderr:
        print(result2.stdout)
        print(result2.stderr)
        raise RuntimeError("Failed to process S2 dataset")

def load_synthetic_data_df() -> pd.DataFrame:
    if not check_synthetic_data_exists():
        run_synthetic_data_tests()
    
    return pd.read_csv(
        "tests/data/test4/synthdata1_out_stats.tsv",
        sep="\t",
    ).merge(
        pd.read_csv(
            "tests/data/test4/synthdata2_out_stats.tsv",
            sep="\t",
        ),
        on="date",
        how="outer",
        suffixes=(" synt1", " synt2"),
    )

def load_synthetic_data_models() -> dict[str, dict[str, callable]]:
    if not check_synthetic_data_exists():
        run_synthetic_data_tests()
    
    _kinds = ("synt1", "synt2")
    _file = "tests/data/test4/synthdata{}_out_regression_results.json"

    _contents = {}

    for k in _kinds:
        with open(_file.format(k[-1])) as f:
            _contents[k] = json.load(f)

    return {
        "synt1": {
            "mean": [
                lambda x: (
                    _contents["synt1"]["mean number of mutations per 7D model"]["parameters"]["m"]*x
                    + _contents["synt1"]["mean number of mutations per 7D model"]["parameters"]["b"]
                ),
                _contents["synt1"]["mean number of mutations per 7D model"]["r2"],
            ],
            "var": [
                lambda x: (
                    _contents["synt1"]["scaled var number of mutations per 7D model"]["parameters"]["m"]*x
                ),
                _contents["synt1"]["scaled var number of mutations per 7D model"]["r2"],
            ]
        },
        "synt2": {
            "mean": [
                lambda x: (
                    _contents["synt2"]["mean number of mutations per 7D model"]["parameters"]["m"]*x
                    + _contents["synt2"]["mean number of mutations per 7D model"]["parameters"]["b"]
                ),
                _contents["synt2"]["mean number of mutations per 7D model"]["r2"],
            ],
            "var": [
                lambda x: (
                    _contents["synt2"]["scaled var number of mutations per 7D model"]["parameters"]["d"]
                    *(x**_contents["synt2"]["scaled var number of mutations per 7D model"]["parameters"]["alpha"])
                ),
                _contents["synt2"]["scaled var number of mutations per 7D model"]["r2"],
            ]
        },
    }

def synthetic_data_plot(df: pd.DataFrame, models: dict[str, any], export: bool = False, show: bool = True) -> None:
    set_matplotlib_global_params()
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Flatten axes for easier iteration
    ax = ax.flatten()

    # Plot counter for subplot index
    plot_idx = 0

    for case in ("mean", "var"):
        for col in (f"{case} number of mutations synt1", f"{case} number of mutations synt2"):
            _type = col.split()[-1].upper()

            # Scatter plot  
            ax[plot_idx].scatter(
                df.index,
                df[col],
                color="#76d6ff",
                edgecolor="k",
                zorder=2,   
            )
            
            # Line plot
            _x = np.arange(-10, 50, 0.5)
            ax[plot_idx].plot(
                _x,
                safe_map(models[_type.lower()][case][0], _x),
                color="#76d6ff",
                label=rf"$R^2 = {round(models[_type.lower()][case][1], 2):.2f}$",
                linewidth=3,
                zorder=1,
            )

            # Styling
            ax[plot_idx].set_xlim(-0.5, 40.5)
            if case == "mean":
                ax[plot_idx].set_ylim(-0.25, 20.25)
                ax[plot_idx].set_ylabel(f"{case}  (# mutations)")
            else:  # var case
                if _type == "SYNT1":
                    ax[plot_idx].set_ylim(-0.5, 40.5)
                else:
                    ax[plot_idx].set_ylim(-0.1, 10.1)
                ax[plot_idx].set_ylabel(f"{case}iance  (# mutations)")

            ax[plot_idx].set_xlabel("time (wk)")
            ax[plot_idx].legend(
                fontsize=16,
                loc="upper left",
            )

            plot_idx += 1

    fig.suptitle(" ", fontsize=1) # To get some space on top
    fig.tight_layout()
    
    # Add subplot annotations
    plt.annotate("a", (0.02, 0.935), xycoords="figure fraction", fontsize=28, fontweight="bold")
    plt.annotate("b", (0.505, 0.935), xycoords="figure fraction", fontsize=28, fontweight="bold")
    plt.annotate("c", (0.02, 0.465), xycoords="figure fraction", fontsize=28, fontweight="bold")
    plt.annotate("d", (0.505, 0.465), xycoords="figure fraction", fontsize=28, fontweight="bold")

    if export:
        fig.savefig(
            "share/synth_figure.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        print("Figure saved as share/synth_figure.pdf")

    if show: plt.show()
    

def load_additional_uk_stats() -> dict[str, pd.DataFrame]:
    """
    Load the additional UK stats files for different time windows.
    """
    _files = {
        "5D": "tests/data/test3/output/20250517164757/UKout_5D_stats.tsv",
        "10D": "tests/data/test3/output/20250517173133/UKout_10D_stats.tsv",
        "14D": "tests/data/test3/output/20250517181004/UKout_14D_stats.tsv",
        "7D": "share/figUK_stats.tsv"
    }
    
    return {
        k: pd.read_csv(v, sep="\t")
        for k, v in _files.items()
    }

def load_additional_uk_models() -> dict[str, dict[str, callable]]:
    """
    Load the additional UK models for different time windows.
    """
    _files = {
        "5D": "tests/data/test3/output/20250517164757/UKout_5D_regression_results.json",
        "10D": "tests/data/test3/output/20250517173133/UKout_10D_regression_results.json",
        "14D": "tests/data/test3/output/20250517181004/UKout_14D_regression_results.json",
        "7D": "share/figUK_regression_results.json"
    }
    
    _contents = {}
    for k, v in _files.items():
        with open(v) as f:
            _contents[k] = json.load(f)
    return {
        k: {
            "mean": [
                {
                    "m": _contents[k][f"mean number of mutations per {k} model"]["parameters"]["m"],
                    "b": _contents[k][f"mean number of mutations per {k} model"]["parameters"]["b"]
                },
                _contents[k][f"mean number of mutations per {k} model"]["r2"]
            ],
            "var": [
                {
                    "d": _contents[k][f"scaled var number of mutations per {k} model"]["parameters"]["d"],
                    "alpha": _contents[k][f"scaled var number of mutations per {k} model"]["parameters"]["alpha"]
                },
                _contents[k][f"scaled var number of mutations per {k} model"]["r2"],
            ]
        }
        for k in _files.keys()
    }

def plot_uk_time_windows(stats: dict[str, pd.DataFrame], models: dict[str, dict[str, callable]], export: bool = False, show: bool = True) -> None:
    """
    Plot a 1x4 subplot of UK data with different time windows.
    
    Args:
        stats: Dictionary of dataframes containing the stats for each time window
        models: Dictionary of models for each time window
        export: Whether to export the figure
        show: Whether to show the figure
    """
    set_matplotlib_global_params()
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    
    # Order of time windows to plot
    windows = ["5D", "7D", "10D", "14D"]

    for idx, window in enumerate(windows):
        df = stats[window]
        model = models[window]
        scaling = {
            "5D": 5/7,
            "7D": 1,
            "10D": 10/7,
            "14D": 14/7,
        }
        for idx2, case in enumerate(("mean", "var")):

            if case == "mean":
                # Plot mean
                ax[idx2, idx].scatter(
                    df.index.to_numpy()*scaling[window],
                    df["mean number of mutations"],
                    color=COLORS["UK"],
                    edgecolor="k",
                    zorder=2,
                )
            
                _x = np.arange(-0.5, 51, 0.5)
                ax[idx2, idx].plot(
                    _x,
                    model["mean"][0]["m"]*(_x/scaling[window]) + model["mean"][0]["b"],
                    color=COLORS["UK"],
                    label=rf"Mean ($R^2 = {round(model['mean'][1], 2):.2f})$",
                    linewidth=3,
                    zorder=1,
                )

            elif case == "var":
                # Plot variance 
                ax[idx2, idx].scatter(
                    df.index.to_numpy()*scaling[window],
                    df["var number of mutations"] - df["var number of mutations"].min(),
                    color=COLORS["UK"],
                    edgecolor="k",
                    zorder=2,
                )
                
                ax[idx2, idx].plot(
                    _x,
                    model["var"][0]["d"]*(_x/scaling[window])**model["var"][0]["alpha"],
                    color=COLORS["UK"],
                    label=rf"Var ($R^2 = {round(model['var'][1], 2):.2f})$",
                    linewidth=3,
                    zorder=1,
                )
            
            # Styling
            ax[idx2, idx].set_xlim(-0.5, 40.5)
            
            if case == "mean":
                ax[idx2, idx].set_ylim(29.5, 45.5)
            else:
                ax[idx2, idx].set_ylim(-0.5, 10.5)
            
            ax[idx2, idx].set_xlabel("time (wk)")
            if idx == 0:
                ax[idx2, idx].set_ylabel(f"{case} (# mutations)")

            ax[idx2, idx].legend(
                fontsize=12,
                loc="upper left",
            )
    
    if export:
        fig.savefig(
            "share/uk_time_windows.pdf",
            dpi=400,
            bbox_inches="tight",
        )
    
    if show:
        plt.show()

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
                "-dt", "7D",
                "-dr", "2020-10-01..2021-08-01",
                "-ep",
                "-xj",
            ]))

    # Load plot data & models
    df = load_final_data_df()
    models = load_models()

    # Main plot
    plot_main_figure(df, models, export=export)

    # Size plot
    size_plot(df, export=export)

    # Anomalous diffusion plot
    anomalous_diffusion_plot(export=export)

    # Synthetic data plot
    synth_df = load_synthetic_data_df()
    synth_models = load_synthetic_data_models()
    synthetic_data_plot(synth_df, synth_models, export=export)

    # UK time windows plot
    additional_uk_stats = load_additional_uk_stats()
    additional_uk_models = load_additional_uk_models()
    plot_uk_time_windows(additional_uk_stats, additional_uk_models, export=export)


if __name__ == "__main__":

    # Doing this way to not raise an out of bounds error when running the script without arguments
    _export_flag = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            _export_flag = True

    main(export=_export_flag)