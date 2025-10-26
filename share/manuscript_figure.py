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
from matplotlib.colors import LinearSegmentedColormap

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                         CONSTANTS                          #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

COLORS = {
    "UK": "#76d6ff",
    "USA": "#FF6346",
}

# Control confidence interval plotting
PLOT_CONFIDENCE_INTERVALS = False

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
            "mean": list(_get_mean_model(_contents["USA"], "USA")),
            "var": list(_get_var_model(_contents["USA"], "USA"))
        },
        "UK": {
            "mean": list(_get_mean_model(_contents["UK"], "UK")),
            "var": list(_get_var_model(_contents["UK"], "UK"))
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


def _calculate_confidence_bounds(x_values: np.ndarray, model_func: callable, confidence_intervals: dict, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Calculate confidence interval bounds for a model.
    
    :param x_values: X values to calculate bounds for
    :type x_values: np.ndarray
    :param model_func: The model function
    :type model_func: callable
    :param confidence_intervals: Dictionary of confidence intervals for parameters
    :type confidence_intervals: dict
    :param model_type: Type of model ('linear_mean', 'linear_var', 'power_law')
    :type model_type: str
    :return: Tuple of (lower_bounds, upper_bounds)
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if not confidence_intervals:
        # No confidence intervals available, return None bounds
        return None, None
    
    if model_type == "linear_mean":
        # For linear mean model: mx + b
        if "m" in confidence_intervals and "b" in confidence_intervals:
            m_lower, m_upper = confidence_intervals["m"]
            b_lower, b_upper = confidence_intervals["b"]
            
            # Calculate bounds for the linear function
            lower_bounds = m_lower * x_values + b_lower
            upper_bounds = m_upper * x_values + b_upper
            
            return lower_bounds, upper_bounds
    
    elif model_type == "linear_var":
        # For linear variance model: mx
        if "m" in confidence_intervals:
            m_lower, m_upper = confidence_intervals["m"]
            
            lower_bounds = m_lower * x_values
            upper_bounds = m_upper * x_values
            
            return lower_bounds, upper_bounds
    
    elif model_type == "power_law":
        # For power law model: d*x^alpha
        if "d" in confidence_intervals and "alpha" in confidence_intervals:
            d_lower, d_upper = confidence_intervals["d"]
            alpha_lower, alpha_upper = confidence_intervals["alpha"]
            
            # For power law, we need to be careful about the bounds
            # We'll use the parameter bounds to create approximate confidence bounds
            lower_bounds = d_lower * (x_values ** alpha_lower)
            upper_bounds = d_upper * (x_values ** alpha_upper)
            
            return lower_bounds, upper_bounds
    
    return None, None

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
            _x_shifted = _x + (8 if _country == "USA" else 0)
            
            # Plot the main model line
            ax[idx].plot(
                _x_shifted,
                safe_map(models[_country][case][0], _x),
                color=COLORS[_country],
                label=rf"{_country} ($R^2 = {round(models[_country][case][1], 2):.2f})$",
                linewidth=3,
                zorder=1,
            )
            
            # Plot confidence intervals if available and enabled
            if PLOT_CONFIDENCE_INTERVALS and len(models[_country][case]) > 2 and models[_country][case][2]:
                confidence_intervals = models[_country][case][2]
                
                # Determine model type for confidence interval calculation
                if case == "mean":
                    model_type = "linear_mean"
                else:  # case == "var"
                    # Check if it's linear or power law based on the model function
                    # We'll determine this by checking if the model has 'alpha' parameter
                    if "alpha" in confidence_intervals:
                        model_type = "power_law"
                    else:
                        model_type = "linear_var"
                
                lower_bounds, upper_bounds = _calculate_confidence_bounds(
                    _x, models[_country][case][0], confidence_intervals, model_type
                )
                
                if lower_bounds is not None and upper_bounds is not None:
                    # Plot confidence interval as filled area
                    # The x-axis shift is already applied to _x_shifted, so we use the original bounds
                    ax[idx].fill_between(
                        _x_shifted,
                        lower_bounds,
                        upper_bounds,
                        color=COLORS[_country],
                        alpha=0.2,
                        zorder=0,
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
            "share/figure.eps",
            dpi=400,
            bbox_inches="tight",
        )
        print("Figure saved as share/figure.eps")

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
            "tests/data/test4/S1.fasta",
            "tests/data/test4/S1.tsv",
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
            "tests/data/test4/S2.fasta",
            "tests/data/test4/S2.tsv",
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

def _get_mean_model(data: dict, kind: str) -> tuple[callable, float, dict]:
    """Extract mean model from data, handling both old and new formats.
    
    :param data: The regression results data dictionary
    :type data: dict
    :param kind: The dataset kind identifier (for error messages)
    :type kind: str
    :return: Tuple of (lambda function, r2 value, confidence intervals)
    :rtype: tuple[callable, float, dict]
    """
    # Try different possible key formats
    possible_keys = [
        "mean number of mutations model",  # New format (current)
        "mean number of mutations per 7D model",  # Old format
        "mean number of substitutions model",  # Alternative format
        "mean number of substitutions per 7D model"  # Alternative old format
    ]
    
    for mean_key in possible_keys:
        if mean_key in data:
            params = data[mean_key]["parameters"]
            r2 = data[mean_key]["r2"]
            confidence_intervals = data[mean_key].get("confidence_intervals", {})
            return lambda x: params["m"] * x + params["b"], r2, confidence_intervals
    
    raise KeyError(f"Could not find mean model in {kind} data. Available keys: {list(data.keys())}")


def _get_var_model(data: dict, kind: str) -> tuple[callable, float, dict]:
    """Extract variance model from data, handling both old and new formats.
    
    :param data: The regression results data dictionary
    :type data: dict
    :param kind: The dataset kind identifier (for error messages)
    :type kind: str
    :return: Tuple of (lambda function, r2 value, confidence intervals)
    :rtype: tuple[callable, float, dict]
    """
    # Try different possible key formats
    possible_keys = [
        "scaled var number of mutations model",  # New format (current)
        "scaled var number of mutations per 7D model",  # Old format
        "scaled var number of substitutions model",  # Alternative format
        "scaled var number of substitutions per 7D model"  # Alternative old format
    ]
    
    for var_key in possible_keys:
        if var_key in data:
            # Check if it has model_selection (new format)
            if "model_selection" in data[var_key]:
                # New format with model selection
                model_selection = data[var_key]["model_selection"]
                selected = model_selection["selected"]
                
                if selected == "linear" and "linear_model" in data[var_key]:
                    # Use linear model
                    linear_model = data[var_key]["linear_model"]
                    params = linear_model["parameters"]
                    r2 = linear_model["r2"]
                    confidence_intervals = linear_model.get("confidence_intervals", {})
                    return lambda x: params["m"] * x, r2, confidence_intervals
                elif selected == "power_law" and "power_law_model" in data[var_key]:
                    # Use power law model
                    power_law_model = data[var_key]["power_law_model"]
                    params = power_law_model["parameters"]
                    r2 = power_law_model["r2"]
                    confidence_intervals = power_law_model.get("confidence_intervals", {})
                    return lambda x: params["d"] * (x ** params["alpha"]), r2, confidence_intervals
            else:
                # Old format or new format without model selection - direct parameters
                params = data[var_key]["parameters"]
                r2 = data[var_key]["r2"]
                confidence_intervals = data[var_key].get("confidence_intervals", {})
                if "m" in params:
                    # Linear model: mx
                    return lambda x: params["m"] * x, r2, confidence_intervals
                elif "d" in params and "alpha" in params:
                    # Power law model: d*x^alpha
                    return lambda x: params["d"] * (x ** params["alpha"]), r2, confidence_intervals
    
    raise KeyError(f"Could not find variance model in {kind} data. Available keys: {list(data.keys())}")


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
            "mean": list(_get_mean_model(_contents["synt1"], "synt1")),
            "var": list(_get_var_model(_contents["synt1"], "synt1"))
        },
        "synt2": {
            "mean": list(_get_mean_model(_contents["synt2"], "synt2")),
            "var": list(_get_var_model(_contents["synt2"], "synt2"))
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
            
            # Plot confidence intervals if available and enabled
            if PLOT_CONFIDENCE_INTERVALS and len(models[_type.lower()][case]) > 2 and models[_type.lower()][case][2]:
                confidence_intervals = models[_type.lower()][case][2]
                
                # Determine model type for confidence interval calculation
                if case == "mean":
                    model_type = "linear_mean"
                else:  # case == "var"
                    # Check if it's linear or power law based on the model function
                    if "alpha" in confidence_intervals:
                        model_type = "power_law"
                    else:
                        model_type = "linear_var"
                
                lower_bounds, upper_bounds = _calculate_confidence_bounds(
                    _x, models[_type.lower()][case][0], confidence_intervals, model_type
                )
                
                if lower_bounds is not None and upper_bounds is not None:
                    # Plot confidence interval as filled area
                    ax[plot_idx].fill_between(
                        _x,
                        lower_bounds,
                        upper_bounds,
                        color="#76d6ff",
                        alpha=0.2,
                        zorder=0,
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
                    "m": _contents[k]["mean number of mutations model"]["parameters"]["m"],
                    "b": _contents[k]["mean number of mutations model"]["parameters"]["b"]
                },
                _contents[k]["mean number of mutations model"]["r2"],
                _contents[k]["mean number of mutations model"]["confidence_intervals"]
            ],
            "var": [
                {
                    "d": _contents[k]["scaled var number of mutations model"]["power_law_model"]["parameters"]["d"],
                    "alpha": _contents[k]["scaled var number of mutations model"]["power_law_model"]["parameters"]["alpha"]
                },
                _contents[k]["scaled var number of mutations model"]["power_law_model"]["r2"],
                _contents[k]["scaled var number of mutations model"]["power_law_model"]["confidence_intervals"]
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
        scaling = { # For the models to be comparable to the 7D model, we need to scale the x-axis by the square of the time window ratio
            "5D": (5/7)**2,
            "7D": 1,
            "10D": (10/7)**2,
            "14D": (14/7)**2,
        }
        for idx2, case in enumerate(("mean", "var")):

            if case == "mean":
                # Plot mean
                ax[idx2, idx].scatter(
                    df["dt_idx"],
                    df["mean number of mutations"],
                    color=COLORS["UK"],
                    edgecolor="k",
                    zorder=2,
                )
            
                _x = np.arange(-0.5, 51, 0.5)
                ax[idx2, idx].plot(
                    _x,
                    model["mean"][0]["m"]*_x + model["mean"][0]["b"],
                    color=COLORS["UK"],
                    label=rf"Mean ($R^2 = {round(model['mean'][1], 2):.2f})$",
                    linewidth=3,
                    zorder=1,
                )
                
                # Plot confidence intervals if available and enabled
                if PLOT_CONFIDENCE_INTERVALS and len(model["mean"]) > 2 and model["mean"][2]:
                    confidence_intervals = model["mean"][2]
                    
                    # For mean, it's always linear model
                    model_type = "linear_mean"
                    
                    # Calculate confidence bounds
                    lower_bounds, upper_bounds = _calculate_confidence_bounds(
                        _x, model["mean"][0], confidence_intervals, model_type
                    )
                    
                    if lower_bounds is not None and upper_bounds is not None:
                        # Plot confidence interval as filled area
                        ax[idx2, idx].fill_between(
                            _x,
                            lower_bounds,
                            upper_bounds,
                            color=COLORS["UK"],
                            alpha=0.2,
                            zorder=0,
                        )

            elif case == "var":
                # Plot variance 
                ax[idx2, idx].scatter(
                    df["dt_idx"],
                    df["var number of mutations"] - df["var number of mutations"].min(),
                    color=COLORS["UK"],
                    edgecolor="k",
                    zorder=2,
                )
                
                ax[idx2, idx].plot(
                    _x,
                    model["var"][0]["d"]*(_x*scaling[window])**model["var"][0]["alpha"],
                    color=COLORS["UK"],
                    label=rf"Var ($R^2 = {round(model['var'][1], 2):.2f})$",
                    linewidth=3,
                    zorder=1,
                )
                
                # Plot confidence intervals if available and enabled
                if PLOT_CONFIDENCE_INTERVALS and len(model["var"]) > 2 and model["var"][2]:
                    confidence_intervals = model["var"][2]
                    
                    # For variance, it's always power law model
                    model_type = "power_law"
                    
                    # Calculate confidence bounds for the scaled x values
                    scaled_x = _x * scaling[window]
                    lower_bounds, upper_bounds = _calculate_confidence_bounds(
                        scaled_x, model["var"][0], confidence_intervals, model_type
                    )
                    
                    if lower_bounds is not None and upper_bounds is not None:
                        # Plot confidence interval as filled area
                        ax[idx2, idx].fill_between(
                            _x,
                            lower_bounds,
                            upper_bounds,
                            color=COLORS["UK"],
                            alpha=0.2,
                            zorder=0,
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

def load_model_selection_results(directory: str) -> list[dict]:
    """Load all regression results from a directory for model selection analysis.
    
    This function recursively walks through the directory tree to find all
    regression results JSON files, supporting both flat and nested directory structures.
    
    Expected structure:
        directory/
        ├── {timestamp}/
        │   ├── {dataset_01}/
        │   │   └── *_out_regression_results.json
        │   ├── {dataset_02}/
        │   │   └── *_out_regression_results.json
        │   └── ...
        └── (or any nested structure)
    
    :param directory: Root directory to search for regression results
    :type directory: str
    :return: List of dictionaries containing model selection information
    :rtype: list[dict]
    """
    results = []
    
    # Walk through directory tree to find all regression results files
    # This works with any directory structure (flat or nested)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("out_regression_results.json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Extract the model selection info
                        model_selection = data.get("scaled var number of substitutions model", {}).get("model_selection", {})
                        results.append({
                            'file': file_path,
                            'selected_model': model_selection.get("selected", "unknown"),
                            'linear_AIC': model_selection.get("linear_AIC", None),
                            'power_law_AIC': model_selection.get("power_law_AIC", None),
                            'delta_AIC_linear': model_selection.get("delta_AIC_linear", None),
                            'delta_AIC_power_law': model_selection.get("delta_AIC_power_law", None),
                            'akaike_weight_linear': model_selection.get("akaike_weight_linear", None),
                            'akaike_weight_power_law': model_selection.get("akaike_weight_power_law", None)
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return results

def create_confusion_matrix_plot(export: bool = False, show: bool = True) -> None:
    """
    Create a confusion matrix plot for model selection accuracy analysis.
    
    Analyzes regression results from test5 synthetic datasets to assess
    model selection accuracy. Works with organized directory structure:
    
        tests/data/test5/
        ├── linear/output/{timestamp}/
        │   ├── linear_01/
        │   │   └── linear_01_out_regression_results.json
        │   ├── linear_02/
        │   └── ...
        └── powerlaw/output/{timestamp}/
            ├── powerlaw_01/
            └── ...
    
    Args:
        export: Whether to export the figure to share/confusion_matrix_heatmap.pdf
        show: Whether to display the figure
    """
    set_matplotlib_global_params()
    
    # Define paths - searches recursively through all subdirectories
    base_path = "tests/data/test5"
    linear_dir = os.path.join(base_path, "linear", "output")
    powerlaw_dir = os.path.join(base_path, "powerlaw", "output")
    
    print("Loading model selection results...")
    
    # Load results from both directories
    linear_results = load_model_selection_results(linear_dir)
    powerlaw_results = load_model_selection_results(powerlaw_dir)
    
    print(f"Loaded {len(linear_results)} linear results")
    print(f"Loaded {len(powerlaw_results)} powerlaw results")
    
    # Analyze results
    linear_success = sum(1 for r in linear_results if r['selected_model'] == 'linear')
    linear_failure = len(linear_results) - linear_success
    
    powerlaw_success = sum(1 for r in powerlaw_results if r['selected_model'] == 'power_law')
    powerlaw_failure = len(powerlaw_results) - powerlaw_success
    
    # Create confusion matrix
    # Format: [True Linear, False Linear], [False Powerlaw, True Powerlaw]
    # Transpose to flip axes: true model on x-axis, predicted model on y-axis
    confusion_matrix = np.array([
        [linear_success, linear_failure],     # Predicted Linear: [True Linear, False Linear]
        [powerlaw_failure, powerlaw_success]   # Predicted Powerlaw: [False Powerlaw, True Powerlaw]
    ])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create custom colormap from white to UK blue
    
    colors = ['white', '#76d6ff']
    n_bins = 30
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Linear', 'Power Law'])  # Actual model (x-axis)
    ax.set_yticklabels(['Linear', 'Power Law'])  # Predicted model (y-axis)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Actual Model', fontsize=16)
    ax.set_ylabel('Predicted Model', fontsize=16)
    ax.set_title('Model Selection Confusion Matrix', fontsize=18, fontweight='bold')
    
    # Calculate and display accuracy
    total_tests = len(linear_results) + len(powerlaw_results)
    total_successes = linear_success + powerlaw_success
    overall_accuracy = total_successes / total_tests if total_tests > 0 else 0
    
    # Add accuracy text
    ax.text(0.5, -0.25, f'Overall Accuracy: {overall_accuracy:.3f} ({total_successes}/{total_tests})',
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if export:
        fig.savefig(
            "share/confusion_matrix_heatmap.pdf",
            dpi=400,
            bbox_inches="tight",
        )
        print("Confusion matrix saved as share/confusion_matrix_heatmap.pdf")
    
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

    # # Size plot
    size_plot(df, export=export)

    # # Anomalous diffusion plot
    anomalous_diffusion_plot(export=export)

    # # Synthetic data plot
    synth_df = load_synthetic_data_df()
    synth_models = load_synthetic_data_models()
    synthetic_data_plot(synth_df, synth_models, export=export)

    # # UK time windows plot
    additional_uk_stats = load_additional_uk_stats()
    additional_uk_models = load_additional_uk_models()
    plot_uk_time_windows(additional_uk_stats, additional_uk_models, export=export)

    # # Confusion matrix plot
    create_confusion_matrix_plot(export=export)


if __name__ == "__main__":

    # Doing this way to not raise an out of bounds error when running the script without arguments
    _export_flag = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            _export_flag = True

    main(export=_export_flag)