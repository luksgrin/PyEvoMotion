import os
import numpy as np
import PyEvoMotion
import pandas as pd
from datetime import datetime

def date_grouper(df: pd.DataFrame, DT: str, origin: str) -> pd.core.groupby.generic.DataFrameGroupBy:
    return PyEvoMotion.PyEvoMotion.date_grouper(df, DT, origin)

def equal_date_distribution_sample(df: pd.DataFrame, DT: str, origin: str, n: int) -> pd.DataFrame:
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

def main(date: str, DT: str) -> None:
    df = (
        pd.read_csv(
            "test3.tsv",
            sep="\t",
            index_col=0,
            parse_dates=["Date"],
        )
        .rename(columns={"ID": "id", "Date": "date"})
    )
    _origin = df["date"].min()

    (
        pd.concat([
            df.iloc[[0]],
            equal_date_distribution_sample(df.iloc[1:,:], DT, _origin, 9999)
        ])
        .reset_index(drop=True)
        .to_csv(f"output/{date}/sample_test3.tsv", sep="\t")
    )

if __name__ == "__main__":
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    _dt = "7D"
    os.makedirs(f"output/{date}", exist_ok=True)
    main(date, _dt)
    os.system(" ".join([
        f"poetry",
        "run",
        "PyEvoMotion",
        "test3.fasta",
        f"output/{date}/sample_test3.tsv",
        f"output/{date}/out",
        "-k", "total",
        "-n", "5",
        "-dt", _dt,
        "-dr", "2020-10-01..2021-08-01",
        "-ep",
        "-xj",
    ]))