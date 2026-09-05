"""Build the fixed-seed CI subset of the UK/USA dataset.

Run from the repository root with the full dataset present in
tests/data/test3 (see tests/helpers/test_UK_USA_dataset_helpers.py):

    python -m tests.helpers.make_ci_subset

For each set it keeps the reference (first id of the manuscript sample) and a
date-stratified sample of N ids drawn, with a fixed seed, from
ids_sampled_for_figure.json, and writes tests/data/test3/ci/test3{UK,USA}.{tsv,fasta}
plus ids_sampled_for_ci.json. The ci/ directory is then archived as
test3-ci-subset-<version>.tar.gz, which is what the tests download from the
GitHub release `ci-data-<version>` when the files are absent.
"""
import json
import os
import tarfile

import pandas as pd
from PyEvoMotion import PyEvoMotionParser

from .test_UK_USA_dataset_helpers import (
    CI_DATA_VERSION,
    CI_DIR,
    ROOT,
    SAMPLE_SEED,
    equal_date_distribution_sample,
)

N_PER_SET = 1000


def build(n: int = N_PER_SET, seed: int = SAMPLE_SEED) -> str:
    os.makedirs(CI_DIR, exist_ok=True)
    with open(f"{ROOT}/ids_sampled_for_figure.json") as f:
        figure_ids = json.load(f)

    chosen = {}
    for set_ in ("UK", "USA"):
        df = pd.read_csv(f"{ROOT}/test3{set_}.tsv", sep="\t", index_col=0, parse_dates=["date"])
        ids = figure_ids[set_]
        ref_id = ids[0]
        pool = df[df["id"].isin(ids[1:])]
        sample = equal_date_distribution_sample(pool, "7D", pool["date"].min(), n, random_state=seed)
        sample = sample.sort_values("date", kind="stable")
        keep = [ref_id] + sample["id"].tolist()
        chosen[set_] = keep

        meta = pd.concat([df[df["id"] == ref_id], sample]).reset_index(drop=True)
        meta.to_csv(f"{CI_DIR}/test3{set_}.tsv", sep="\t")

        wanted = set(keep)
        print(f"{set_}: scanning FASTA for {len(wanted)} records ...", flush=True)
        with open(f"{CI_DIR}/test3{set_}.fasta", "w") as out:
            for rec in PyEvoMotionParser.read_fasta(f"{ROOT}/test3{set_}.fasta"):
                if rec.id in wanted:
                    out.write(rec.format())
                    wanted.discard(rec.id)
                    if not wanted:
                        break
        if wanted:
            raise RuntimeError(f"{set_}: {len(wanted)} sampled ids not found in the FASTA file: {sorted(wanted)[:5]}")

    with open(f"{ROOT}/ids_sampled_for_ci.json", "w") as f:
        json.dump(chosen, f)

    archive = f"test3-ci-subset-{CI_DATA_VERSION}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(os.listdir(CI_DIR)):
            tar.add(os.path.join(CI_DIR, name), arcname=name)
    print(f"wrote {archive}")
    return archive


if __name__ == "__main__":
    build()
