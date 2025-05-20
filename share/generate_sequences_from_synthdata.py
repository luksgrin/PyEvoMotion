#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                           IMPORTS                          #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                          CONSTANTS                         #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

WUHAN_REF_PATH = "tests/data/test3/test3UK.fasta"
DATE_MAP = dict(zip(
    range(41),
    pd.date_range(start="2020-01-01", periods=41, freq="7D")
))
NUCLEOTIDES = {"A", "C", "G", "T"}

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                          FUNCTIONS                         #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

def load_synthdata(path: str) -> pd.DataFrame:
    """Loads the synthdata from the given path.
    """

    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [f"seq{k}" for k in range(len(df.columns))]

    return df

def get_wuhan_ref() -> SeqRecord:
    """Returns the Wuhan reference sequence.
    """
    # We know the Wuhan sequence is the first one in the file
    with open(WUHAN_REF_PATH, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            return record

def create_synthetic_sequence(mut_num: int, wuhan_ref: SeqRecord) -> str:
    """Creates a synthetic sequence from the Wuhan reference sequence.
    """
    _seq = list(wuhan_ref.seq)
    mut_positions = np.random.choice(range(len(_seq)), mut_num, replace=False)
    nucleotides = [
        np.random.choice(list(NUCLEOTIDES - {_seq[i]}))
        for i in mut_positions
    ]
    for pos, nuc in zip(mut_positions, nucleotides):
        _seq[pos] = nuc
    return "".join(_seq)


def create_synthetic_sequences(output_path: str, synthdata: pd.DataFrame, wuhan_ref: SeqRecord) -> None:
    """Creates the synthetic dataset from the synthdata reference and the Wuhan reference sequence.
    """
    with open(f"{output_path}.fasta", "w") as f, open(f"{output_path}.tsv", "w") as g:
        f.write(f">{wuhan_ref.id}\n{wuhan_ref.seq}\n")
        g.write(f"id\tdate\n{wuhan_ref.id}\t2019-12-29\n")
        for row_idx, row in synthdata.iterrows():
            for col_name, col in row.items():
                # Fetch the row number, the column name and the entry in the synthdata
                f.write(f">{col_name}_{row_idx}\n{create_synthetic_sequence(col, wuhan_ref)}\n")
                g.write(f"{col_name}_{row_idx}\t{DATE_MAP[row_idx].strftime('%Y-%m-%d')}\n")

#´:°•.°+.*•´.*:˚.°*.˚•´.°:°•.°•.*•´.*:˚.°*.˚•´.°:°•.°+.*•´.*:#
#                           MAIN                             #
#.•°:°.´+˚.*°.˚:*.´•*.+°.•°:´*.´•*.•°.•°:°.´:•˚°.*°.˚:*.´+°.•#

def main():
    WUHAN_REF = get_wuhan_ref()

    for name in ("synthdata1", "synthdata2"):
        create_synthetic_sequences(
            f"{name}",
            load_synthdata(f"tests/data/test4/{name}.txt"),
            WUHAN_REF
        )
    print("Synthetic datasets created successfully.")

if __name__ == "__main__":
    main()
