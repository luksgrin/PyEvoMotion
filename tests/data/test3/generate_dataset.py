import pandas as pd
from io import StringIO

LOCAL_METADATA_PATH = "/Users/orion/Lucas_Goiriz_Beltran/rodrigolab/covid-diffusion/data/dataset_USA.parquet.gzip"
MASTER_METADATA_PATH = "/Volumes/Rodrigolab1/COVIDiffusion/masterMetadata.tsv"
MASTER_SEQUENCES_PATH = "/Volumes/Rodrigolab1/COVIDiffusion/raw_sequences/masterSequences.fasta"

def load_ref_metadata() -> pd.DataFrame:
    lines = ""
    with open(MASTER_METADATA_PATH) as f:
        for _ in range(2): lines += f.readline()
    df = pd.read_csv(
        StringIO(lines),
        sep="\t",
        usecols=[
            "date",
            "strain",
            "country",
            "segment",
            "sex",
            "pangolin_lineage",
        ]
    )
    df["variant"] = "Primal"
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.rename(columns={"date": "Date", "strain": "ID"}, inplace=True)
    return df

def treat_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            load_ref_metadata(),
            df.loc[
                df["variant"] == "Alpha",
                [
                    "Date",
                    "ID",
                    "country",
                    "segment",
                    "sex",
                    "pangolin_lineage",
                    "variant",
                ]
            ]
            .reset_index(drop=True),
        ],
        ignore_index=True,
    )

def load_metadata(metadata_path: str) -> pd.DataFrame:
    return treat_metadata(
        pd.read_parquet(metadata_path)
    )

def extract_sequences(sequences_path: str, metadata: pd.DataFrame) -> None:
    ids = metadata["ID"].tolist()

    with open("test3.fasta", "w") as output:
        _f_ptr = open(sequences_path)
        _id_ptr = ""
        _seq = ""

        while not(_id_ptr):
            line = _f_ptr.readline().strip()
            if line.startswith(">"):
                _id_ptr = line[1:]

        for line in _f_ptr:
            line = line.strip()
            if line.startswith(">"):
                if _id_ptr in ids:
                    output.write(f">{_id_ptr}\n{_seq}\n")
                _id_ptr = line[1:]
                _seq = ""
            else:
                _seq += line


def main(metadata_path: str, sequences_path: str) -> None:
    metadata = load_metadata(metadata_path)
    metadata.to_csv("test3.tsv", sep="\t")
    extract_sequences(sequences_path, metadata)

if __name__ == "__main__":
    main(LOCAL_METADATA_PATH, MASTER_SEQUENCES_PATH)