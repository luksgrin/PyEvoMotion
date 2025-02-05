from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

import numpy as np
import pandas as pd
from io import StringIO
from itertools import groupby
from datetime import datetime
from operator import itemgetter
from subprocess import Popen, PIPE


class PyEvoMotionParser():
    """
    This class is responsible for parsing the input fasta and metadata files. It is inherited by the :class:`PyEvoMotion` class.

    :param input_fasta: The path to the input ``.fasta`` file.
    :type input_fasta: str
    :param input_meta: The path to the input metadata file. The metadata file must contain a ``date`` column and can be in either ``.csv`` or ``.tsv`` format.
    :type input_meta: str
    :param filters: The filters to be applied to the data. The keys are the column names and the values are the values to be filtered.
    :type filters: dict[str, list[str] | str]
    :param positions: The start and end positions to filter the data by.
    :type positions: tuple[int]
    :param date_range: The start and end dates to filter the data by. If ``None``, the date range is not filtered.
    :type date_range: tuple[datetime] | None
    
    On construction, it invokes the following methods:
        - :meth:`parse_metadata`: Parses the metadata file.
        - :meth:`parse_sequence_by_id`: Parses the sequence with the ID of the first entry in the metadata file.
        - :meth:`filter_columns`: Filters the metadata based on the input filters.
        - :meth:`filter_by_daterange`: Filters the metadata based on the input date range.
        - :meth:`parse_data`: Parses the input fasta file and appends the mutations that differ between the reference sequence and the input sequences to the metadata.
        - :meth:`filter_by_position`: Filters the metadata based on the input start and end positions.

    Attributes:
    -----------
    data: ``pd.DataFrame``
        DataFrame containing metadata.
    reference: ``SeqRecord``
        The reference sequence parsed from the fasta file.
    """

    def __init__(self,
        input_fasta: str,
        input_meta: str,
        filters: dict[str, list[str] | str],
        positions: tuple[int],
        date_range: tuple[datetime] | None
    ) -> None:
        """
        Initializes the class with input FASTA and metadata files.

        :param input_fasta: The path to the input FASTA file.
        :type input_fasta: str
        :param input_meta: The path to the input metadata file.
        :type input_meta: str
        :param filters: The filters to be applied to the data. The keys are the column names and the values are the values to be filtered.
        :type filters: dict[str, list[str] | str]
        :param positions: The start and end positions to filter the data by.
        :type positions: tuple[int]
        :param date_range: The start and end dates to filter the data by. If None, the date range is not filtered.
        :type date_range: tuple[datetime] | None
        """

        self.data = self.parse_metadata(input_meta)
        self.reference = self.parse_sequence_by_id(
            input_fasta,
            self.data.iloc[0]["id"]
        )
        # Implicitly filters the data
        self.filter_columns(filters)
        if date_range:
            self.filter_by_daterange(*date_range)
        # Appends the mutations that differ between the reference sequence and the input sequences to the data
        self.parse_data(input_fasta, self.data["id"])
        # Applies the position filter if provided
        self.filter_by_position(*positions)

    def parse_data(self, input_fasta: str, selection: pd.Series) -> None:
        """
        Parse the input fasta file and store the resulting data in the ``data`` attribute.

        :param input_fasta: The path to the input ``.fasta`` file.
        :type input_fasta: str
        :param selection: The selection of sequence ids to be parsed.
        :type selection: pd.Series
        """

        self.data = (
            self.data
            .merge(
                self.get_differing_mutations(input_fasta, selection),
                on="id",
                how="left"
            )
            .reset_index(drop=True)
        )

    def filter_by_daterange(self, start: datetime, end: datetime) -> None:
        """
        Filter the data based on a date range.

        The data is filtered to only include entries with dates between the start and end dates. This method modifies the ``data`` attribute in place.

        :param start: The start date.
        :type start: datetime
        :param end: The end date.
        :type end: datetime
        :raises ValueError: If the start date is greater than the end date.
        """

        start = (
            max(self.data["date"].min(), start) if start
            else self.data["date"].min()
        )
        end = (
            min(self.data["date"].max(), end)
            if end else self.data["date"].max()
        )

        if start > end:
            raise ValueError("Start date must be smaller than end date")    

        self.data = self.data[
            (self.data["date"] >= start) & (self.data["date"] <= end)
        ]

    def filter_by_position(self, start: int, end: int) -> None:
        """
        Filter the data based on some start and end positions in the reference sequence.

        *Note that the positions are 1-indexed, and that the end position is inclusive.*

        :param start: The start position index.
        :type start: int
        :param end: The end position index.
        :type end: int
        :raises ValueError: If the start position is greater than the end position., or if the start position is greater than the length of the reference sequence.
        """

        start = max(1, start)  # Ensure start is at least 1
        end = end if end > 0 else len(self.reference.seq) + 1  # Set end if not provided        

        if start >= end:
            raise ValueError("Start position must be smaller than end position")
        elif start > len(self.reference.seq):
            raise ValueError("Start position is out of range")

        self.data["mutation instructions"] = self.data["mutation instructions"].apply(
            lambda x: [
                mod
                for mod in x
                if start - 1 < int(mod.split("_")[1]) < end
            ]
        )
        self.data = self.data[
            self.data["mutation instructions"].apply(len) > 0
        ]

    def filter_columns(self, filters: dict[str, list[str] | str]) -> None:
        """
        Filter the data based on the input filters provided.

        :param filters: The filters to be applied to the data. The keys are the column names and the values are the values to be filtered from the provided metadata.
        :type filters: dict[str, list[str] | str]
        """

        # Only keep filters that are columns in the data
        _filters = {
            k: v
            for k,v in filters.items()
            if k in self.data.columns
        }

        for col, vals in _filters.items():
            if isinstance(vals, str):
                vals = [vals]
            regex_pattern = "|".join(
                val.replace('*', '.*')
                for val in vals
            )
            self.data = self.data[
                self.data[col]
                .str.contains(
                    regex_pattern,
                    regex=True
                )
            ]

    @staticmethod
    def _get_consecutives(data: list[int]) -> list[list[int]]:
        """
        Groups list of ordered integers into list of groups of integers
        
        :param data: a list of ordered integers.
        :type data: list[int]
        :return idxs: a list of lists of consecutive integers.
        :rtype: list[list[int]]
        """
        idxs = []

        for _, g in groupby(
            enumerate(data),
            lambda x: x[0] - x[1]
        ):
            idxs.append(list(map(itemgetter(1), g)))

        return idxs

    @staticmethod
    def _column_decision(col:np.array) -> int:
        """
        Classifies bases in array column

        :param col: column with two rows, containing one of these symbols: A, G, T, C, N, -
        :type col: np.array
        :return: returns an integer indicating if match (0), mismatch (1), insertion (2) or deletion (3).
        :rtype: int
        """

        # If there's a match, return 0. In the case that there's an insertion of an N, we also return 0 as it's probably a sequencing error
        if ((col[0] == col[1]) or ("N" in col)): return 0
        
        # If there's an insertion, return 2
        elif col[0] == "-": return 2

        # If there's a deletion, return 3
        elif col[1] == "-": return 3

        # Else, it has to be a mismatch. Return 1
        else: return 1

    @classmethod
    def create_modifs(cls, alignment: MultipleSeqAlignment) -> list[str]:
        """
        Creates a guide on how to modify the root sequence to get the appropriate mutant sequence.

        :param alignment: ``Bio.Align.MultipleSeqAlignment`` object containing the alignment.
        :type alignment: MultipleSeqAlignment
        :return mods: List of modifications encoded as strings that have the format ``<type>_<position>_<bases>``, where ``<type>`` is one of ``i``, ``d`` and ``s`` (insertion, deletion and substitution), ``<position>`` is the position in the sequence where the modification should be made, and ``<bases>`` are the bases to be inserted, deleted or substituted.
        :rtype: ``list[str]``
        """

        # Turn alignment into np.array
        coding = np.array([
            list(alignment[0].upper()),
            list(alignment[1].upper())
        ])

        # Get classification
        clsMut = np.apply_along_axis(cls._column_decision, 0, coding)

        # Encode substitutions
        subst = list(map(
            lambda x: f"s_{x + 1}_{coding[1, x]}",
            list(np.where(clsMut == 1)[0])
        ))

        # Encode insertions
        insertions = list(map(
            lambda x: f"i_{x[0]}_{''.join(coding[1, x])}",
            cls._get_consecutives(
                list(np.where(clsMut == 2)[0])
            )
        ))

        # Encode deletions
        deletions = list(map(
            lambda x: f"d_{x[0]}_{''.join(coding[0, x])}",
            cls._get_consecutives(
                list(np.where(clsMut == 3)[0])
            )
        ))

        # Blend modifications and order them
        mods = sorted(
            insertions + deletions + subst,
            key=lambda x: int(x.split("_")[1])
        )

        reindex = [
            (mods.index(el),len(el.split("_")[-1]))
            for el in mods
            if el.startswith("i")
        ]

        for idx,v in reindex:
            mods[idx + 1:] = list(map(
                lambda x: "_".join((
                    x.split("_")[0],
                    str(int(x.split("_")[1]) - v),
                    x.split("_")[-1]
                )),
                mods[idx + 1:]
            ))

        return mods

    def get_differing_mutations(self, input_fasta: str, selection: pd.Series) -> pd.DataFrame:
        """
        Return the mutations that differ between the reference sequence and the input sequence.

        Also, for the sake of sequence selection, it outputs the number of ``N`` found in the sequence.

        :param input_fasta: The path to the input ``.fasta`` file.
        :type input_fasta: str
        :param selection: The selection of sequence ids to be compared with the reference sequence.
        :type selection: pd.Series
        :return: The mutations that differ between the reference sequence and the input sequence. It contains the columns ``id``, ``mutation instructions`` and ``N count`` (the number of ``N`` in the sequence).
        :rtype: ``pd.DataFrame``
        """

        aligments = {}

        with open(input_fasta) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if not(record.id in selection.values):
                    continue
                alignment = self.generate_alignment(
                    self.reference,
                    record
                )
                aligments[record.id] = (
                    self.create_modifs(alignment),
                    alignment[1].seq.count("n") # In alignment fasta files, bases are lowercase
                )

        return pd.DataFrame(
            [(k, v1, v2) for k, (v1, v2) in aligments.items()],
            columns=["id", "mutation instructions", "N count"]
        )

    @classmethod
    def generate_alignment(cls, seq1: str, seq2: str) -> MultipleSeqAlignment:
        """
        Generate a multiple sequence alignment of the input sequences using ``MAFFT``.

        :param seq1: The first sequence to be aligned.
        :type seq1: str
        :param seq2: The second sequence to be aligned.
        :type seq2: str
        :return: The aligned sequences.
        :rtype: ``MultipleSeqAlignment``
        """

        id_1 = seq1.id
        id_2 = seq2.id

        if seq1.id == seq2.id:
            id_1 += "_ref"

        return AlignIO.read(
            StringIO(cls._run_mafft({
                id_1: seq1.seq,
                id_2: seq2.seq
            })),
            "fasta"
        )
 
    @staticmethod
    def parse_sequence_by_id(input_fasta: str, _id: str) -> SeqRecord | None:
        """
        Parse the input ``.fasta`` file and return the ``Bio.SeqRecord`` with the given ``_id``. Returns ``None`` if the ``_id`` is not found.

        :param input_fasta: The path to the input ``.fasta`` file.
        :type input_fasta: str
        :param _id: The ID of the sequence to be returned.
        :type _id: str
        :return: The sequence record with the given ``_id``. ``None`` if the ``_id`` is not found.
        :rtype: SeqRecord | None
        """

        with open(input_fasta) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if record.id == _id:
                    return record
        return None

    @staticmethod
    def _run_mafft(seqs_dict: dict[str,str], outformat: str = "fasta") -> str:
        """
        This function runs the MAFFT multiple sequence alignment tool on the input sequences.

        It raises an exception if the return code is not 0 (i.e. there was an error running MAFFT).

        :param seqs_dict: A dictionary containing the sequences to be aligned. The keys are the sequence names and the values are the sequences.
        :type seqs_dict: dict[str,str]
        :param outformat: The output format of the alignment. Default is fasta.
        :type outformat: str
        :return: The aligned sequences as parsed from stdout. If the output format is clustal, it returns the alignment in clustal format; otherwise, it returns the alignment in fasta format.
        :rtype: str
        """

        cmd = ["mafft"]
        template_format = ">{}\n{}\n"

        if outformat == "clustal":
            cmd.extend(["--clustalout", "-"])

        elif outformat != "fasta":
            print(f"Unknown output format: {outformat}. Defaulting to fasta.")

        cmd.append("-")

        input_data = bytes(
            "".join(
                template_format.format(name, seq)
                for name, seq in seqs_dict.items()
            ),
            "utf-8"
        )

        ps = Popen(
            cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            shell=False
        )
        ps.stdin.write(input_data)
        ps.stdin.close()

        err = ps.stderr.read().decode("utf-8")
        out = ps.stdout.read().decode("utf-8")

        if (ps.returncode != 0) and not(ps.returncode is None):
            raise Exception(
                f"Error running MAFFT:\nStdout:\n{out}\n\nStderr:\n{err}\nReturn code: {ps.returncode}"
            )

        return out

    @staticmethod
    def parse_metadata(input_meta: str) -> pd.DataFrame:
        """
        Parse the metadata file into a ``pandas.DataFrame``.

        :param input_meta: The path to the metadata file, in either ``.csv`` or ``.tsv`` format.
        :type input_meta: str
        :return: The metadata as a ``pd.DataFrame``. It must contain a ``date`` column. Other columns are optional.
        :rtype: ``pd.DataFrame``
        :raises ValueError: If the metadata file does not contain a ``date`` column.
        """
        
        seps = {
            "csv": ",",
            "tsv": "\t"
        }

        try:
            if input_meta.endswith(".csv"):
                df = pd.read_csv(input_meta, sep=seps["csv"])
            elif input_meta.endswith(".tsv"):
                df = pd.read_csv(input_meta, sep=seps["tsv"])
        except Exception as e:
            print(f"Error reading metadata file: {e}")
            return
        
        if not "date" in df.columns:
            raise ValueError("Metadata file must contain a \"date\" column")
        
        df["date"] = pd.to_datetime(df["date"])

        return df.sort_values(by="date")
        
