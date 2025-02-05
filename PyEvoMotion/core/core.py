import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .base import PyEvoMotionBase
from .parser import PyEvoMotionParser


class PyEvoMotion(PyEvoMotionParser, PyEvoMotionBase):
    """
    Main class to analyze the data as intended by ``PyEvoMotion``. This class inherits from :class:`PyEvoMotionParser` and :class:`PyEvoMotionBase`. On construction, it calls :meth:`count_mutation_types`.

    :param input_fasta: The path to the input ``.fasta`` file.
    :type input_fasta: str
    :param input_meta: The path to the input metadata file. It has to have a column named ``date``. Accepts ``.csv`` and ``.tsv`` files. Default is ``None``.
    :type input_meta: str
    :param dt: The string datetime interval that will govern the grouping for the statistics. Default is 7 days (``7D``).
    :type dt: str
    :param filters: The filters to apply to the data. Default is ``None``.
    :type filters: dict[str, list[str] | str] | None
    :param positions: The positions to filter by. Default is ``None``.
    :type positions: tuple[int] | None
    :param date_range: The date range to filter by. Default is ``None``.
    :type date_range: tuple[str] | None

    Attributes:
    -----------
    data: ``pd.DataFrame``
        The parsed data from the input files.
    reference: ``str``
        The reference sequence.
    _MUTATION_TYPES: ``list[str]``
        The types of mutations that can be found in the data. Namely ``substitutions`` and ``indels``.
    """

    _MUTATION_TYPES = ["substitutions", "indels"]

    def __init__(self,
        input_fasta: str,
        input_meta: str,
        dt: str = "7D",
        filters: dict[str, list[str] | str] | None = None,
        positions: tuple[int] | None = None,
        date_range: tuple[str] | None = None
    ) -> None:
        """
        Initialize the class.

        It invokes the ``__init__()`` method of ``PyEvoMotionParser`` and ``count_mutation_types``.

        :param input_fasta: The path to the input fasta file.
        :type input_fasta: str
        :param input_meta: The path to the input metadata file.
        :type input_meta: str
        :param dt: The string datetime interval that will govern the grouping for the statistics. Default is 7 days.
        :type dt: str
        :param filters: The filters to apply to the data. Default is None.
        :type filters: dict[str, list[str] | str] | None
        :param positions: The positions to filter by. Default is None.
        :type positions: tuple[int] | None
        :param date_range: The date range to filter by. Default is None.
        :type date_range: tuple[str] | None
        """

        self.dt = dt

        # Parse the input fasta and metadata files
        super().__init__(
            input_fasta,
            input_meta,
            filters if filters else {},
            positions if positions else (0, 0),
            date_range
        )

        self._check_dataset_is_not_empty(
            self.data,
            "Perhaps there were no entries or the filters provided (if any) are too restrictive."
        )

        # Set the origin of the data
        self.origin = self.data["date"].min()
        if date_range:
            self.origin = min(self.origin, date_range[0]) if date_range[0] else self.origin

        self.count_mutation_types()

    @classmethod
    def plot_results(cls,
        stats: pd.DataFrame,
        regs: dict[str, dict[str, any]],
        data_xlabel_units: str
    ) -> None:
        """
        Plot the results of the analysis.

        :param stats: The statistics of the data. The first column has to be the date, the second column has to be the mean and the third column has to be the variance.
        :type stats: pd.DataFrame
        :param regs: The regression models.
        :type regs: dict[str, dict[str, any]]
        :param data_xlabel: The data ``xlabel`` units.
        :type data_xlabel: str
        """

        _, ax = plt.subplots(3, 1, figsize=(10, 10))

        # Mean
        _model = next(
            v
            for k,v in regs.items()
            if k.startswith("mean")
        )
        _mean_data = stats[stats.columns[1]]
        cls.plot_single_data_and_model(
            stats.index,
            _mean_data,
            _mean_data.name,
            _model["model"],
            r"$r^2$: " + f"{_model['r2']:.2f}",
            data_xlabel_units,
            ax[0]
        )

        # Variance
        _model = next(
            v
            for k,v in regs.items()
            if k.startswith("scaled var")
        )
        _variance_data = stats[stats.columns[2]]
        cls.plot_single_data_and_model(
            stats.index,
            _variance_data,
            _variance_data.name,
            _model["model"],
            r"$r^2$: " + f"{_model['r2']:.2f}",
            data_xlabel_units,
            ax[1]
        )

        # Dispersion index
        cls.plot_single_data_and_model(
            stats.index,
            _mean_data/_variance_data,
            f"dispersion index of {' '.join(_mean_data.name.split()[1:])}",
            lambda x: [1]*len(x),
            "Poissonian regime",
            data_xlabel_units,
            ax[2],
            line_linestyle="--",
            line_color="black"
        )

        plt.tight_layout()
        plt.show()

    @classmethod
    def export_plot_results(cls,
        stats: pd.DataFrame,
        regs: dict[str, dict[str, any]],
        data_xlabel_units: str,
        output_ptr: str | None = None
    ) -> None:
        """
        Export the results of the analysis to a ``.pdf`` file.

        :param stats: The statistics of the data.
        :type stats: pd.DataFrame
        :param regs: The regression models.
        :type regs: dict[str, dict[str, any]]
        :param data_xlabel_units: The data ``xlabel`` units for the plot.
        :type data_xlabel_units: str
        :param output_ptr: The output ``.pdf`` file. If ``None``, it will create a new ``.pdf`` file.
        :type output: str
        """

        pdf = output_ptr if output_ptr else PdfPages("output_plots.pdf")
            
        plt.figure()
        # Mean
        _model = next(
            v
            for k,v in regs.items()
            if k.startswith("mean")
        )
        _mean_data = stats[stats.columns[1]]
        cls.plot_single_data_and_model(
            stats.index,
            _mean_data,
            _mean_data.name,
            _model["model"],
            r"$r^2$: " + f"{_model['r2']:.2f}",
            data_xlabel_units,
            plt.gca()
        )

        plt.title(_mean_data.name)
        pdf.savefig()
        plt.close()

        plt.figure()
        # Variance
        _model = next(
            v
            for k,v in regs.items()
            if k.startswith("scaled var")
        )
        _variance_data = stats[stats.columns[2]]
        cls.plot_single_data_and_model(
            stats.index,
            _variance_data,
            _variance_data.name,
            lambda x: _model["model"](x) + _variance_data.min(), # Adjust the model to the original variance
            r"$r^2$: " + f"{_model['r2']:.2f}",
            data_xlabel_units,
            plt.gca()
        )

        plt.title(_variance_data.name)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure()
        # Dispersion index
        _name = " ".join(_mean_data.name.split()[1:])
        cls.plot_single_data_and_model(
            stats.index,
            _mean_data/_variance_data,
            f"dispersion index of {_name}",
            lambda x: [1]*len(x),
            "Poissonian regime",
            data_xlabel_units,
            plt.gca(),
            line_linestyle="--",
            line_color="black"
        )

        plt.title(f"Dispersion index of {_name}")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    def count_mutation_types(self) -> None:
        """
        Count the number of substitutions, insertions and deletions in the data.
        
        It updates the ``data`` attribute by adding the columns ``number of substitutions``, ``number of indels`` and ``number of mutations``.
        """

        for _type in self._MUTATION_TYPES + ["insertions", "deletions"]:
            self.data[f"number of {_type}"] = self.data["mutation instructions"].apply(
                lambda x: self.count_prefixes(_type[0], x)
            )

        # Set indels together just in case
        self.data["number of indels"] = (
            self.data["number of insertions"]
            + self.data["number of deletions"]
        )

        self.data["number of mutations"] = self.data["mutation instructions"].apply(len)

    def get_lengths(self) -> pd.Series:
        """
        Get the lengths of the sequences in the dataset.

        :return: The lengths of the sequences.
        :rtype: ``pd.Series``
        """

        return (
            self.data["mutation instructions"].apply(
                lambda x: sum(map(
                    lambda y: self.mutation_length_modification(y),
                    x
                ))
            )
            + len(self.reference)
        )
    
    def length_filter(self, length: int, how: str="gt") -> None:
        """
        Filter the data by sequence length.

        It updates the ``data`` attribute by filtering the data by the sequence length.

        :param length: The length to filter by.
        :type length: int
        :param how: The filter condition. It can be ``gt`` (greater than), ``lt`` (less than) or ``eq`` (equal to).
        :type how: str
        """

        if how == "gt":
            self.data[self.get_lengths() > length]
        elif how == "lt":
            self.data[self.get_lengths() < length]
        elif how == "eq":
            self.data[self.get_lengths() == length]
        else:
            raise ValueError(f"Filter \"{how}\" not recognized")
        
        self.data.reset_index(drop=True, inplace=True)
        
    def n_filter(self, threshold: float | int = 0.01, how: str = "lt") -> None:
        """
        Filter the data by the number of ``N`` in the sequence.

        It updates the ``data`` attribute by filtering the data by the number of ``N`` in the sequence.
        
        :param threshold: The threshold to filter by. Must be between 0 and 1. Default is 0.01.
        :type threshold: float | int
        :param how: The filter condition. It can be ``gt`` (greater than), ``lt`` (less than) or ``eq`` (equal to).
        :type how: str
        """

        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")

        N_freq = self.data["N count"]/self.get_lengths()

        if how == "gt":
            self.data[N_freq > threshold]
        elif how == "lt":
            self.data[N_freq < threshold]
        elif how == "eq":
            self.data[N_freq == threshold]
        else:
            raise ValueError(f"Filter \"{how}\" not recognized")
        
        self.data.reset_index(drop=True, inplace=True)
        
    @classmethod
    def _mutation_type_switch(cls, mutation_kind: str) -> list[str]:
        """
        Switch the mutation kind to the corresponding list of mutation types.

        This is used to subset the analysis to the desired mutation kind.

        :param mutation_kind: the kind of mutation to compute the statistics for. Has to be one of "all", "total", "substitutions" or "indels".
        :type mutation_kind: str
        :return: the list of mutation types.
        :rtype: list[str]
        """

        cases = {
            "all": cls._MUTATION_TYPES + ["mutations"],
            "total": ["mutations"],
            "substitutions": [cls._MUTATION_TYPES[0]],
            "indels": [cls._MUTATION_TYPES[1]]
        }

        choice = cases.get(mutation_kind, None)

        if choice is None:
            raise ValueError(f'Mutation kind \"{mutation_kind}\" not recognized. It has to be one of {", ".join(cases.keys())}')

        return choice

    def compute_stats(self,
        DT: str,
        origin: str,
        n_threshold: int | None = None,
        mutation_kind: str = "all"
    ) -> pd.DataFrame:
        """
        Compute the length, mean and variance of the data.

        It computes the mean and variance of the data for the specified mutation kind (or kinds) in the specified datetime interval and origin.

        :param DT: The string datetime interval that will govern the grouping.
        :type DT: str
        :param origin: The string datetime that will be the origin of the grouping.
        :type origin: str
        :param n_threshold: Minimum number of sequences required in a time interval to compute statistics.
        :type n_threshold: int | None
        :param mutation_kind: The kind of mutation to compute the statistics for. Has to be one of ``all``, ``total``, ``substitutions``, ``insertions``, ``deletions`` or ``indels``. Default is ``all``.
        :return: The statistics of the data.
        :rtype: ``pd.DataFrame``
        """

        grouped = self.date_grouper(self.data, DT, origin)

        # Only keep weeks where the number of observations is greater than the threshold
        if n_threshold:
            grouped = self.date_grouper(
                grouped.filter(lambda x: len(x) >= n_threshold),
                DT,
                origin
            )

        levels = [
            f"number of {x}"
            for x in self._mutation_type_switch(mutation_kind)
        ]

        return pd.concat(
            (
                pd.DataFrame(self._invoke_method(grouped[levels], method))
                .rename(
                    columns=lambda col: f"{method} {col}"
                    if method != "size" else "size"
                )
                for method in ("mean", "var", "size")
            ),
            axis=1
        ).reset_index(level=['date'])

    def analysis(self,
        length: int,
        n_threshold: int | None = None,
        show: bool = False,
        mutation_kind: str = "all",
        export_plots_filename: str | None = None
    ) -> tuple[pd.DataFrame, dict[str,dict[str,any]]]:
        """
        Perform the global analysis of the data.

        It computes the statistics and the regression models for the mean and variance of the data.
        
        :param length: The length to filter by.
        :type length: int
        :param n_threshold: Minimum number of sequences required in a time interval to compute statistics.
        :param show: Whether to show the plots or not. Default is False.
        :type show: bool
        :param mutation_kind: The kind of mutation to compute the statistics for. Has to be one of ``all``, ``total``, ``substitutions`` or ``indels``. Default is ``all``.
        :type mutation_kind: str
        :param export_plots: Filename to export the plots. Default is None and does not export the plots.
        :type export_plots: str | None
        :return: The statistics and the regression models.
        :rtype: ``tuple[pd.DataFrame, dict[str, dict[str, any]]]``
        """

        # Apply filters
        self.n_filter()
        self.length_filter(length=length)

        # Compute the statistics for the specified mutation kinds
        stats = self.compute_stats(
            self.dt,
            self.origin,
            n_threshold,
            mutation_kind
        )


        regs = {}
        # For each column in the statistics (except the date and the size), compute the corresponding regression model
        for col in stats.columns[1:-1]:
            if col.startswith("mean"):
                _single_regression = {
                    f"{col} per {self.dt} model": self.linear_regression(
                        *self._remove_nan(
                            stats.index, # Regression is given by the index, so in time, it is the same as multiplying by dt days
                            stats[col]
                        )
                    )
                }
            elif col.startswith("var"):
                _single_regression = self.adjust_model(
                    stats.index,
                    stats[col] - stats[col].min(),
                    name=f"scaled {col} per {self.dt} model"
                )
            # Save the regression model
            regs.update(_single_regression)

        # Sets of mutation types used in the analysis
        _sets = sorted({
            " ".join(x.split()[1:])
            for x in stats.columns[1:-1]
        })

        # Plot the results
        if show:
            # For each set of mutation types
            for _type in _sets:
                self.plot_results(
                    stats[["date", f"mean {_type}", f"var {_type}"]],
                    {
                        k: v
                        for k, v in regs.items()
                        if k in (
                            f"mean {_type} per {self.dt} model",
                            f"scaled var {_type} per {self.dt} model"
                        )
                    },
                    f"in steps of {self.dt} since {self.origin}"
                )
        # Export the plots
        if export_plots_filename:
            # Open pdf file pointer
            pdf = PdfPages(f"{export_plots_filename}.pdf")
            # For each set of mutation types save the plots
            for _type in _sets:
                self.export_plot_results(
                    stats[["date", f"mean {_type}", f"var {_type}"]],
                    {
                        k: v
                        for k, v in regs.items()
                        if k in (
                            f"mean {_type} per {self.dt} model",
                            f"scaled var {_type} per {self.dt} model"
                        )
                    },
                    f"in steps of {self.dt} since {self.origin}",
                    pdf
                )
            # Close pdf file pointer
            pdf.close()

        return stats, regs
