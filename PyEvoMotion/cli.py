"""
Command line interface for :class:`PyEvoMotion`.

It parses the arguments from the command line and runs the analysis with the specified parameters.

This module is not meant to be inherited from, but to be used as a standalone script in the command line.
"""

import json
import argparse
from datetime import datetime

from .core.core import PyEvoMotion

PACKAGE_DESCRIPTION = "PyEvoMotion"
BANNER = r"""
Welcome to Rodrigolab's
 _____       ______          __  __       _   _             
|  __ \     |  ____|        |  \/  |     | | (_)            
| |__) |   _| |____   _____ | \  / | ___ | |_ _  ___  _ __  
|  ___/ | | |  __\ \ / / _ \| |\/| |/ _ \| __| |/ _ \| '_ \ 
| |   | |_| | |___\ V / (_) | |  | | (_) | |_| | (_) | | | |
|_|    \__, |______\_/ \___/|_|  |_|\___/ \__|_|\___/|_| |_|
        __/ |                                               
       |___/                                                
"""

class _ArgumentParserWithHelpOnError(argparse.ArgumentParser):
    """
    Custom ArgumentParser that prints the help message when an error occurs.
    """

    def error(self, message: str) -> None:
        """
        Print the help message and the error message.

        :param message: the error message to print.
        :type message: str
        """
        self.print_help()
        print(f"\nError: {message}\n")
        super().exit(2)

class _ParseFilter(argparse.Action):
    """
    Custom action to parse the filters from the command line.

    The filters are passed as key-value pairs, where the key is followed by multiple values, specified in square brackets.
    """
    def __call__(self, _: argparse.ArgumentParser, namespace: argparse.Namespace, values: list[str], option_string: str | None = None) -> None:
        """
        Call the action to parse the filters.

        :param _: the parser.
        :type _: argparse.ArgumentParser
        :param namespace: the namespace to store the parsed filters.
        :type namespace: argparse.Namespace
        :param values: the values to parse.
        :type values: list[str]
        :param option_string: the option string.
        :type option_string: str
        :raises ValueError: if the values are not in the correct format.
        """
        
        setattr(namespace, self.dest, self.parse_filters(values))

    @staticmethod
    def parse_filters(values: list[str] | None) -> dict[str, str | list[str]] | None:
        """
        Parse the filters from the values.

        :param values: the values to parse.
        :type values: list[str] | None
        :return: the parsed filters as a dictionary.
        :rtype: dict[str, str | list[str]] | None
        """
    
        if values is None: return None

        # Create an iterator to process values one by one
        cleaned_values = []
        buffer = []
        inside_brackets = False

        # Loop through the input values and handle brackets
        for value in values:
            if value.startswith('[') and value.endswith(']'):  # Single value inside brackets
                cleaned_values.append(value[1:-1])
            if value.startswith('['):  # Start of a bracketed group
                inside_brackets = True
                buffer.append(value[1:])  # Strip the '['
            elif value.endswith(']'):  # End of a bracketed group
                buffer.append(value[:-1])  # Strip the ']'
                cleaned_values.append(buffer)
                buffer = []
                inside_brackets = False
            elif inside_brackets:  # Values inside the brackets
                buffer.append(value)
            else:  # Regular values outside of brackets
                cleaned_values.append(value)

        return dict(zip(
            cleaned_values[::2],
            cleaned_values[1::2]
        ))

class _ParseGenomePosition(argparse.Action):
    """
    Custom action to parse the genome positions from the command line.

    The genome positions are passed as a string with two dots separating the start and end positions. Open start or end positions are allowed by omitting the first or last position, respectively.
    """
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str | None = None):
        """
        Call the action to parse the genome positions.

        :param parser: the parser.
        :type parser: argparse.ArgumentParser
        :param namespace: the namespace to store the parsed genome positions.
        :type namespace: argparse.Namespace
        :param values: the values to parse.
        :type values: str
        :param option_string: the option string.
        :type option_string: str
        :raises ValueError: if the values are not in the correct format.
        """
        

        
        setattr(namespace, self.dest, self.parse_genome_position(parser, values))

    @staticmethod
    def parse_genome_position(parser: argparse.ArgumentParser, values: str | None) -> tuple[int, int] | None:
        """
        Parse the genome positions from the values.

        :param parser: the parser.
        :type parser: argparse.ArgumentParser
        :param values: the values to parse.
        :type values: str | None
        :return: the parsed genome positions.
        :rtype: tuple[int, int] | None
        :raises ValueError: if the values are not in the correct format.
        """

        if values is None: return None

        if not(".." in values):
            parser.error("The genome positions must be separated by two dots. Example: 1..1000")

        _split = values.split("..")

        positions = []
        for el in _split:
            if not el.isdigit() and el != "":
                parser.error("The genome positions must be positive integers")
            positions.append(0 if el == "" else int(el))

        return tuple(positions)

class _ParseDateRange(argparse.Action):
    """
    Custom action to parse the date range from the command line.

    The date range is passed as a string with two dots separating the start and end dates. The format must be YYYY-MM-DD.
    """
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: str, option_string: str | None = None):
        
        setattr(namespace, self.dest, self.parse_date_range(parser, values))

    @staticmethod
    def parse_date_range(parser: argparse.ArgumentParser, values: str | None) -> tuple[datetime | None, datetime | None] | None:
        """
        Parse the date range from the values.

        :param parser: the parser.
        :type parser: argparse.ArgumentParser
        :param values: the values to parse.
        :type values: str | None
        :return: the parsed date range.
        :rtype: tuple[datetime | None, datetime | None] | None
        """

        if values is None: return None

        if not(".." in values):
            parser.error("The date range must be separated by two dots. Example: 2020-01-01..2020-12-31")
        if values.count(".") > 2:
            parser.error("The date range must contain '..' as separator")

        _split = values.split("..")

        range = []
        for date in _split:
            if date == "":
                range.append(None)
                continue
            try:
                range.append(datetime.strptime(date, "%Y-%m-%d"))
            except ValueError:
                parser.error("Incorrect date format, should be YYYY-MM-DD")

        return tuple(range)


def _parse_arguments() -> argparse.Namespace:
    """
    Parse the arguments from the command line.

    :return: the parsed arguments.
    :rtype: argparse.Namespace
    """

    # True parser. If the -ij argument is not passed, it will be used to parse the arguments
    parser = _ArgumentParserWithHelpOnError(description=PACKAGE_DESCRIPTION)
    parser.add_argument(
        "seqs",
        type=str,
        help="Path to the input fasta file containing the sequences."
    )
    parser.add_argument(
        "meta",
        type=str,
        help="Path to the corresponding metadata file for the sequences."
    )
    parser.add_argument(
        "out",
        type=str,
        help="Path to the output filename prefix used to save the different results."
    )
    parser.add_argument(
        "-dt",
        "--delta_t",
        type=str,
        default="7D",
        help="Time interval to calculate the statistics. Default is 7 days (7D)."
    )
    parser.add_argument(
        "-sh",
        "--show",
        action="store_true",
        help="Show the plots of the analysis."
    )
    parser.add_argument(
        "-ep",
        "--export_plots",
        action="store_true",
        help="Export the plots of the analysis."
    )
    parser.add_argument(
        "-l",
        "--length_filter",
        type=int,
        default=0,
        help="Length filter for the sequences (removes sequences with length less than the specified value). Default is 0."
    )
    parser.add_argument(
        "-n",
        "--n_threshold",
        type=int,
        default=2,
        help="Minimum number of sequences required in a time interval to compute statistics. Default is 2."
    )
    parser.add_argument(
        "-xj",
        "--export_json",
        action="store_true",
        help="Export the run arguments to a json file."
    )
    parser.add_argument(
        "-ij",
        "--import_json",
        type=str,
        help="Import the run arguments from a JSON file. If this argument is passed, the other arguments are ignored. The JSON file must contain the mandatory keys 'seqs', 'meta', and 'out'."
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        choices=["all", "total", "substitutions",  "indels"],
        default="all",
        help="Kind of mutations to consider for the analysis. Default is 'all'."
    )
    parser.add_argument(
        "-f",
        "--filter",
        nargs='+',  # Accepts multiple arguments
        action=_ParseFilter,
        default=None,
        help="Specify filters to be applied on the data with keys followed by values. If the values are multiple, they must be enclosed in square brackets. Example: --filter key1 value1 key2 [value2 value3] key3 value4. If either the keys or values contain spaces, they must be enclosed in quotes. keys must be present in the metadata file as columns for the filter to be applied. Use '*' as a wildcard, for example Bio* to filter all columns starting with 'Bio'."
    )
    parser.add_argument(
        "-gp",
        "--genome_positions",
        type=str,
        action=_ParseGenomePosition,
        default=None,
        help="Genome positions to restrict the analysis. The positions must be separated by two dots. Example: 1..1000. Open start or end positions are allowed by omitting the first or last position, respectively. If not specified, the whole reference genome is considered."
    )
    parser.add_argument(
        "-dr",
        "--date_range",
        type=str,
        action=_ParseDateRange,
        default=None,
        help="Date range to filter the data. The date range must be separated by two dots and the format must be YYYY-MM-DD. Example: 2020-01-01..2020-12-31. If not specified, the whole dataset is considered. Note that if the origin is specified, the most restrictive date range is considered."
    )

    # Initial parser to parse just the -ij argument
    json_input_parser = argparse.ArgumentParser(add_help=False)
    json_input_parser.add_argument(
        "-ij",
        "--import_json",
        type=str
    )
    json_input_args, _ = json_input_parser.parse_known_args()

    # If the -ij argument is passed, the arguments are imported from the JSON file
    if json_input_args.import_json:
        with open(json_input_args.import_json, "r") as file:
            # Dumps the arguments to the namespace
            _args = json.load(file)

            # Checks if the JSON file contains the minimum required keys
            if not {"seqs", "meta", "out"}.issubset(set(_args.keys())):
                parser.error("The JSON file must contain the keys 'seqs', 'meta', and 'out'")

            # Initialize a new namespace
            namespace = argparse.Namespace()

            # Apply the JSON values to the namespace
            for action in parser._actions:
                if action.dest in _args:
                    value = _args[action.dest]

                    # If the argument has a custom action, apply the action manually
                    if isinstance(action, (_ParseFilter, _ParseGenomePosition, _ParseDateRange)):
                        action(parser, namespace, value)
                    else:
                        # For regular arguments, just set them in the namespace
                        setattr(namespace, action.dest, value)
                else:
                    # If no value from JSON, use the default value
                    setattr(namespace, action.dest, action.default)

            return namespace

    return parser.parse_args()

def _simple_serializer(k: str, v: any) -> any:
    """
    Simple serializer to convert the arguments to JSON.

    :param k: the key of the argument.
    :type k: str
    :param v: the value of the argument.
    :type v: any
    :return: the serialized value.
    :rtype: any
    """

    if k == "date_range":
        return "..".join(map(lambda x: x.strftime("%Y-%m-%d") if x else "", v))
    return v

def _main():
    """
    Command line interface for :class:`PyEvoMotion`.

    It parses the arguments from the command line and runs the analysis with the specified parameters.
    """
    print(BANNER)
    args = _parse_arguments()

    # If the -xj argument is passed, the arguments are exported to a JSON file before running the analysis altogether
    if args.export_json:
        with open(f"{args.out}_run_args.json", "w") as file:
            json.dump(
                {
                    k: _simple_serializer(k, v)
                    for k, v in vars(args).items()
                    if k not in ["export_json", "import_json"]
                },
                file,
                indent=4
            )

    # Instantiates the PyEvoMotion class, which parses the data on construction
    instance = PyEvoMotion(
        args.seqs,
        args.meta,
        dt=args.delta_t,
        filters=args.filter,
        positions=args.genome_positions,
        date_range=args.date_range,
    )

    # Exports the data to a TSV file
    instance.data.to_csv(
        f"{args.out}.tsv",
        sep="\t",
        index=False
    )
    
    # Runs the analysis
    stats, reg = instance.analysis(
        length=args.length_filter,
        n_threshold=args.n_threshold,
        show=args.show,
        mutation_kind=args.kind,
        export_plots_filename=(
            f"{args.out}_plots"
            if args.export_plots
            else None
        )
    )

    _reg = reg.copy()

    for k in _reg.keys():
        del _reg[k]["model"]

    # Exports the statistic results to TSV file
    stats.to_csv(
        f"{args.out}_stats.tsv",
        sep="\t",
        index=False
    )

    # Exports the regression models to a JSON file
    with open(f"{args.out}_regression_results.json", "w") as file:
        json.dump(_reg, file, indent=4)

    # Exits the program with code 0 (success)
    exit(0)
    
if __name__ == "__main__":
    _main()