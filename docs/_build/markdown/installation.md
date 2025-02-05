# Installation

#### NOTE
`PyEvoMotion` uses [mafft](https://mafft.cbrc.jp/alignment/software/) to do the sequence alignment. If it’s not available in your system, the installation script will install it locally.

If so, ensure to restart your shell session or run `source ~/.bashrc` to update the PATH environment variable, so that the `mafft` executable is available in your shell.

To install `PyEvoMotion` you must clone the repository and run the installation script:

```bash
git clone https://github.com/luksgrin/PyEvoMotion
cd PyEvoMotion
python3 -m pip install .
```

This will install the package and its dependencies.

To check if the installation was successful, you can run the following command:

```bash
PyEvoMotion
```

If the installation was successful, you should see the following output:

```bash
Welcome to Rodrigolab's
_____       ______          __  __       _   _
|  __ \     |  ____|        |  \/  |     | | (_)
| |__) |   _| |____   _____ | \  / | ___ | |_ _  ___  _ __
|  ___/ | | |  __\ \ / / _ \| |\/| |/ _ \| __| |/ _ \| '_ \
| |   | |_| | |___\ V / (_) | |  | | (_) | |_| | (_) | | | |
|_|    \__, |______\_/ \___/|_|  |_|\___/ \__|_|\___/|_| |_|
      __/ |
      |___/

usage: PyEvoMotion [-h] [-dt DELTA_T] [-sh] [-ep] [-l LENGTH_FILTER] [-n N_THRESHOLD] [-xj] [-ij IMPORT_JSON] [-k {all,total,substitutions,insertions,deletions,indels}] [-f FILTER [FILTER ...]] [-gp GENOME_POSITIONS] [-dr DATE_RANGE] seqs meta out

PyEvoMotion

positional arguments:
seqs                  Path to the input fasta file containing the sequences.
meta                  Path to the corresponding metadata file for the sequences.
out                   Path to the output filename prefix used to save the different results.

options:
-h, --help            show this help message and exit
-dt DELTA_T, --delta_t DELTA_T
                        Time interval to calculate the statistics. Default is 7 days (7D).
-sh, --show           Show the plots of the analysis.
-ep, --export_plots   Export the plots of the analysis.
-l LENGTH_FILTER, --length_filter LENGTH_FILTER
                        Length filter for the sequences (removes sequences with length less than the specified value). Default is 0.
-n N_THRESHOLD, --n_threshold N_THRESHOLD
                        Minimum number of sequences required in a time interval to compute statistics. Default is 2.
-xj, --export_json    Export the run arguments to a json file.
-ij IMPORT_JSON, --import_json IMPORT_JSON
                        Import the run arguments from a JSON file. If this argument is passed, the other arguments are ignored. The JSON file must contain the mandatory keys 'seqs', 'meta', and 'out'.
-k {all,total,substitutions,insertions,deletions,indels}, --kind {all,total,substitutions,insertions,deletions,indels}
                        Kind of mutations to consider for the analysis. Default is 'all'.
-f FILTER [FILTER ...], --filter FILTER [FILTER ...]
                        Specify filters to be applied on the data with keys followed by values. If the values are multiple, they must be enclosed in square brackets. Example: --filter key1 value1 key2 [value2 value3] key3 value4. If either the keys or values
                        contain spaces, they must be enclosed in quotes. keys must be present in the metadata file as columns for the filter to be applied. Use '*' as a wildcard, for example Bio* to filter all columns starting with 'Bio'.
-gp GENOME_POSITIONS, --genome_positions GENOME_POSITIONS
                        Genome positions to restrict the analysis. The positions must be separated by two dots. Example: 1..1000. Open start or end positions are allowed by omitting the first or last position, respectively. If not specified, the whole
                        reference genome is considered.
-dr DATE_RANGE, --date_range DATE_RANGE
                        Date range to filter the data. The date range must be separated by two dots and the format must be YYYY-MM-DD. Example: 2020-01-01..2020-12-31. If not specified, the whole dataset is considered. Note that if the origin is specified, the
                        most restrictive date range is considered.

Error: the following arguments are required: seqs, meta, out
```
