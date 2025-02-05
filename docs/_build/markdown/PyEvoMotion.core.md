# PyEvoMotion.core package

## Submodules

## PyEvoMotion.core.base module

### *class* PyEvoMotion.core.base.PyEvoMotionBase

Bases: `object`

Base class for the `PyEvoMotion` project.

This class contains no data and is meant to be used as a mixin (provides utility methods for the project). It is inherited by [`PyEvoMotion`](PyEvoMotion.md#module-PyEvoMotion).

#### *static* F_test(model1: dict[str, any], model2: dict[str, any], data: ndarray) → tuple[float, float]

Perform an F-test between two models.

See [https://en.wikipedia.org/wiki/F-test#Regression_problems](https://en.wikipedia.org/wiki/F-test#Regression_problems) for more details.

* **Parameters:**
  * **model1** (*dict* *[**str* *,* *any* *]*) – The first model.
  * **model2** (*dict* *[**str* *,* *any* *]*) – The second model.
  * **data** (*np.ndarray*) – The data to test the models.
* **Returns:**
  A tuple with the F-value and the p-value.
* **Return type:**
  `tuple[float, float]`

#### *classmethod* adjust_model(x: Series, y: Series, name: str = None) → dict[str, any]

Adjust a model to the data.

* **Parameters:**
  * **x** (*pd.Series*) – The features. It is a single pandas Series.
  * **y** (*pd.Series*) – The target. It is a single pandas Series.
  * **name** (*str*) – The name of the data. Default is `None`.
* **Returns:**
  A dictionary with the model.
* **Return type:**
  `dict[str, any]`
* **Raises:**
  **ValueError** – If the dataset is empty or full of NaN values. This may occur if the grouped data contains only one entry per group, indicating that the variance cannot be computed.

#### *static* count_prefixes(prefix: str, mutations: list[str]) → int

Count the number of mutations that start with a specific prefix.

* **Parameters:**
  * **prefix** (*str*) – The prefix to count. It must be a single character.
  * **mutations** (*list* *[**str* *]*) – The list of mutations where to count the prefix.
* **Returns:**
  The number of mutations that start with the prefix.
* **Return type:**
  `int`

#### *static* date_grouper(df: DataFrame, DT: str, origin: str) → DataFrameGroupBy

Create grouped dataframe based on a `datetime` frequency.

* **Parameters:**
  * **df** (*pd.DataFrame*) – The dataframe to group. It must have a `date` column.
  * **DT** (*str*) – The string datetime that will govern the grouping.
  * **origin** (*str*) – The string datetime that will be the origin of the grouping frequency.
* **Return grouped:**
  The dataset’s corresponding pandas groupby object.
* **Return type:**
  `pd.core.groupby.generic.DataFrameGroupBy`

#### *classmethod* linear_regression(x: ndarray, y: ndarray, fit_intercept=True) → dict[str, any]

Perform a linear regression on a set of data.

* **Parameters:**
  * **x** (*np.ndarray*) – A numpy array of the features.
  * **y** (*np.ndarray*) – A numpy array of the target.
  * **fit_intercept** (*bool*) – Whether to fit the intercept. Default is `True`.
* **Returns:**
  A dictionary containing:
  * `model`: A `lambda` function that computes predictions based on the fitted model.
  * `parameters`: A dictionary with the slope of the regression line.
  * `expression`: A string representation of the regression equation.
  * `r2`: The $R^2$ score of the regression.
* **Return type:**
  `dict[str, any]`

#### *static* mutation_length_modification(mutation: str) → int

Get the length modification induced by a mutation.

* **Parameters:**
  **mutation** (*str*) – The mutation whose length modification to get.
* **Returns:**
  The length modification induced by the mutation.
* **Return type:**
  `int`
* **Raises:**
  **ValueError** – If the mutation is not one of `s`, `i` or `d`.

#### *static* plot_single_data_and_model(data_x: RangeIndex, data_y: Series, data_ylabel: str, model: callable, model_label: str, data_xlabel_units: str, ax: any, \*\*kwargs: dict[str, any]) → None

Low level utility function to plot the data and a model.

* **Parameters:**
  * **data_x** – The x-axis data.
  * **data_y** – The y-axis data.
  * **data_ylabel** (*str*) – The `ylabel` of the data.
  * **model** (*dict* *[**str* *,* *any* *]*) – The model to plot.
  * **model_label** (*str*) – The label of the model.
  * **data_xlabel_units** (*str*) – The units of the x-axis data.
  * **ax** (*any*) – The axis to plot.
  * **kwargs** (*dict* *[**str* *,* *any* *]*) – Additional arguments to pass to the plot

#### *classmethod* power_law_fit(x: ndarray, y: ndarray) → dict[str, any]

Perform a power law fit on a set of data.

* **Parameters:**
  * **x** (*np.ndarray*) – A numpy array of the features.
  * **y** (*np.ndarray*) – A numpy array of the target.
* **Returns:**
  A dictionary containing:
  * `model`: A `lambda` function that computes predictions based on the fitted model.
  * `parameters`: A dictionary with the parameters of the fitted power law.
  * `expression`: A string representation of the regression equation.
  * `r2`: The $R^2$ score of the regression.
* **Return type:**
  `dict[str, any]`

## PyEvoMotion.core.core module

### *class* PyEvoMotion.core.core.PyEvoMotion(input_fasta: str, input_meta: str, dt: str = '7D', filters: dict[str, list[str] | str] | None = None, positions: tuple[int] | None = None, date_range: tuple[str] | None = None)

Bases: [`PyEvoMotionParser`](#PyEvoMotion.core.parser.PyEvoMotionParser), [`PyEvoMotionBase`](#PyEvoMotion.core.base.PyEvoMotionBase)

Main class to analyze the data as intended by `PyEvoMotion`. This class inherits from `PyEvoMotionParser` and `PyEvoMotionBase`. On construction, it calls [`count_mutation_types()`](#PyEvoMotion.core.core.PyEvoMotion.count_mutation_types).

* **Parameters:**
  * **input_fasta** (*str*) – The path to the input `.fasta` file.
  * **input_meta** (*str*) – The path to the input metadata file. It has to have a column named `date`. Accepts `.csv` and `.tsv` files. Default is `None`.
  * **dt** (*str*) – The string datetime interval that will govern the grouping for the statistics. Default is 7 days (`7D`).
  * **filters** (*dict* *[**str* *,* *list* *[**str* *]*  *|* *str* *]*  *|* *None*) – The filters to apply to the data. Default is `None`.
  * **positions** (*tuple* *[**int* *]*  *|* *None*) – The positions to filter by. Default is `None`.
  * **date_range** (*tuple* *[**str* *]*  *|* *None*) – The date range to filter by. Default is `None`.

### Attributes:

data: `pd.DataFrame`
: The parsed data from the input files.

reference: `str`
: The reference sequence.

\_MUTATION_TYPES: `list[str]`
: The types of mutations that can be found in the data. Namely `substitutions` and `indels`.

#### analysis(length: int, n_threshold: int | None = None, show: bool = False, mutation_kind: str = 'all', export_plots_filename: str | None = None) → tuple[DataFrame, dict[str, dict[str, any]]]

Perform the global analysis of the data.

It computes the statistics and the regression models for the mean and variance of the data.

* **Parameters:**
  * **length** (*int*) – The length to filter by.
  * **n_threshold** – Minimum number of sequences required in a time interval to compute statistics.
  * **show** (*bool*) – Whether to show the plots or not. Default is False.
  * **mutation_kind** (*str*) – The kind of mutation to compute the statistics for. Has to be one of `all`, `total`, `substitutions` or `indels`. Default is `all`.
  * **export_plots** (*str* *|* *None*) – Filename to export the plots. Default is None and does not export the plots.
* **Returns:**
  The statistics and the regression models.
* **Return type:**
  `tuple[pd.DataFrame, dict[str, dict[str, any]]]`

#### compute_stats(DT: str, origin: str, n_threshold: int | None = None, mutation_kind: str = 'all') → DataFrame

Compute the length, mean and variance of the data.

It computes the mean and variance of the data for the specified mutation kind (or kinds) in the specified datetime interval and origin.

* **Parameters:**
  * **DT** (*str*) – The string datetime interval that will govern the grouping.
  * **origin** (*str*) – The string datetime that will be the origin of the grouping.
  * **n_threshold** (*int* *|* *None*) – Minimum number of sequences required in a time interval to compute statistics.
  * **mutation_kind** – The kind of mutation to compute the statistics for. Has to be one of `all`, `total`, `substitutions`, `insertions`, `deletions` or `indels`. Default is `all`.
* **Returns:**
  The statistics of the data.
* **Return type:**
  `pd.DataFrame`

#### count_mutation_types() → None

Count the number of substitutions, insertions and deletions in the data.

It updates the `data` attribute by adding the columns `number of substitutions`, `number of indels` and `number of mutations`.

#### *classmethod* export_plot_results(stats: DataFrame, regs: dict[str, dict[str, any]], data_xlabel_units: str, output_ptr: str | None = None) → None

Export the results of the analysis to a `.pdf` file.

* **Parameters:**
  * **stats** (*pd.DataFrame*) – The statistics of the data.
  * **regs** (*dict* *[**str* *,* *dict* *[**str* *,* *any* *]* *]*) – The regression models.
  * **data_xlabel_units** (*str*) – The data `xlabel` units for the plot.
  * **output_ptr** – The output `.pdf` file. If `None`, it will create a new `.pdf` file.

#### get_lengths() → Series

Get the lengths of the sequences in the dataset.

* **Returns:**
  The lengths of the sequences.
* **Return type:**
  `pd.Series`

#### length_filter(length: int, how: str = 'gt') → None

Filter the data by sequence length.

It updates the `data` attribute by filtering the data by the sequence length.

* **Parameters:**
  * **length** (*int*) – The length to filter by.
  * **how** (*str*) – The filter condition. It can be `gt` (greater than), `lt` (less than) or `eq` (equal to).

#### n_filter(threshold: float | int = 0.01, how: str = 'lt') → None

Filter the data by the number of `N` in the sequence.

It updates the `data` attribute by filtering the data by the number of `N` in the sequence.

* **Parameters:**
  * **threshold** (*float* *|* *int*) – The threshold to filter by. Must be between 0 and 1. Default is 0.01.
  * **how** (*str*) – The filter condition. It can be `gt` (greater than), `lt` (less than) or `eq` (equal to).

#### *classmethod* plot_results(stats: DataFrame, regs: dict[str, dict[str, any]], data_xlabel_units: str) → None

Plot the results of the analysis.

* **Parameters:**
  * **stats** (*pd.DataFrame*) – The statistics of the data. The first column has to be the date, the second column has to be the mean and the third column has to be the variance.
  * **regs** (*dict* *[**str* *,* *dict* *[**str* *,* *any* *]* *]*) – The regression models.
  * **data_xlabel** (*str*) – The data `xlabel` units.

## PyEvoMotion.core.parser module

### *class* PyEvoMotion.core.parser.PyEvoMotionParser(input_fasta: str, input_meta: str, filters: dict[str, list[str] | str], positions: tuple[int], date_range: tuple[datetime] | None)

Bases: `object`

This class is responsible for parsing the input fasta and metadata files. It is inherited by the [`PyEvoMotion`](PyEvoMotion.md#module-PyEvoMotion) class.

* **Parameters:**
  * **input_fasta** (*str*) – The path to the input `.fasta` file.
  * **input_meta** (*str*) – The path to the input metadata file. The metadata file must contain a `date` column and can be in either `.csv` or `.tsv` format.
  * **filters** (*dict* *[**str* *,* *list* *[**str* *]*  *|* *str* *]*) – The filters to be applied to the data. The keys are the column names and the values are the values to be filtered.
  * **positions** (*tuple* *[**int* *]*) – The start and end positions to filter the data by.
  * **date_range** (*tuple* *[**datetime* *]*  *|* *None*) – The start and end dates to filter the data by. If `None`, the date range is not filtered.

On construction, it invokes the following methods:
: - [`parse_metadata()`](#PyEvoMotion.core.parser.PyEvoMotionParser.parse_metadata): Parses the metadata file.
  - [`parse_sequence_by_id()`](#PyEvoMotion.core.parser.PyEvoMotionParser.parse_sequence_by_id): Parses the sequence with the ID of the first entry in the metadata file.
  - [`filter_columns()`](#PyEvoMotion.core.parser.PyEvoMotionParser.filter_columns): Filters the metadata based on the input filters.
  - [`filter_by_daterange()`](#PyEvoMotion.core.parser.PyEvoMotionParser.filter_by_daterange): Filters the metadata based on the input date range.
  - [`parse_data()`](#PyEvoMotion.core.parser.PyEvoMotionParser.parse_data): Parses the input fasta file and appends the mutations that differ between the reference sequence and the input sequences to the metadata.
  - [`filter_by_position()`](#PyEvoMotion.core.parser.PyEvoMotionParser.filter_by_position): Filters the metadata based on the input start and end positions.

### Attributes:

data: `pd.DataFrame`
: DataFrame containing metadata.

reference: `SeqRecord`
: The reference sequence parsed from the fasta file.

#### *classmethod* create_modifs(alignment: MultipleSeqAlignment) → list[str]

Creates a guide on how to modify the root sequence to get the appropriate mutant sequence.

* **Parameters:**
  **alignment** (*MultipleSeqAlignment*) – `Bio.Align.MultipleSeqAlignment` object containing the alignment.
* **Return mods:**
  List of modifications encoded as strings that have the format `<type>_<position>_<bases>`, where `<type>` is one of `i`, `d` and `s` (insertion, deletion and substitution), `<position>` is the position in the sequence where the modification should be made, and `<bases>` are the bases to be inserted, deleted or substituted.
* **Return type:**
  `list[str]`

#### filter_by_daterange(start: datetime, end: datetime) → None

Filter the data based on a date range.

The data is filtered to only include entries with dates between the start and end dates. This method modifies the `data` attribute in place.

* **Parameters:**
  * **start** (*datetime*) – The start date.
  * **end** (*datetime*) – The end date.
* **Raises:**
  **ValueError** – If the start date is greater than the end date.

#### filter_by_position(start: int, end: int) → None

Filter the data based on some start and end positions in the reference sequence.

*Note that the positions are 1-indexed, and that the end position is inclusive.*

* **Parameters:**
  * **start** (*int*) – The start position index.
  * **end** (*int*) – The end position index.
* **Raises:**
  **ValueError** – If the start position is greater than the end position., or if the start position is greater than the length of the reference sequence.

#### filter_columns(filters: dict[str, list[str] | str]) → None

Filter the data based on the input filters provided.

* **Parameters:**
  **filters** (*dict* *[**str* *,* *list* *[**str* *]*  *|* *str* *]*) – The filters to be applied to the data. The keys are the column names and the values are the values to be filtered from the provided metadata.

#### *classmethod* generate_alignment(seq1: str, seq2: str) → MultipleSeqAlignment

Generate a multiple sequence alignment of the input sequences using `MAFFT`.

* **Parameters:**
  * **seq1** (*str*) – The first sequence to be aligned.
  * **seq2** (*str*) – The second sequence to be aligned.
* **Returns:**
  The aligned sequences.
* **Return type:**
  `MultipleSeqAlignment`

#### get_differing_mutations(input_fasta: str, selection: Series) → DataFrame

Return the mutations that differ between the reference sequence and the input sequence.

Also, for the sake of sequence selection, it outputs the number of `N` found in the sequence.

* **Parameters:**
  * **input_fasta** (*str*) – The path to the input `.fasta` file.
  * **selection** (*pd.Series*) – The selection of sequence ids to be compared with the reference sequence.
* **Returns:**
  The mutations that differ between the reference sequence and the input sequence. It contains the columns `id`, `mutation instructions` and `N count` (the number of `N` in the sequence).
* **Return type:**
  `pd.DataFrame`

#### parse_data(input_fasta: str, selection: Series) → None

Parse the input fasta file and store the resulting data in the `data` attribute.

* **Parameters:**
  * **input_fasta** (*str*) – The path to the input `.fasta` file.
  * **selection** (*pd.Series*) – The selection of sequence ids to be parsed.

#### *static* parse_metadata(input_meta: str) → DataFrame

Parse the metadata file into a `pandas.DataFrame`.

* **Parameters:**
  **input_meta** (*str*) – The path to the metadata file, in either `.csv` or `.tsv` format.
* **Returns:**
  The metadata as a `pd.DataFrame`. It must contain a `date` column. Other columns are optional.
* **Return type:**
  `pd.DataFrame`
* **Raises:**
  **ValueError** – If the metadata file does not contain a `date` column.

#### *static* parse_sequence_by_id(input_fasta: str, \_id: str) → SeqRecord | None

Parse the input `.fasta` file and return the `Bio.SeqRecord` with the given `_id`. Returns `None` if the `_id` is not found.

* **Parameters:**
  * **input_fasta** (*str*) – The path to the input `.fasta` file.
  * **\_id** (*str*) – The ID of the sequence to be returned.
* **Returns:**
  The sequence record with the given `_id`. `None` if the `_id` is not found.
* **Return type:**
  SeqRecord | None

## Module contents

The main functionality of the `PyEvoMotion` project is abstracted into the following classes:

* [`PyEvoMotion`](PyEvoMotion.md#module-PyEvoMotion) - The main class that encapsulates the entire analysis.
* `PyEvoMotionBase` - The base class that provides basic utility functions inherited by [`PyEvoMotion`](PyEvoMotion.md#module-PyEvoMotion).
* `PyEvoMotionParser` - The class that provides the functionality to parse the input data for the analysis, inherited by [`PyEvoMotion`](PyEvoMotion.md#module-PyEvoMotion).
