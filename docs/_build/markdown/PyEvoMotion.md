# PyEvoMotion package

## Subpackages

* [PyEvoMotion.core package](PyEvoMotion.core.md)
  * [Submodules](PyEvoMotion.core.md#submodules)
  * [PyEvoMotion.core.base module](PyEvoMotion.core.md#module-PyEvoMotion.core.base)
    * [`PyEvoMotionBase`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase)
      * [`PyEvoMotionBase.F_test()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.F_test)
      * [`PyEvoMotionBase.adjust_model()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.adjust_model)
      * [`PyEvoMotionBase.count_prefixes()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.count_prefixes)
      * [`PyEvoMotionBase.date_grouper()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.date_grouper)
      * [`PyEvoMotionBase.linear_regression()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.linear_regression)
      * [`PyEvoMotionBase.mutation_length_modification()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.mutation_length_modification)
      * [`PyEvoMotionBase.plot_single_data_and_model()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.plot_single_data_and_model)
      * [`PyEvoMotionBase.power_law_fit()`](PyEvoMotion.core.md#PyEvoMotion.core.base.PyEvoMotionBase.power_law_fit)
  * [PyEvoMotion.core.core module](PyEvoMotion.core.md#module-PyEvoMotion.core.core)
    * [`PyEvoMotion`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion)
      * [`PyEvoMotion.analysis()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.analysis)
      * [`PyEvoMotion.compute_stats()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.compute_stats)
      * [`PyEvoMotion.count_mutation_types()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.count_mutation_types)
      * [`PyEvoMotion.export_plot_results()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.export_plot_results)
      * [`PyEvoMotion.get_lengths()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.get_lengths)
      * [`PyEvoMotion.length_filter()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.length_filter)
      * [`PyEvoMotion.n_filter()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.n_filter)
      * [`PyEvoMotion.plot_results()`](PyEvoMotion.core.md#PyEvoMotion.core.core.PyEvoMotion.plot_results)
  * [PyEvoMotion.core.parser module](PyEvoMotion.core.md#module-PyEvoMotion.core.parser)
    * [`PyEvoMotionParser`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser)
      * [`PyEvoMotionParser.create_modifs()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.create_modifs)
      * [`PyEvoMotionParser.filter_by_daterange()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.filter_by_daterange)
      * [`PyEvoMotionParser.filter_by_position()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.filter_by_position)
      * [`PyEvoMotionParser.filter_columns()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.filter_columns)
      * [`PyEvoMotionParser.generate_alignment()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.generate_alignment)
      * [`PyEvoMotionParser.get_differing_mutations()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.get_differing_mutations)
      * [`PyEvoMotionParser.parse_data()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.parse_data)
      * [`PyEvoMotionParser.parse_metadata()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.parse_metadata)
      * [`PyEvoMotionParser.parse_sequence_by_id()`](PyEvoMotion.core.md#PyEvoMotion.core.parser.PyEvoMotionParser.parse_sequence_by_id)
  * [Module contents](PyEvoMotion.core.md#module-PyEvoMotion.core)

## Submodules

## PyEvoMotion.cli module

Command line interface for [`PyEvoMotion`](#module-PyEvoMotion).

It parses the arguments from the command line and runs the analysis with the specified parameters.

This module is not meant to be inherited from, but to be used as a standalone script in the command line.

## Module contents

The main functionality of the `PyEvoMotion` project is abstracted into the following classes:

* [`PyEvoMotion`](#module-PyEvoMotion) - The main class that encapsulates the entire analysis.
* `PyEvoMotionBase` - The base class that provides basic utility functions inherited by [`PyEvoMotion`](#module-PyEvoMotion).
* `PyEvoMotionParser` - The class that provides the functionality to parse the input data for the analysis, inherited by [`PyEvoMotion`](#module-PyEvoMotion).
