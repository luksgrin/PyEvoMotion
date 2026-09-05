.. _usage_guide:

Usage Guide
===========

This guide provides detailed information on how to use ``PyEvoMotion`` for various evolutionary analyses, including preparation of input data, command-line options, and interpretation of results.

Prerequisites
-------------

Before you begin, ensure you have:

1. ``PyEvoMotion`` installed (see :ref:`Installation <installation>`).
2. A ``FASTA`` file containing DNA sequences.
3. A metadata file with corresponding information for each sequence, of which the first column must match the sequence IDs in the ``FASTA`` file and must contain the collection date in ``YYYY-MM-DD`` format in a ``date`` column for each sequence.

Input Data Requirements
-----------------------

``PyEvoMotion`` requires two primary input files:

1. ``FASTA`` **file**: Contains nucleotide sequences to be analyzed.
2. **Metadata file**: Contains information about each sequence, including:

   - A unique identifier matching the sequence ID in the ``FASTA`` file.
   - Collection date for each sequence (in ``YYYY-MM-DD`` format).
   - Additional metadata fields that can be used for filtering (optional).

Metadata File Format
^^^^^^^^^^^^^^^^^^^^

The metadata file can be in ``CSV`` or ``TSV`` format with headers. Example structure:

.. code-block:: none

   sequence_id,date,country,variant,clade
   seq1,2020-01-15,USA,Alpha,G
   seq2,2020-02-20,Canada,Alpha,G
   seq3,2020-03-10,UK,Alpha,GH
   ...

Command-Line Interface
----------------------

Main Arguments
^^^^^^^^^^^^^^

PyEvoMotion requires three mandatory arguments:

1. ``seqs``: Path to the input ``FASTA`` file.
2. ``meta``: Path to the metadata file.
3. ``out``: Prefix for output files.

Example basic command:

.. code-block:: bash

   PyEvoMotion sequences.fasta metadata.csv results/analysis1

Time Interval Options
^^^^^^^^^^^^^^^^^^^^^

The ``-dt`` or ``--delta_t`` parameter controls the time interval for analysis (please see `pandas' offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_ for more details):

.. code-block:: bash

   # Analyze with 30-day intervals
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -dt 30D

   # Analyze with 2-month intervals
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -dt 2M

   # Analyze with 1-year intervals
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -dt 1Y

Filtering Options
^^^^^^^^^^^^^^^^^

``PyEvoMotion`` provides several filtering options:

**1. Length filtering**

Filter out sequences shorter than a specified length (useful when analyzing low-coverage genomes):

.. code-block:: bash

   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -l 29000

**2. Date range filtering**

Restrict analysis to a specific time period:

.. code-block:: bash

   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -dr 2020-03-01..2020-12-31

**3. Metadata-based filtering**

Filter sequences based on metadata attributes (the filters to be applied will be specific to the particular metadata file used):

.. code-block:: bash

   # Single value filter
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -f country "United States"

   # Multiple value filter
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -f variant [Alpha Beta Gamma]

   # Multiple filters
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -f country "United States" clade [GH GRY]

**4. Genome position filtering**

Restrict analysis to specific regions of the genome (useful when analyzing the evolution of specific genes or genomic regions):

.. code-block:: bash

   # Analyze positions 1000 to 2000
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -gp 1000..2000

   # Analyze from beginning to position 1000
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -gp ..1000

   # Analyze from position 28000 to the end
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -gp 28000..

Visualization Options
^^^^^^^^^^^^^^^^^^^^^

Control the visualization and export of results:

.. code-block:: bash

   # Show plots interactively
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -sh

   # Export plots as image files
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -ep

   # Both show and export plots
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -sh -ep

Mutation Type Analysis
^^^^^^^^^^^^^^^^^^^^^^

Specify which types of mutations to analyze:

.. code-block:: bash

   # Analyze all mutation types separately
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -k all

   # Analyze only substitutions
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -k substitutions

   # Analyze only insertions and deletions combined
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -k indels

``JSON`` Configuration
^^^^^^^^^^^^^^^^^^^^^^

For reproducibility, you can export and import run configurations:

.. code-block:: bash

   # Export run configuration to JSON
   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -xj

   # Import run configuration from JSON
   PyEvoMotion -ij config.json

Reference Sequence
^^^^^^^^^^^^^^^^^^

By default the reference is the sequence with the earliest collection date.
``-ref``/``--refseq`` uses the first record of a FASTA file instead, which
keeps runs on different datasets comparable:

.. code-block:: bash

   PyEvoMotion sequences.fasta metadata.csv results/analysis1 -ref wuhan-hu-1.fasta

Progress Output
^^^^^^^^^^^^^^^

``-v``/``--verbose`` logs each stage of the pipeline and shows an alignment
progress counter on stderr; ``-vv`` adds debug output with timestamps. When
using the library, configure the ``PyEvoMotion`` logger from Python's
``logging`` module instead.

Re-using a Previous Run
^^^^^^^^^^^^^^^^^^^^^^^

Aligning every sequence against the reference is by far the slowest step.
``-load``/``--load_mutation_instructions`` rebuilds the analysis from the
``{prefix}.tsv`` written by an earlier run, so the alignment is skipped.
Filters (``-f``, ``-gp``, ``-dr``), the time interval and the mutation kind
can all be changed:

.. code-block:: bash

   # First run: aligns everything and writes results/full.tsv
   PyEvoMotion sequences.fasta metadata.csv results/full

   # Re-analyse the same data for spring 2020 only, in seconds
   PyEvoMotion sequences.fasta metadata.csv results/spring \
      -load results/full.tsv -dr 2020-03-01..2020-05-31

In this mode the metadata file is ignored and the FASTA file is only read to
fetch the reference sequence (unless ``-ref`` is given). The data TSV is not
rewritten. Add ``-recount``/``--recount_mutation_types`` to recompute the
per-sequence mutation counts from the loaded instructions rather than
reusing the stored columns.

Interpreting Results
--------------------

Output Files
^^^^^^^^^^^^

``PyEvoMotion`` generates several output files:

1. **Data File**:

   - ``{prefix}.tsv``: One row per sequence with its metadata, ``N count``,
     mutation counts and ``mutation instructions``: a list of ``s_P_B``
     (base ``B`` at position ``P``), ``i_P_BASES`` (``BASES`` inserted so
     that the first inserted base sits at position ``P``) and ``d_P_BASES``
     (``BASES`` deleted starting at position ``P``). Positions are 1-based
     reference coordinates for all three kinds (since 0.2.0; earlier
     versions wrote insertions and deletions 0-based). This file can be fed
     back in with ``-load``.

2. **Statistics Files**:

   - ``{prefix}_stats.csv``: Contains raw statistics for each time interval.
   - ``{prefix}_regression_results.json``: The model parameters and the goodness of fit.

3. **Plot Files** (when using ``-ep``):

   - ``{prefix}_plots.pdf``: The time series of mutation statistics.

Understanding Results
^^^^^^^^^^^^^^^^^^^^^

The main output of PyEvoMotion includes:

1. **Statistical parameters** (along with the best-fit model and the corresponding :math:`R^2` value indicating the goodness of fit) describing the evolutionary process:

   - Anomalous molecular clock model:
     
     - ``expression: d*x^alpha``
     - ``parameters``:
       
       - :math:`\alpha` (alpha): The diffusion exponent, indicating whether evolution follows standard diffusion (:math:`\alpha = 1`) or anomalous diffusion (:math:`\alpha \neq 1`)
       - :math:`D`: The diffusion coefficient, representing the overall rate of evolution
     - ``r2``: The :math:`R^2` value of the model

   - Standard molecular clock model:
     
     - ``expression: mx``, ``mx + b`` if the intercept is fit to be non-zero
     - ``parameters``:
       
       - ``m``: The rate of mutation
       - ``b``: The intercept of the model (initial mutation rate; only present if the intercept is fit to be non-zero)
     - ``r2``: The :math:`R^2` value of the model

2. **Time series data** showing the mean and variance of the number of mutations over time along with the fit model.

A value of :math:`\alpha \approx 1` suggests that mutations accumulate according to standard molecular clock models, while deviations indicate more complex evolutionary dynamics. In the case of :math:`\alpha \approx 1`, then the anomalous molecular clock model is equivalent to the standard molecular clock model with :math:`m = D`, :math:`b = 0` and :math:`\alpha = 1`.


Example 1: Simple Analysis
--------------------------

Let's start with a simple analysis using the default parameters:

.. code-block:: bash

   PyEvoMotion path/to/sequences.fasta path/to/metadata.tsv output_prefix

This command will:
   - Load sequences from the ``FASTA`` file.
   - Load corresponding metadata.
   - Align every sequence against the reference with the bundled ``MAFFT`` port.
   - Calculate mutation statistics using a 7-day interval.
   - Save the results with the specified output prefix.

The output will include:
   - Summary statistics for mutations over time.
   - Distribution parameters for the evolution model.
   - Alignment files for further analysis.

Example 2: Filtering Data by Date Range
---------------------------------------

To restrict your analysis to a specific time period:

.. code-block:: bash

   PyEvoMotion path/to/sequences.fasta path/to/metadata.tsv output_prefix -dr 2020-01-01..2020-06-30

This will analyze only sequences from the first half of 2020.

Example 3: Analyzing Specific Mutation Types
--------------------------------------------

If you're interested in analyzing only certain types of mutations:

.. code-block:: bash

   PyEvoMotion path/to/sequences.fasta path/to/metadata.csv output_prefix -k substitutions

Options for `-k` include: `all`, `total`, `substitutions` and `indels`.

Example 4: Visualizing Results
------------------------------

To generate and display plots of your analysis:

.. code-block:: bash

   PyEvoMotion path/to/sequences.fasta path/to/metadata.csv output_prefix -sh -ep

The `-sh` flag shows the plots interactively, while `-ep` exports them as ``.pdf`` files.

Example 5: Advanced Filtering
-----------------------------

``PyEvoMotion`` allows for complex filtering based on metadata:

.. code-block:: bash

   PyEvoMotion path/to/sequences.fasta path/to/metadata.csv output_prefix -f country "United States" variant [Alpha Delta]

This example filters sequences from the United States that belong to either the Alpha or Delta variants.

Advanced Topics
---------------

Using Docker
^^^^^^^^^^^^

``PyEvoMotion`` is available as a Docker image for easy deployment:

.. code-block:: bash

   # Pull the Docker image
   docker pull ghcr.io/luksgrin/pyevomotion:latest

   #  Run the image interactively. It contains ``PyEvoMotion`` fully installed.
   docker run -it ghcr.io/luksgrin/pyevomotion:latest

Programmatic Usage
^^^^^^^^^^^^^^^^^^^

``PyEvoMotion`` can also be used as a Python library:

.. code-block:: python

   from PyEvoMotion import PyEvoMotion
   
   # Initialize PyEvoMotion with input files
   # Please see the module documentation for more details on constructor arguments
   pem = PyEvoMotion(
      "sequences.fasta",
      "metadata.csv",
      dt="10D", # 10 days time interval
   )
   
    # Runs the analysis
    # Please see the module documentation for more details on the analysis method
    stats, reg = instance.analysis(
      length=29000, # Set a length filter to 29000 (for example; to remove short sequences)
      mutation_kind="substitutions" # Set the type of mutations to analyze
   )

   # Pandas DataFrames with the statistics
   print(stats)
   # Dictionary with the model parameters and the goodness of fit
   print(reg)

Troubleshooting
---------------

**Problem**: Too few sequences per time interval.
**Solution**: Increase the time interval with ``-dt`` to attempt to have more sequences per time interval.
