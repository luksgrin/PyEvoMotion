PyEvoMotion package
===================

``PyEvoMotion`` is a single compiled extension module (written in Rust,
built with PyO3). Every public name lives directly under ``PyEvoMotion``.
The analysis classes form one inheritance chain, so an instance of
:class:`~PyEvoMotion.PyEvoMotion` has every method listed on this page.
FASTA input is handled by :class:`~PyEvoMotion.SequenceRecord` and
:class:`~PyEvoMotion.FastaReader`; the dataset itself lives in a
:class:`~PyEvoMotion.Table` and is exposed to Python as a pandas DataFrame
through the ``data`` attribute.

.. automodule:: PyEvoMotion

Analysis
--------

.. autoclass:: PyEvoMotion.PyEvoMotion(input_fasta, input_meta, dt="7D", filters=None, positions=None, date_range=None, refseq=None, verbose=0, load_mutation_instructions=None, recount_mutation_types=False)
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PyEvoMotion._PyEvoMotionCore(*args, **kwargs)
   :members:
   :undoc-members:
   :show-inheritance:

Parsing and alignment
---------------------

.. autoclass:: PyEvoMotion.PyEvoMotionParser(input_fasta, input_meta, filters, positions, date_range=None, refseq=None, verbose=0, load_mutation_instructions=None)
   :members:
   :undoc-members:
   :show-inheritance:

FASTA records
-------------

.. autoclass:: PyEvoMotion.SequenceRecord(id, seq, description=None)
   :members:
   :undoc-members:

.. autoclass:: PyEvoMotion.FastaReader(path)
   :members:
   :undoc-members:

Internal table
--------------

.. autoclass:: PyEvoMotion.Table
   :members:
   :undoc-members:

Mathematical utilities
----------------------

.. autoclass:: PyEvoMotion.PyEvoMotionBase()
   :members:
   :undoc-members:
   :show-inheritance:

Command-line entry point
------------------------

.. autofunction:: PyEvoMotion._main
