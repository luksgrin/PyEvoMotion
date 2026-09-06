# Changelog

## 0.2.0 (unreleased)

### The package is now written in Rust

`PyEvoMotion` is a single compiled extension module (PyO3 + maturin). The
public API is unchanged and everything is importable directly from
`PyEvoMotion`: `PyEvoMotion`, `PyEvoMotionBase`, `PyEvoMotionParser`. The
old `PyEvoMotion.core.*` import paths are gone. Pre-built wheels are
published for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon) and
Windows on CPython 3.12; building from the sdist needs a Rust toolchain.

Sequence alignment uses the pure-Rust [`mafft`](https://crates.io/crates/mafft)
port bundled in the wheel. There is no external `mafft` binary to install
any more and the first-run installer is gone.

### New command-line options

Contributed by Tom Eulenfeld ([#1](https://github.com/luksgrin/PyEvoMotion/pull/1)),
ported to the Rust code base:

- `-ref/--refseq FASTA`: use the first record of a FASTA file as the
  reference instead of the earliest-dated sequence.
- `-v/--verbose`: log each pipeline stage and show an alignment progress
  counter on stderr (`-vv` for debug output with timestamps). Library users
  can configure the `PyEvoMotion` Python logger instead.
- `-load/--load_mutation_instructions TSV`: rebuild the analysis from the
  `<out>.tsv` of a previous run, skipping the alignment step. Filters still
  apply, and the data TSV is not rewritten in this mode.
- `-recount/--recount_mutation_types`: recompute the per-sequence counts
  from loaded instructions instead of reusing the stored columns.

All of them round-trip through `-xj`/`-ij`. The same options exist as
keyword arguments of the `PyEvoMotion` constructor.

### Internals in Rust: the data pipeline no longer runs on pandas

Between reading the inputs and returning results, the pipeline now works
on an internal Rust column store (`rust/src/table.rs`, design in
`rust/DESIGN_internal_table.md`): TSV/CSV reading with pandas' inference
rules, date parsing, filters, the alignment-result merge, per-sequence
counts, the per-window statistics and the TSV writer are all Rust.
pandas is used only at the API boundary: `instance.data` is still a
`pandas.DataFrame` (materialised when you read it, re-ingested if you
assign or edit it), `analysis()` still returns `(DataFrame, dict)`, and
`parse_metadata` / `parse_mutation_data` / `get_differing_mutations` /
`compute_stats` still return DataFrames. Subclasses that read and assign
`self.data` keep working unchanged. New for subclass authors:
`PyEvoMotion.Table` with `Table.from_pandas(df)`, `table.to_pandas()`,
`table.to_tsv(path)`, and `self.data = table`.

Two accidental behaviours of the pandas implementation were replaced by
deterministic ones (**results are now identical on every platform**):

- **Row order**: sequences sharing a collection date used to be ordered by
  numpy's unstable quicksort; they now keep the order of the metadata
  file (stable sort by date). `<out>.tsv` rows may therefore appear in a
  different order than with 0.1.x.
- **Variance**: the per-window sample variance is computed with a fixed
  summation order and no fused multiply-add. Values differ from 0.1.x in
  the last digits (about 1e-15 absolute on the bundled data), and no
  longer differ between Apple Silicon and x86_64 machines as they did
  before. Fitted parameters move by about 1e-14 relative.

Means, window sizes, window dates and the set of rows are unchanged. The
data and statistics tables are byte-identical across platforms; the fitted
parameters in `<out>_regression_results.json` still depend on the
platform's math library through exp/log and the t and F distributions:
linear fits agree to about 1e-12 relative and the iterative power-law fits
to about 1e-6 (the golden tests check exactly that).

### Removed dependencies

- Biopython is no longer used. FASTA files are read by a streaming Rust
  parser; `PyEvoMotion.reference`, the records yielded by the new
  `PyEvoMotionParser.read_fasta(path)` and the pair returned by
  `generate_alignment` are `PyEvoMotion.SequenceRecord` objects with `id`,
  `description` and `seq` (a plain `str`) attributes. Code that only used
  `.id`, `.seq`, `len()` or `str()` on the reference keeps working.
- `scikit-learn` was declared but never used; `pytest` is a development
  dependency (`uv sync` installs it, `pip install .` does not). Runtime
  dependencies are now pandas, numpy and matplotlib.

### Changed

- **Mutation instruction positions are 1-based for every kind.**
  Substitutions always were (`s_241_T`); insertions and deletions were
  0-based. `d_P_BASES` now deletes starting at reference position P and
  `i_P_BASES` inserts so that the first inserted base sits at P. Only the
  written coordinates change: the genome-window filter keeps admitting
  exactly the same mutations as before, so statistics and fitted models are
  unchanged. TSV files written by 0.1.x still carry 0-based indel
  positions; `-load` does not translate them.
- Metadata rows whose id has no record in the FASTA file are dropped with a
  warning (count and example ids on stderr) instead of crashing later.
- The FASTA file is no longer read to the end once every selected id has
  been seen.

### Testing

- The UK/USA dataset tests default to a fixed-seed, date-stratified subset of
  the manuscript sample (1,000 sequences per country), published as the
  `ci-data-v1` GitHub release asset and cached by CI. `PYEVOMOTION_FULL_TEST_DATA=1`
  selects the full 11 GB dataset; the "Full UK/USA dataset tests" workflow
  runs it on demand, one test per job. The synthetic-dataset tests likewise
  run on a fixed-seed sample of 400 of their 2,001 sequences unless the
  variable is set. Random samples in the tests use a fixed seed.

### Documentation

- The site is built and deployed by the `Docs` workflow from the `docs/`
  sources on `prod` (Sphinx plus rustdoc under `/rustdoc/`), replacing the
  hand-pushed `docs-gh-pages` branch and `build_docs.sh`. The unused
  Markdown build is gone.

### Fixed

- `-ep` plot export failed with `NameError: 'model'`.
- `-xj` without `-dr` crashed while serialising the run arguments.
- A filter that emptied the dataset raised `KeyError: 'mutation
  instructions'` instead of the "dataset empty" message.
- A reference id missing from the input FASTA raises a clear error.
- `-load` combined with `-dr`/`-f` and `-recount` (or a loaded file without
  count columns) produced float count columns with NaN holes when the
  filtered table had a gapped index; counts are now computed row-wise.

## 0.1.3

Last pure-Python release. See the git history of the `prod` branch.
