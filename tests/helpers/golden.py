"""Golden-output comparison for CLI runs.

PyEvoMotion's outputs are deterministic across platforms (stable ordering,
canonical arithmetic), so the files a run produces are compared byte for
byte with committed references under tests/data/golden/<name>/.

Set PYEVOMOTION_UPDATE_GOLDENS=1 to (re)write the references instead of
comparing; do that only after an intentional behaviour change, and commit
the result together with the change.
"""
import os
import shutil

GOLDEN_ROOT = "tests/data/golden"
UPDATE_ENV = "PYEVOMOTION_UPDATE_GOLDENS"

STATS_FILES = ("_stats.tsv", "_regression_results.json")
ALL_FILES = (".tsv",) + STATS_FILES


def updating() -> bool:
    return os.environ.get(UPDATE_ENV, "").lower() not in ("", "0", "false", "no")


def check_golden(output_prefix: str, name: str, files=STATS_FILES) -> None:
    """Compare ``<output_prefix><suffix>`` with ``tests/data/golden/<name>/out<suffix>``
    for every suffix in ``files`` (or write them when updating)."""
    golden_dir = os.path.join(GOLDEN_ROOT, name)
    if updating():
        os.makedirs(golden_dir, exist_ok=True)
        for suffix in files:
            shutil.copyfile(output_prefix + suffix, os.path.join(golden_dir, "out" + suffix))
        return
    if not os.path.isdir(golden_dir):
        raise AssertionError(
            f"No golden outputs under {golden_dir}. Run the tests once with {UPDATE_ENV}=1 to create them."
        )
    for suffix in files:
        got = open(output_prefix + suffix, "rb").read()
        exp = open(os.path.join(golden_dir, "out" + suffix), "rb").read()
        assert got == exp, (
            f"{output_prefix}{suffix} differs from the golden {golden_dir}/out{suffix}. "
            f"If the change is intentional, rerun with {UPDATE_ENV}=1 and commit the new goldens."
        )
