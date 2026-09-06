"""Golden-output comparison for CLI runs.

PyEvoMotion's data and statistics tables are deterministic across platforms
(stable ordering, canonical arithmetic), so `<out>.tsv` and `<out>_stats.tsv`
are compared byte for byte with committed references under
tests/data/golden/<name>/. The fitted models (`<out>_regression_results.json`)
depend on the platform's libm through exp/log and the t and F distributions,
so their numbers are compared with a relative tolerance (REL_TOL) while their
structure and text must match exactly.

Set PYEVOMOTION_UPDATE_GOLDENS=1 to (re)write the references instead of
comparing; do that only after an intentional behaviour change, and commit
the result together with the change.
"""
import json
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
        got_path = output_prefix + suffix
        exp_path = os.path.join(golden_dir, "out" + suffix)
        hint = f"If the change is intentional, rerun with {UPDATE_ENV}=1 and commit the new goldens."
        if suffix.endswith(".json"):
            # Fitted models: the regressions use exp/log and the t and F
            # distributions, whose last digits depend on the platform's libm.
            # Structure and text must match exactly; numbers within tolerance.
            with open(got_path) as f, open(exp_path) as g:
                worst = _compare_json(json.load(f), json.load(g), path="")
            assert worst[0] <= REL_TOL, (
                f"{got_path} differs from the golden {exp_path}: max relative difference "
                f"{worst[0]:.3e} at {worst[1]} (tolerance {REL_TOL:.0e}). {hint}"
            )
        else:
            got = open(got_path, "rb").read()
            exp = open(exp_path, "rb").read()
            assert got == exp, f"{got_path} differs from the golden {exp_path}. {hint}"


# Relative tolerance for fitted-model numbers (tables are compared byte for
# byte). Linear fits agree across platforms to ~1e-12; the power-law fit is
# an iterative Levenberg-Marquardt optimisation whose stopping point moves
# with the platform's exp/log, observed up to ~3e-7 between arm64 and x86_64.
REL_TOL = 1e-6
ABS_TOL = 1e-12


def _compare_json(got, exp, path):
    """Return (max relative difference, where) over all numbers; raise on any
    structural or textual difference."""
    if isinstance(exp, dict):
        assert isinstance(got, dict) and got.keys() == exp.keys(), f"keys differ at {path or '/'}: {sorted(got.keys()) if isinstance(got, dict) else type(got)} vs {sorted(exp.keys())}"
        worst = (0.0, path)
        for k in exp:
            w = _compare_json(got[k], exp[k], f"{path}/{k}")
            worst = max(worst, w)
        return worst
    if isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), f"list length differs at {path}"
        worst = (0.0, path)
        for i, (a, b) in enumerate(zip(got, exp)):
            worst = max(worst, _compare_json(a, b, f"{path}[{i}]"))
        return worst
    if isinstance(exp, bool) or exp is None or isinstance(exp, str):
        assert got == exp, f"value differs at {path}: {got!r} vs {exp!r}"
        return (0.0, path)
    # numbers (NaN == NaN)
    assert isinstance(got, (int, float)), f"type differs at {path}"
    if isinstance(exp, float) and exp != exp:
        assert isinstance(got, float) and got != got, f"expected NaN at {path}, got {got!r}"
        return (0.0, path)
    diff = abs(got - exp)
    rel = 0.0 if diff <= ABS_TOL else diff / max(abs(exp), ABS_TOL)
    return (rel, path)
