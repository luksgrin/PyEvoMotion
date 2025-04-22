import pytest
from os import makedirs

from PyEvoMotion import PyEvoMotion

# Setup
@pytest.fixture
def setup():
    yield PyEvoMotion(
        "tests/data/test1/test1.sequences.fasta",
        "tests/data/test1/test1.metadata.tsv"
    )

def test_parse_file(setup):
    """Tests that PyEvoMotion can parse the input files correctly.
    """

    # Create an instance
    instance = setup

    # Create the output directory
    makedirs("tests/data/test1/output", exist_ok=True)

    # Save the data
    instance.data.to_csv("tests/data/test1/output/test1.data.tsv", sep="\t", index=False)

    # Assert True if no errors are raised
    assert instance.data is not None
