# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# autodoc imports the *installed* PyEvoMotion extension module (there is no
# Python source tree), so the package must be installed in the environment
# that runs sphinx-build: `uv sync --extra docs`.

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyEvoMotion'
copyright = '2024, Lucas Goiriz & Guillermo Rodrigo'
author = 'Lucas Goiriz & Guillermo Rodrigo'
from importlib.metadata import version as _pkg_version
release = _pkg_version('PyEvoMotion')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex'
]

bibtex_bibfiles = ["reference.bib"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
napoleon_use_rtype = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    '_css/style_extension.css',
]