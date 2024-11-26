# Configuration file for the Sphinx documentation builder.

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('../../src'))  # Adjust as needed

# -- Project information -----------------------------------------------------
project = 'torchHydroNodes'
author = 'Jesus Perez Curbelo'
copyright = f'{datetime.datetime.now().year}, {author}'
release = '0.1.0'  # Version of your project

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',      # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.todo',          # Support for TODO directives
    'sphinx.ext.mathjax',  # latex rendering of equations using MathJax
    'sphinx.ext.intersphinx',   # Link to other Sphinx documentation
    'nbsphinx',  # For rendering Jupyter notebooks
    'nbsphinx_link', # For linking to Jupyter notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', 
                    '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# Path to logo image file
html_logo = '_static/img/torchhydronodes-logo.webp'
html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
templates_path = ['_templates']

# -- Options for extensions --------------------------------------------------
# autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

# napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True