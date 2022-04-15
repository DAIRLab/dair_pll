# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'dair_pll'
copyright = '2022, Mathew Halm & DAIR Lab'
author = 'Mathew Halm'

# The full version, including alpha/beta/rc tags
release = 'v0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxcontrib.napoleon',
              'sphinx_toolbox.more_autodoc.typehints',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'sphinxcontrib.bibtex'
              ]

bibtex_bibfiles = ['references.bib']

autoclass_content = 'both'

intersphinx_mapping = {
    'pydrake': ('https://drake.mit.edu/pydrake/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/1.21/', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['index_text']

autodoc_type_aliases = {'DrakeSpatialInertia':
                            'dair_pll.drake_utils.DrakeSpatialInertia',
                        'DrakeBody':
                            'dair_pll.drake_utils.DrakeBody'
                        }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 8,
    'sticky_navigation': True,
    'collapse_navigation': False
}

autodoc_member_order = 'bysource'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
