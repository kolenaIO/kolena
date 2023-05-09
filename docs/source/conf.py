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
import os
import sys
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

from sphinx.application import Sphinx

sys.path.insert(0, os.path.abspath("../kolena"))


# -- Project information -----------------------------------------------------

project = "kolena"
copyright = f"{datetime.now().year} Kolena. All rights reserved"
author = "Kolena Engineering <eng@kolena.io>"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

# Preserve ordering of class attributes in documentation (instead of alphabetical)
autodoc_member_order = "bysource"


def process_bases(app: str, name: str, obj: object, options: Dict[str, Any], bases: List[Any]) -> None:
    if any("kolena.detection._internal" in str(base) for base in bases):
        options["inherited-members"] = True


def setup(app: Sphinx) -> None:
    app.connect("autodoc-process-bases", process_bases)


# Add any paths that contain templates here, relative to this directory.
# not to be confused with templates used by sphinx-apidoc to render reST
templates_path: List[str] = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"  # default
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}
html_theme_options = dict(
    favicons=[
        dict(
            rel="icon",
            href="favicon.png",
        ),
    ],
    logo=dict(
        link="index.html",
        image_light="wordmark-purple.svg",
        image_dark="wordmark-white.svg",
    ),
    external_links=[
        dict(name="App", url="https://app.kolena.io"),
        dict(name="Docs", url="https://docs.kolena.io"),
    ],
    secondary_sidebar_items=[],  # remove right sidebar, not useful
    footer_items=["copyright"],
    pygment_light_style="friendly",  # code syntax highlighting -- see: https://pygments.org/styles
    pygment_dark_style="material",
)
