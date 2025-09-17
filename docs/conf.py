"""Sphinx configuration."""

import importlib.metadata
import sys
from pathlib import Path
from typing import Any

here = Path(__file__).parent
sys.path.insert(0, str((here.parent / "src").resolve()))

project = "potamides"
copyright = "2025, Sirui"
author = "Sirui"
version = release = importlib.metadata.version("potamides")

extensions = [
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "sphinx_book_theme"

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/wsr1998/potamides",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_isses_button": True,
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wsr1998/potamides",
            "icon": "fa-brands fa-github",
        },
    ],
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "attrs_block",
]

myst_directives = {
    "plot": "matplotlib.sphinxext.plot_directive.plot_directive",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

# Configure matplotlib plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["png"]
plot_rcparams = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
}

# Configure doctest
doctest_global_setup = """
import jax
import jax.numpy as jnp
import potamides as ptd
import matplotlib.pyplot as plt
import numpy as np
"""

always_document_param_types = True
