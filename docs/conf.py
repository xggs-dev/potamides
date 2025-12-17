"""Sphinx configuration."""

import importlib.metadata
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from docutils import nodes
from sphinx import addnodes
from sphinx.transforms import SphinxTransform

here = Path(__file__).parent
sys.path.insert(0, str((here.parent / "src").resolve()))

project: Final[str] = "potamides"
copyright: Final[str] = "2025, Sirui"
author: Final[str] = "Sirui"

# Try to get version from package metadata, fallback to importing the module
try:
    version = release = importlib.metadata.version("potamides")
except importlib.metadata.PackageNotFoundError:
    # If package not installed, try to import and get version
    try:
        import potamides as ptd

        version = release = ptd.__version__
    except (ImportError, AttributeError):
        # If all else fails, use a placeholder
        version = release = "unknown"

extensions: Final[list[str]] = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

source_suffix: Final[list[str]] = [".rst", ".md"]
exclude_patterns: Final[list[str]] = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme: Final[str] = "sphinx_book_theme"

html_theme_options: dict[str, Any] = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/wsr1998/potamides",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
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

# Enable MyST parsing in docstrings
myst_update_mathjax: Final = True

# MyST-NB configuration for executing notebooks
nb_execution_mode: Final = "cache"  # "off", "auto", "cache", or "force"
nb_execution_timeout: Final = 180  # seconds
nb_execution_allow_errors: Final = False
nb_execution_raise_on_error: Final = True
nb_execution_show_tb: Final = True

myst_directives: Final[dict[str, str]] = {
    "plot": "matplotlib.sphinxext.plot_directive.plot_directive",
}

intersphinx_mapping: Final[dict[str, tuple[str, str | None]]] = {
    "python": ("https://docs.python.org/3", None),
    "interpax": ("https://interpax.readthedocs.io/en/latest", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping/", "static/jaxtyping.inv"),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "unxt": ("https://unxt.readthedocs.io/en/latest/", None),
}

nitpick_ignore: Final[list[tuple[str, str]]] = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

# Configure matplotlib plot directive
plot_include_source: Final = True
plot_html_show_source_link: Final = False
plot_html_show_formats: Final = False
plot_formats: Final[list[str]] = ["png"]
plot_rcparams: Final[dict[str, Any]] = {
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
doctest_global_setup: Final = """
import jax
import jax.numpy as jnp
import potamides as ptd
import matplotlib.pyplot as plt
import numpy as np
"""

always_document_param_types: Final = True

# Configure Napoleon to preserve math syntax
napoleon_use_math: Final = True
napoleon_custom_sections: Final = None

# Ensure autodoc processes docstrings as CommonMark/MyST
autodoc_docstring_signature: Final = True

# Tuples: "(N,)", "(N, 2)", "(n1, n2, n3)", "(..., 2)", etc.
_SHAPE_TUPLE_RE: Final[str] = (
    r"^\(\s*(?:\.\.\.|[A-Za-z_]\w*|\d+)(?:\s*,\s*(?:\.\.\.|[A-Za-z_]\w*|\d+))*\s*\)$"
)

# Only allow these tokens: N, S, 1, 2, or literal "..."
_SHAPE_NAME_RE: Final[str] = r"^(?:F|N|S|1|2|\.\.\.)$"

nitpick_ignore_regex: Final[list[tuple[str, str]]] = [
    ("py:class", _SHAPE_TUPLE_RE),
    ("py:data", _SHAPE_TUPLE_RE),
    ("py:class", _SHAPE_NAME_RE),
    ("py:data", _SHAPE_NAME_RE),
]

# -----------------------------------------------------------------------------
# --- Auto-link bare tokens like "Array" only inside parameter/return terms ----


# Map visible token → (domain:role, fully-qualified target)
BARE_XREFS: Mapping[str, tuple[str, str]] = {
    "Array": ("py:class", "jaxtyping.Array"),
    "Float": ("py:data", "jaxtyping.Float"),
    "Real": ("py:data", "jaxtyping.Real"),
    "Int": ("py:data", "jaxtyping.Int"),
}

_SKIP_PARENTS: Final = (
    nodes.literal,  # inline code ``like this``
    nodes.literal_block,  # code blocks
    nodes.reference,  # existing links
    nodes.title,  # section titles
    nodes.emphasis,
    nodes.strong,
)


def _under_param_or_return_term(text_node: nodes.Text) -> bool:
    """Return True if this text node is somewhere under a 'term' node of a
    definition list (e.g., the 'x : Array' term produced by Napoleon)."""
    p = text_node.parent
    while p is not None:
        if isinstance(p, nodes.term):
            return True
        p = p.parent
    return False


class AutoLinkBareWordsInTerms(SphinxTransform):
    default_priority = 850  # after parsing; before writing

    def apply(self) -> None:
        if not BARE_XREFS:
            return
        # Compile one regex that matches any configured token as a whole word
        pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, BARE_XREFS.keys())) + r")\b"
        )

        for text_node in list(self.document.traverse(nodes.Text)):
            # Skip code, links, titles, etc.
            if isinstance(text_node.parent, _SKIP_PARENTS):
                continue
            # Only operate in parameter/return terms (e.g., "x : Array", "Array" in Returns)
            if not _under_param_or_return_term(text_node):
                continue

            text = text_node.astext()
            out: list[nodes.Node] = []
            last = 0
            changed = False

            for m in pattern.finditer(text):
                if m.start() > last:
                    out.append(nodes.Text(text[last : m.start()]))

                word = m.group(0)
                role, target = BARE_XREFS[word]
                domain, reftype = role.split(":", 1)

                # Create a cross-ref that intersphinx can resolve (to your local jaxtyping.inv)
                ref = addnodes.pending_xref(
                    "",
                    refdomain=domain,
                    reftype=reftype,
                    reftarget=target,
                    modname=None,
                    classname=None,
                )
                # Link text: keep as plain text (or use nodes.literal for code style)
                ref += nodes.Text(word)
                out.append(ref)

                last = m.end()
                changed = True

            if not changed:
                continue
            if last < len(text):
                out.append(nodes.Text(text[last:]))
            text_node.parent.replace(text_node, out)


def _process_docstring_math(
    _app,
    _what,
    _name,
    _obj,
    _options,
    lines,
):
    """Convert $$ and $ syntax to RST math directives in docstrings."""

    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for block math ($$)
        if line.strip() == "$$":
            # Start of block math
            result.append("")
            result.append(".. math::")
            result.append("")
            i += 1

            # Collect math content until closing $$
            while i < len(lines) and lines[i].strip() != "$$":
                # Add proper indentation for math content
                if lines[i].strip():
                    result.append("    " + lines[i])
                else:
                    result.append("")
                i += 1

            # Skip closing $$
            i += 1
            result.append("")
        else:
            # Convert inline math $...$ to :math:`...`
            converted_line = re.sub(r"\$([^\$]+)\$", r":math:`\1`", line)
            result.append(converted_line)
            i += 1

    lines[:] = result


def setup(app):
    app.add_transform(AutoLinkBareWordsInTerms)
    # Convert $$ and $ in docstrings to RST math directives
    app.connect("autodoc-process-docstring", _process_docstring_math)
    # app.connect("missing-reference", _shape_missing_ref)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
