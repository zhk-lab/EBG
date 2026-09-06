"""Shared, quiet parsing for repository Python sources."""

from __future__ import annotations

import ast
import warnings


def parse_python(source: str, path: str) -> ast.Module:
    """Parse source while containing warnings caused by legacy literals."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename=path)
