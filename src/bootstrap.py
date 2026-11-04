"""Sanitize import path before loading third-party packages.

Shared hosts sometimes inject ~/.local via PYTHONPATH, which breaks
venv-installed onnx/protobuf/pydicom stacks.
"""

from __future__ import annotations

import os
import sys


def sanitize_sys_path() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    cleaned: list[str] = []
    for entry in sys.path:
        norm = entry.replace("\\", "/")
        if "/.local/lib/python" in norm:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


sanitize_sys_path()
