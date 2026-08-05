"""Prevent unbuilt PyTorch checkouts from shadowing installed torch."""

from __future__ import annotations

import os
from importlib.machinery import PathFinder
from pathlib import Path
import sys


class _InstalledTorchFinder:
    """Resolve the initial torch import without the unbuilt checkout."""

    def __init__(self, checkout: Path) -> None:
        self.checkout = checkout

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" or path is not None:
            return None

        search_path = []
        for entry in sys.path:
            try:
                resolved = Path(entry or os.curdir).resolve()
            except OSError:
                search_path.append(entry)
                continue
            if resolved != self.checkout:
                search_path.append(entry)

        spec = PathFinder.find_spec(fullname, search_path, target)
        if spec is not None:
            sys.meta_path.remove(self)
        return spec


def _prefer_installed_torch_for_python_c() -> None:
    """
    Keep nested ``python -c`` commands on the installed torch package.

    The test runner uses a PyTorch source checkout as pytest's working
    directory. Python normally adds that directory to sys.path for ``-c``,
    even when the checkout is only providing tests and has not generated
    torch/version.py. Route the initial torch import around only that checkout
    entry. Normal ``python path/to/script.py`` commands retain their script
    directory so sibling imports continue to work.
    """
    if os.environ.get("FRAMEWORK_SCRIPTS_SAFE_PYTHON_C") != "1":
        return
    if sys.argv[0] != "-c":
        return

    cwd = Path.cwd().resolve()
    source_torch = cwd / "torch"
    if not (source_torch / "__init__.py").is_file():
        return
    if (source_torch / "version.py").is_file():
        return

    # CPython adds the ``python -c`` working directory to sys.path after
    # sitecustomize runs, so removing it here would be undone. A temporary
    # finder instead filters that directory when the first torch import is
    # resolved, then removes itself.
    sys.meta_path.insert(0, _InstalledTorchFinder(cwd))


_prefer_installed_torch_for_python_c()
