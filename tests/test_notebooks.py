"""Contains tests to ensure that the notebooks run without errors."""
import typing as t
from pathlib import Path

import papermill as pm
import pytest

from pythonpractice import REPO_ROOT


@pytest.mark.notebook
@pytest.mark.parametrize(
    "notebook_path",
    [x for x in (REPO_ROOT / "notebooks").glob("*.ipynb") if not x.name.startswith("_")],
    ids=lambda x: x.name,
)
def test_notebooks(notebook_path: Path, tmp_path: Path) -> None:
    """Test that the notebooks run without errors."""
    pm.execute_notebook(
        input_path=notebook_path,
        output_path=tmp_path / notebook_path.name,
        parameters={},
        kernel_name="python3",
        log_output=True,
    )
