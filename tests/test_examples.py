"""Les exemples s'exécutent (téléchargent des poids)."""
import runpy
from pathlib import Path

import pytest


@pytest.mark.weights
def test_quickstart(capsys):
    pytest.importorskip("deepinv")
    runpy.run_path(str(Path(__file__).parents[1] / "examples" / "quickstart.py"), run_name="__main__")
    assert "USRNet" in capsys.readouterr().out
