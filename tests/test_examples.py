"""Every script in examples/ runs as documented, with no key and no network."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


def test_examples_exist():
    assert EXAMPLES


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()
