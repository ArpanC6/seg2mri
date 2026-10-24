import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg2mri.smoke import make_smoke  # noqa: E402


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("smoke")
    return make_smoke(str(root / "data"), n_cases=6, size=32, seed=0)
