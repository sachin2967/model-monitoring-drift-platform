import sys
from pathlib import Path

import pytest

# Add project root to sys.path
proj_root = Path(__file__).resolve().parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFT_STATE_PATH", str(tmp_path / "reports.sqlite3"))
    monkeypatch.setenv("APP_ENV", "local")
    for name in ("DRIFT_API_KEY", "FRAUD_REFERENCE_PATH", "FRAUD_REFERENCE_SHA256"):
        monkeypatch.delenv(name, raising=False)
