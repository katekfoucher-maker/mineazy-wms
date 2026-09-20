"""Shared fixtures.

The temp DB + output dir env vars are set at import time (before any ``wms``
module is imported) so the engine binds to the throw-away database.
"""
import os
import tempfile

os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db").replace("\\", "/")
os.environ["OUTPUT_DIR"] = tempfile.mkdtemp()
# the weekly model's neural candidates train real nets — skip them under the
# suite (test_weekly_neural.py exercises ES-RNN on its own)
os.environ.setdefault("WEEKLY_ESRNN", "false")
os.environ.setdefault("WEEKLY_NEURALPROPHET", "false")
# the real app defaults the weekly forecasting engine to monthly-sourced data
# (see weekly_forecast.load_panel) - the suite's many synthetic-weekly-data
# fixtures (monkeypatched weekly_dir()) test the weekly-cadence code paths
# directly, so they need the original "weekly" source pinned here
os.environ.setdefault("WEEKLY_DATA_SOURCE", "weekly")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _fast_hash():
    """Swap bcrypt for a cheap hash so the suite isn't dominated by KDF cost."""
    import hashlib
    import wms.security as sec

    def h(p: str) -> str:
        return "sha256$" + hashlib.sha256(p.encode()).hexdigest()

    def v(p: str, hashed) -> bool:
        return bool(hashed) and hashed == h(p)

    sec.hash_password, sec.verify_password = h, v
    import wms.web.routes as wr           # rebind the names it imported directly
    wr.hash_password, wr.verify_password = h, v
    yield


@pytest.fixture(scope="session")
def seeded(_fast_hash):
    from wms.db import Base, engine
    from wms import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    from wms.scripts.seed import run
    run()
    yield
    engine.dispose()


@pytest.fixture()
def db(seeded):
    from wms.db import SessionLocal
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client(seeded):
    from fastapi.testclient import TestClient
    from wms.api.main import app
    with TestClient(app) as c:
        c.headers.update({"X-Actor": "controller"})
        yield c
