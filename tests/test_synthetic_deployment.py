"""Exercise a dedicated database, temp index, loopback providers and actual HTTP sockets."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from sqlalchemy.engine import make_url


def test_synthetic_deployment_and_offline_recovery(tmp_path):
    url = os.getenv("MEDEVIDENCE_TEST_POSTGRES_URL")
    redis = os.getenv("MEDEVIDENCE_TEST_REDIS_URL")
    if not url or not redis:
        pytest.skip("requires explicit disposable PostgreSQL and Redis URLs")
    target = make_url(url)
    db = "verification_" + uuid4().hex
    uploads, raw = tmp_path / "uploads", tmp_path / "raw"
    uploads.mkdir()
    raw.mkdir()
    env = dict(
        os.environ,
        POSTGRESQL_URL=target.set(database=db).render_as_string(False),
        REDIS_URL=redis,
        DATA_ROOT=str(tmp_path),
        CHROMA_PERSIST_DIRECTORY=str(tmp_path / "idx"),
        CHROMA_COLLECTION_NAME="verification",
        ANONYMIZED_TELEMETRY="False",
        NO_PROXY="127.0.0.1,localhost",
    )
    admin_url = target.set(drivername="postgresql").render_as_string(False)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
        try:
            child = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "support" / "synthetic_deployment.py"),
                ],
                env=env,
                check=False,
                timeout=180,
            )
            assert child.returncode == 0
            assert list(uploads.rglob("interrupted.txt"))
            recovery = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "med_evidence_agentic_rag.services.recovery_cli",
                    "--apply",
                    "--uploads",
                    str(uploads),
                    "--base",
                    str(raw),
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            assert "'emails_unknown': 1" in recovery.stdout
            assert "'completed': 1" in recovery.stdout
            assert not list(uploads.rglob("interrupted.txt"))
            with psycopg.connect(
                target.set(drivername="postgresql", database=db).render_as_string(False)
            ) as connection:
                states = dict(
                    connection.execute(
                        "SELECT status, count(*) FROM email_actions GROUP BY status"
                    ).fetchall()
                )
                assert states == {"sent": 1, "unknown": 1}
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(db)))
