import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

CLI = Path(__file__).resolve().parents[1] / "scripts" / "maintenance.py"


def run_cli(*arguments, url):
    env = dict(os.environ)
    if url:
        env["POSTGRESQL_URL"] = url
    else:
        env.pop("POSTGRESQL_URL", None)
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, arguments)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_maintenance_refuses_missing_opt_in_or_mismatched_database(tmp_path):
    artifact = tmp_path / "backup.dump"
    url = "postgresql+asyncpg://fixture:replace-with-fixture-password@127.0.0.1:1/disposable"
    no_opt_in = run_cli(
        "restore", "--target", "127.0.0.1:1/disposable", "--archive", artifact,
        "--sha256", "a" * 64, url=url,
    )
    assert no_opt_in.returncode == 2
    assert "--apply" in no_opt_in.stderr
    wrong_target = run_cli(
        "backup", "--quiesced", "--target", "127.0.0.1:1/another", "--archive", artifact, url=url
    )
    assert wrong_target.returncode == 2
    assert "does not match" in wrong_target.stderr
    missing_url = run_cli(
        "backup", "--quiesced", "--target", "127.0.0.1:1/disposable", "--archive", artifact,
        url=None,
    )
    assert "Explicit POSTGRESQL_URL" in missing_url.stderr
    assert "replace-with-fixture-password" not in wrong_target.stderr
    assert not artifact.exists()


def test_restore_rejects_an_archive_with_a_mismatched_checksum(tmp_path):
    artifact = tmp_path / "untrusted.dump"
    artifact.write_bytes(b"synthetic corrupt archive")
    result = run_cli(
        "restore", "--apply", "--target", "127.0.0.1:1/isolated", "--archive", artifact,
        "--sha256", "a" * 64, url="postgresql://test:test@127.0.0.1:1/isolated",
    )
    assert result.returncode == 1
    assert "SHA-256 does not match" in result.stderr


def test_postgresql_backup_restore_and_rollback_validation(tmp_path):
    url = os.getenv("MEDEVIDENCE_TEST_POSTGRES_URL")
    if not url or not all(shutil.which(tool) for tool in ("pg_dump", "pg_restore", "psql")):
        pytest.skip("A disposable PostgreSQL URL and matching client utilities are required")
    original = urlsplit(url)
    dbs = [f"preview_{uuid4().hex}" for _ in range(2)]
    archive = tmp_path / "database.dump"
    filesystem = tmp_path / "snapshot"
    for subdir in ("uploads", "raw", "chroma"):
        (filesystem / subdir).mkdir(parents=True)
    (filesystem / "chroma/chroma.sqlite3").touch()

    def target(database):
        candidate = original._replace(path=f"/{database}")
        return urlunsplit(candidate), f"{candidate.hostname}:{candidate.port}/{database}"

    def checked_tool(*args, url):
        completed = run_cli(*args, url=url)
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    with psycopg.connect(
        url.replace("postgresql+asyncpg://", "postgresql://"), autocommit=True
    ) as control:
        created = []
        try:
            for db in dbs:
                control.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
                created.append(db)
            source, source_name = target(dbs[0])
            destination, dest_name = target(dbs[1])
            env = dict(os.environ, POSTGRESQL_URL=source)
            subprocess.run(
                [
                    sys.executable, "-m", "med_evidence_agentic_rag.persistence.migrations",
                    "upgrade",
                ],
                env=env, check=True, capture_output=True, timeout=90,
            )
            with psycopg.connect(source.replace("+asyncpg", "", 1), autocommit=True) as conn:
                conn.execute("CREATE TABLE synthetic_receipts (id integer)")
                conn.execute("INSERT INTO synthetic_receipts VALUES (7)")
            digest = checked_tool(
                "backup", "--quiesced", "--target", source_name, "--archive", archive, url=source
            ).split()[-1]
            occupied = run_cli(
                "restore", "--apply", "--target", source_name, "--archive", archive,
                "--sha256", digest, url=source,
            )
            assert occupied.returncode == 1
            assert "empty database" in occupied.stderr
            checked_tool(
                "restore", "--apply", "--target", dest_name, "--archive", archive,
                "--sha256", digest, url=destination,
            )
            checked_tool(
                "validate-rollback", "--target", dest_name, "--archive", archive,
                "--sha256", digest, "--filesystem-snapshot", filesystem, url=destination,
            )
            with psycopg.connect(destination.replace("+asyncpg", "", 1)) as conn:
                assert conn.execute("SELECT id FROM synthetic_receipts").fetchone() == (7,)
        finally:
            for db in reversed(created):
                control.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(db))
                )
