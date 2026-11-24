"""Back up, restore or validate a PostgreSQL deployment with explicit target selection."""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"[a-z][a-z0-9_]{0,62}", re.ASCII)
HOST = re.compile(r"[A-Za-z0-9.-]+", re.ASCII)
SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
EMPTY_DATABASE = """
SELECT NOT (
    EXISTS (SELECT FROM pg_stat_activity
        WHERE datname = current_database() AND pid <> pg_backend_pid())
    OR EXISTS (SELECT FROM pg_namespace
        WHERE nspname !~ '^pg_' AND nspname NOT IN ('public', 'information_schema'))
    OR EXISTS (SELECT FROM pg_class WHERE relnamespace = 'public'::regnamespace)
    OR EXISTS (SELECT FROM pg_proc WHERE pronamespace = 'public'::regnamespace)
    OR EXISTS (SELECT FROM pg_type WHERE typnamespace = 'public'::regnamespace)
    OR EXISTS (SELECT FROM pg_extension WHERE extname <> 'plpgsql')
) AND NOT pg_is_in_recovery()
"""


class MaintenanceError(RuntimeError):
    """A safe message that carries no operator values."""


def target_environment(target: str) -> dict[str, str]:
    raw = os.environ.get("POSTGRESQL_URL")
    if not raw:
        raise MaintenanceError("Explicit POSTGRESQL_URL process environment is required.")
    try:
        url = urlsplit(raw)
        host, port, user = url.hostname, url.port, url.username
        database = url.path.removeprefix("/")
        options = parse_qsl(url.query, strict_parsing=True)
        if (
            url.scheme not in {"postgresql", "postgresql+asyncpg"}
            or url.fragment
            or not host
            or not HOST.fullmatch(host)
            or not port
            or not user
            or not url.password
            or not NAME.fullmatch(database)
            or database in {"postgres", "template0", "template1"}
            or any(
                key != "sslmode" or value not in {"require", "verify-ca", "verify-full", "disable"}
                for key, value in options
            )
            or len(options) > 1
        ):
            raise ValueError
    except ValueError:
        raise MaintenanceError(
            "Use a dedicated database URL with a single explicit DNS/IPv4 host, port, "
            "username, password and DB name; only sslmode is supported as an option."
        ) from None
    if target != f"{host}:{port}/{database}":
        raise MaintenanceError("--target does not match the name, host and port of POSTGRESQL_URL.")
    environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("PG")
    }
    environment.update(
        PGHOST=host,
        PGPORT=str(port),
        PGDATABASE=database,
        PGUSER=unquote(user),
        PGPASSWORD=unquote(url.password),
        PGCONNECT_TIMEOUT="5",
        PGSSLMODE=dict(options).get("sslmode", "prefer"),
    )
    return environment


def run_tool(environment: dict[str, str], name: str, *args: str) -> str:
    try:
        result = subprocess.run(
            [name, *args], env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3600, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise MaintenanceError(
            f"{name} failed. Check the client version, target permissions and connectivity."
        ) from None
    return result.stdout.decode("utf-8", errors="replace").strip()


def query(environment: dict[str, str], statement: str) -> str:
    return run_tool(environment, "psql", "-XAtw", "-v", "ON_ERROR_STOP=1", "-c", statement)


def archive_path(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve().is_relative_to(ROOT):
        raise MaintenanceError("Archive must be an absolute, external, regular file path.")
    return path.resolve()


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_archive(environment: dict[str, str], path: Path, checksum: str) -> None:
    if not SHA256.fullmatch(checksum) or not path.is_file() or digest(path) != checksum:
        raise MaintenanceError("Archive is missing or its recorded SHA-256 does not match.")
    run_tool(environment, "pg_restore", "--list", str(path))


def backup(environment: dict[str, str], path: Path) -> str:
    if path.exists() or not path.parent.is_dir():
        raise MaintenanceError("Archive must be a new file under an existing external directory.")
    if query(
        environment,
        "SELECT NOT EXISTS (SELECT FROM pg_stat_activity "
        "WHERE datname = current_database() AND pid <> pg_backend_pid())",
    ) != "t":
        raise MaintenanceError("Disconnect other database sessions before backing up.")
    with tempfile.TemporaryDirectory(prefix=".medevidence-backup-", dir=path.parent) as workspace:
        temporary = Path(workspace) / "database.dump"
        run_tool(
            environment, "pg_dump", "--no-password", "--format=custom",
            "--no-owner", "--no-privileges", f"--file={temporary}",
        )
        run_tool(environment, "pg_restore", "--list", str(temporary))
        checksum = digest(temporary)
        # Publish the complete archive without ever replacing an existing file.
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    return checksum


def restore(environment: dict[str, str], path: Path) -> None:
    if query(environment, EMPTY_DATABASE) != "t":
        raise MaintenanceError("Restore requires a dedicated, disconnected and empty database.")
    run_tool(
        environment, "pg_restore", "--no-password", "--single-transaction", "--exit-on-error",
        "--no-owner", "--no-privileges", "--dbname", environment["PGDATABASE"], str(path),
    )


def validate_rollback(
    environment: dict[str, str], filesystem: Path, chroma: Path | None
) -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    if (
        not filesystem.is_absolute()
        or filesystem.resolve().is_relative_to(ROOT)
        or not all((filesystem / name).is_dir() for name in ("uploads", "raw"))
    ):
        raise MaintenanceError("Provide the external snapshot with raw and uploads directories.")
    index = chroma or filesystem / "chroma"
    if (
        not index.is_absolute()
        or index.resolve().is_relative_to(ROOT)
        or not (index / "chroma.sqlite3").is_file()
    ):
        raise MaintenanceError("Provide the matching external Chroma snapshot.")
    revision = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini"))).get_current_head()
    if query(environment, "SELECT version_num FROM public.alembic_version") != revision:
        raise MaintenanceError("Restored database revision does not match this application's head.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("backup", "restore", "validate-rollback"):
        sub = subcommands.add_parser(operation)
        sub.add_argument("--target", required=True, metavar="HOST:PORT/DATABASE")
        sub.add_argument("--archive", type=Path, required=True)
        if operation == "backup":
            sub.add_argument("--quiesced", action="store_true", required=True)
        else:
            sub.add_argument("--sha256", required=True)
            if operation == "restore":
                sub.add_argument("--apply", action="store_true", required=True)
            else:
                sub.add_argument("--filesystem-snapshot", type=Path, required=True)
                sub.add_argument("--chroma-snapshot", type=Path)
    args = parser.parse_args()
    try:
        environment = target_environment(args.target)
        path = archive_path(args.archive)
    except (OSError, MaintenanceError) as error:
        parser.error(str(error) if isinstance(error, MaintenanceError) else "Invalid archive path")
    try:
        if args.operation == "backup":
            print("Archive saved. SHA-256", backup(environment, path))
            return 0
        verify_archive(environment, path, args.sha256)
        if args.operation == "restore":
            restore(environment, path)
            print("Archive restored to the empty database.")
        else:
            validate_rollback(environment, args.filesystem_snapshot, args.chroma_snapshot)
            print("Archive checksum, snapshot layout and database revision verified.")
    except MaintenanceError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Maintenance failed. Check external storage and release assets.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
