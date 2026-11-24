"""Explicit business schema maintenance, independent of LangGraph's checkpoint schema."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import cast

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, MetaData, bindparam, inspect, text

from med_evidence_agentic_rag.utils.logger import configure_server_logging, setup_logger

BASELINE = "0001_business"
CONFIG_FILE = Path(__file__).resolve().parents[3] / "alembic.ini"
logger = setup_logger(__name__)


class MigrationError(RuntimeError):
    """An actionable schema error with no database or document values."""


def migrate(connection: Connection, operation: str) -> None:
    config = Config(str(CONFIG_FILE))
    config.attributes["connection"] = connection
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision(BASELINE)
    baseline = cast(MetaData, revision.module.schema)
    if connection.dialect.name == "postgresql" and operation != "check":
        connection.execute(text("SELECT pg_advisory_xact_lock(1719283401, 1)"))
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    current = MigrationContext.configure(connection).get_current_heads()
    checkpoint = {"checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"}
    if operation == "check":
        if current != (scripts.get_current_head(),):
            raise MigrationError(
                "Run python -m med_evidence_agentic_rag.persistence.migrations upgrade"
            )
    elif operation == "adopt-legacy":
        if current or "alembic_version" in tables:
            raise MigrationError("Cannot adopt legacy schema with an existing version table")
        # Checkpoint tables have their own migration history.
        if tables - checkpoint != set(baseline.tables):
            raise MigrationError("Unexpected legacy business tables")

        def business_object(
            obj: object, name: str | None, kind: str, reflected: bool, compare_to: object
        ) -> bool:
            return not (kind == "table" and name in checkpoint)

        ctx = MigrationContext.configure(
            connection,
            opts={
                "compare_server_default": True,
                "include_object": business_object,
            },
        )
        if compare_metadata(ctx, baseline):
            raise MigrationError("Incompatible legacy columns, indexes or constraints")
        for name in baseline.tables:
            if inspector.get_pk_constraint(name)["constrained_columns"] != ["id"]:
                raise MigrationError("Incompatible legacy primary key")
            if inspector.get_check_constraints(name):
                raise MigrationError("Unexpected legacy check constraint")
        # SQLAlchemy reflects the FK definition but omits PostgreSQL's validation
        # flag. Resolve the same visible tables as the inspector on this connection.
        if connection.dialect.name == "postgresql":
            invalid_fk = text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_catalog.pg_constraint AS fk
                    JOIN pg_catalog.pg_class AS relation ON relation.oid = fk.conrelid
                    WHERE fk.contype = 'f' AND NOT fk.convalidated
                    AND pg_catalog.pg_table_is_visible(relation.oid)
                    AND relation.relname IN :tables
                )
            """).bindparams(bindparam("tables", expanding=True))
            if connection.scalar(invalid_fk, {"tables": tuple(baseline.tables)}):
                raise MigrationError("Unvalidated legacy foreign key")
        command.stamp(config, BASELINE)
    elif operation == "upgrade":
        if not current and tables.intersection(baseline.tables):
            raise MigrationError(
                "Existing business schema requires explicit adopt-legacy validation"
            )
        if not current and tables - checkpoint:
            raise MigrationError("Refusing to upgrade an unrecognized unversioned schema")
        command.upgrade(config, "head")
    else:
        raise ValueError("Unknown migration operation")


async def run(operation: str) -> None:
    from med_evidence_agentic_rag.models_and_database.postgres import engine

    try:
        async with engine.begin() as connection:
            await connection.run_sync(migrate, operation)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["check", "upgrade", "adopt-legacy"])
    operation = parser.parse_args().operation
    if not os.getenv("POSTGRESQL_URL"):
        parser.error("Explicit POSTGRESQL_URL process environment is required")
    configure_server_logging()
    try:
        asyncio.run(run(operation))
    except MigrationError as error:
        parser.exit(1, f"{error}\n")
    except Exception:
        logger.exception("Migration command failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
