"""Alembic environment for the registry (run through hcs_studio.db.engine.upgrade)."""

from __future__ import annotations

from alembic import context

from hcs_studio.db.schema import metadata

config = context.config


def run() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        engine = config.attributes.get("engine")
        if engine is None:
            raise RuntimeError("Run migrations through hcs_studio.db.engine.upgrade()")
        with engine.begin() as conn:
            context.configure(connection=conn, target_metadata=metadata, render_as_batch=True)
            with context.begin_transaction():
                context.run_migrations()
        return
    context.configure(connection=connection, target_metadata=metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


run()
