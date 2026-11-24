"""All tables (import every model module here so Alembic sees the full schema)."""

from hcs_studio.db import (  # noqa: F401
    assistant,
    automation,
    catalog,
    model_registry,
    models,
    security,
    server,
)
from hcs_studio.db.base import Base

metadata = Base.metadata
