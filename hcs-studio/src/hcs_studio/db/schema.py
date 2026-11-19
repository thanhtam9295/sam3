"""All tables (import every model module here so Alembic sees the full schema)."""

from hcs_studio.db import catalog, models  # noqa: F401
from hcs_studio.db.base import Base

metadata = Base.metadata
