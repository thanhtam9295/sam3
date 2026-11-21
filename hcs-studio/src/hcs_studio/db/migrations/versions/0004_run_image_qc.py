"""image QC summary of runs (Phase 18)

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-21 06:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.add_column(sa.Column("image_qc", sa.JSON(), nullable=False, server_default="{}"))


def downgrade() -> None:
    with op.batch_alter_table("run") as batch:
        batch.drop_column("image_qc")
