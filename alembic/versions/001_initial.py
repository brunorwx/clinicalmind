# alembic/versions/001_initial.py
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    op.create_table("trials",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("status", sa.String, default="active"),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table("patients",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("trial_id", sa.String, sa.ForeignKey("trials.id")),
        sa.Column("external_id", sa.String, nullable=False),
        sa.Column("arm", sa.String),
        sa.Column("enrolled_date", sa.DateTime),
        sa.Column("status", sa.String, default="active"),
    )
    op.create_table("adverse_events",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("patient_id", sa.UUID, sa.ForeignKey("patients.id")),
        sa.Column("grade", sa.Integer, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("onset_day", sa.Integer),
        sa.Column("resolved", sa.Boolean, default=False),
    )
    op.create_table("lab_results",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("patient_id", sa.UUID, sa.ForeignKey("patients.id")),
        sa.Column("test_name", sa.String),
        sa.Column("value", sa.Float),
        sa.Column("unit", sa.String),
        sa.Column("collected_at", sa.DateTime),
    )
    op.create_table("documents",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("trial_id", sa.String, sa.ForeignKey("trials.id")),
        sa.Column("filename", sa.String),
        sa.Column("s3_key", sa.String),
        sa.Column("doc_type", sa.String),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table("document_chunks",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", sa.UUID, sa.ForeignKey("documents.id")),
        sa.Column("trial_id", sa.String, nullable=False),
        sa.Column("chunk_index", sa.Integer),
        sa.Column("content", sa.Text),
        sa.Column("token_count", sa.Integer),
        sa.Column("embedding", Vector(1536)),
        sa.Column("meta", sa.JSON),
    )
    op.create_table("audit_logs",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", sa.String),
        sa.Column("trial_id", sa.String),
        sa.Column("question", sa.Text),
        sa.Column("agents_invoked", sa.ARRAY(sa.String)),
        sa.Column("tools_used", sa.ARRAY(sa.String)),
        sa.Column("chunk_ids", sa.ARRAY(sa.String)),
        sa.Column("created_at", sa.DateTime),
    )
    op.create_table("review_flags",
        sa.Column("id", sa.UUID, primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("trial_id", sa.String),
        sa.Column("user_id", sa.String),
        sa.Column("question", sa.Text),
        sa.Column("reason", sa.Text),
        sa.Column("priority", sa.String),
        sa.Column("resolved", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime),
    )
    # HNSW index for fast ANN search
    op.execute("""
        CREATE INDEX ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX ON document_chunks (trial_id)")

def downgrade():
    for t in ["review_flags","audit_logs","document_chunks","documents",
              "lab_results","adverse_events","patients","trials"]:
        op.drop_table(t)