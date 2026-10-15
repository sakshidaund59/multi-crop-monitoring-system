"""Initial schema migration with PostGIS support

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-07 15:58:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

    # 2. Create tables
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=False),
        sa.Column(
            'role',
            sa.Enum('farmer', 'agronomist', 'admin', name='user_role'),
            nullable=False,
            server_default='farmer'
        ),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table(
        'crops',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('scientific_name', sa.String(length=150), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_crops_name'), 'crops', ['name'], unique=True)

    op.create_table(
        'farms',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('location', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_farms_owner_id'), 'farms', ['owner_id'], unique=False)

    op.create_table(
        'fields',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('crop_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('crops.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('boundary', geoalchemy2.types.Geometry(geometry_type='POLYGON', srid=4326, spatial_index=True), nullable=True),
        sa.Column('area_ha', sa.Float(), nullable=True),
        sa.Column('sowing_date', sa.Date(), nullable=True),
        sa.Column('growth_stage', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_fields_farm_id'), 'fields', ['farm_id'], unique=False)
    op.create_index(op.f('ix_fields_crop_id'), 'fields', ['crop_id'], unique=False)

    op.create_table(
        'pest_diseases',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('crop_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('crops.id', ondelete='SET NULL'), nullable=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('category', sa.Enum('pest', 'disease', name='pest_disease_category'), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_pest_diseases_crop_id'), 'pest_diseases', ['crop_id'], unique=False)

    op.create_table(
        'detections',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('field_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('fields.id', ondelete='CASCADE'), nullable=False),
        sa.Column('crop_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('crops.id', ondelete='SET NULL'), nullable=True),
        sa.Column('pest_disease_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('pest_diseases.id', ondelete='SET NULL'), nullable=True),
        sa.Column('detection_type', sa.Enum('pest', 'disease', name='detection_type'), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('severity', sa.Enum('low', 'medium', 'high', name='severity_level'), nullable=False),
        sa.Column('density_count', sa.Integer(), nullable=True),
        sa.Column('is_unknown', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('image_path', sa.String(length=500), nullable=False),
        sa.Column('location', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=True), nullable=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_detections_field_id'), 'detections', ['field_id'], unique=False)
    op.create_index(op.f('ix_detections_crop_id'), 'detections', ['crop_id'], unique=False)
    op.create_index(op.f('ix_detections_pest_disease_id'), 'detections', ['pest_disease_id'], unique=False)
    op.create_index(op.f('ix_detections_detected_at'), 'detections', ['detected_at'], unique=False)

    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('field_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('fields.id', ondelete='CASCADE'), nullable=False),
        sa.Column('detection_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detections.id', ondelete='SET NULL'), nullable=True),
        sa.Column('level', sa.Enum('info', 'warning', 'critical', name='alert_level'), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('status', sa.Enum('open', 'acknowledged', 'resolved', name='alert_status'), nullable=False, server_default='open'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_alerts_field_id'), 'alerts', ['field_id'], unique=False)
    op.create_index(op.f('ix_alerts_detection_id'), 'alerts', ['detection_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_alerts_detection_id'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_field_id'), table_name='alerts')
    op.drop_table('alerts')

    op.drop_index(op.f('ix_detections_detected_at'), table_name='detections')
    op.drop_index(op.f('ix_detections_pest_disease_id'), table_name='detections')
    op.drop_index(op.f('ix_detections_crop_id'), table_name='detections')
    op.drop_index(op.f('ix_detections_field_id'), table_name='detections')
    op.drop_table('detections')

    op.drop_index(op.f('ix_pest_diseases_crop_id'), table_name='pest_diseases')
    op.drop_table('pest_diseases')

    op.drop_index(op.f('ix_fields_crop_id'), table_name='fields')
    op.drop_index(op.f('ix_fields_farm_id'), table_name='fields')
    op.drop_table('fields')

    op.drop_index(op.f('ix_farms_owner_id'), table_name='farms')
    op.drop_table('farms')

    op.drop_index(op.f('ix_crops_name'), table_name='crops')
    op.drop_table('crops')

    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    op.execute("DROP TYPE IF EXISTS alert_status;")
    op.execute("DROP TYPE IF EXISTS alert_level;")
    op.execute("DROP TYPE IF EXISTS severity_level;")
    op.execute("DROP TYPE IF EXISTS detection_type;")
    op.execute("DROP TYPE IF EXISTS pest_disease_category;")
    op.execute("DROP TYPE IF EXISTS user_role;")
