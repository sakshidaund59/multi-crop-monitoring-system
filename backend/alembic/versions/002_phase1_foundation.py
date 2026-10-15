"""Phase 1 foundation models: model_registry, analyses, pest_detections, disease_results, severity_results, weather_snapshots, ndvi_snapshots, risk_results, chat_messages, and farm/alert updates

Revision ID: 002_phase1_foundation
Revises: 001_initial_schema
Create Date: 2026-10-14 15:05:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import geoalchemy2


# revision identifiers, used by Alembic.
revision: str = '002_phase1_foundation'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update farms table with boundary polygon and area_ha if not present
    op.add_column(
        'farms',
        sa.Column('boundary', geoalchemy2.types.Geometry(geometry_type='POLYGON', srid=4326, spatial_index=True), nullable=True)
    )
    op.add_column('farms', sa.Column('area_ha', sa.Float(), nullable=True))

    # 2. Update alerts table with farm_id, alert_type, title, is_resolved
    op.add_column('alerts', sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=True))
    op.add_column('alerts', sa.Column('alert_type', sa.String(length=50), server_default='pest_disease', nullable=False))
    op.add_column('alerts', sa.Column('title', sa.String(length=200), server_default='Alert', nullable=False))
    op.add_column('alerts', sa.Column('is_resolved', sa.Boolean(), server_default='false', nullable=False))
    op.create_index(op.f('ix_alerts_farm_id'), 'alerts', ['farm_id'], unique=False)

    # 3. Create model_registry table
    op.create_table(
        'model_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('weights_hash', sa.String(length=128), nullable=False),
        sa.Column('dataset_version', sa.String(length=100), nullable=False),
        sa.Column('metrics_path', sa.String(length=500), nullable=True),
        sa.Column('trained_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_model_registry_name'), 'model_registry', ['name'], unique=False)

    # 4. Create analyses table (referencing model_registry for provenance!)
    op.create_table(
        'analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('model_registry.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('image_path', sa.String(length=500), nullable=True),
        sa.Column('analysis_type', sa.String(length=50), server_default='disease', nullable=False),
        sa.Column('status', sa.String(length=50), server_default='completed', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_analyses_farm_id'), 'analyses', ['farm_id'], unique=False)
    op.create_index(op.f('ix_analyses_user_id'), 'analyses', ['user_id'], unique=False)
    op.create_index(op.f('ix_analyses_model_id'), 'analyses', ['model_id'], unique=False)

    # 5. Create pest_detections table
    op.create_table(
        'pest_detections',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('pest_name', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('density_count', sa.Integer(), nullable=True),
        sa.Column('severity', sa.String(length=50), server_default='medium', nullable=False),
        sa.Column('location', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, spatial_index=True), nullable=True),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_pest_detections_analysis_id'), 'pest_detections', ['analysis_id'], unique=False)
    op.create_index(op.f('ix_pest_detections_detected_at'), 'pest_detections', ['detected_at'], unique=False)

    # 6. Create disease_results table
    op.create_table(
        'disease_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('disease_name', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('symptoms', sa.Text(), nullable=True),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_disease_results_analysis_id'), 'disease_results', ['analysis_id'], unique=False)

    # 7. Create severity_results table
    op.create_table(
        'severity_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('affected_area_pct', sa.Float(), nullable=False),
        sa.Column('severity_level', sa.String(length=50), server_default='medium', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_severity_results_analysis_id'), 'severity_results', ['analysis_id'], unique=False)

    # 8. Create weather_snapshots table
    op.create_table(
        'weather_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False),
        sa.Column('humidity', sa.Float(), nullable=False),
        sa.Column('precipitation', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('wind_speed', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('raw_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(op.f('ix_weather_snapshots_farm_id'), 'weather_snapshots', ['farm_id'], unique=False)
    op.create_index(op.f('ix_weather_snapshots_recorded_at'), 'weather_snapshots', ['recorded_at'], unique=False)

    # 9. Create ndvi_snapshots table
    op.create_table(
        'ndvi_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('mean_ndvi', sa.Float(), nullable=False),
        sa.Column('min_ndvi', sa.Float(), nullable=True),
        sa.Column('max_ndvi', sa.Float(), nullable=True),
        sa.Column('cloud_coverage', sa.Float(), nullable=True),
        sa.Column('tile_date', sa.Date(), nullable=False),
        sa.Column('image_path', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_ndvi_snapshots_farm_id'), 'ndvi_snapshots', ['farm_id'], unique=False)
    op.create_index(op.f('ix_ndvi_snapshots_tile_date'), 'ndvi_snapshots', ['tile_date'], unique=False)

    # 10. Create risk_results table
    op.create_table(
        'risk_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('farm_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('farms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('risk_type', sa.String(length=100), nullable=False),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(length=50), server_default='medium', nullable=False),
        sa.Column('advisory', sa.Text(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_risk_results_farm_id'), 'risk_results', ['farm_id'], unique=False)
    op.create_index(op.f('ix_risk_results_evaluated_at'), 'risk_results', ['evaluated_at'], unique=False)

    # 11. Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index(op.f('ix_chat_messages_user_id'), 'chat_messages', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('chat_messages')
    op.drop_table('risk_results')
    op.drop_table('ndvi_snapshots')
    op.drop_table('weather_snapshots')
    op.drop_table('severity_results')
    op.drop_table('disease_results')
    op.drop_table('pest_detections')
    op.drop_table('analyses')
    op.drop_table('model_registry')
    op.drop_column('alerts', 'is_resolved')
    op.drop_column('alerts', 'title')
    op.drop_column('alerts', 'alert_type')
    op.drop_column('alerts', 'farm_id')
    op.drop_column('farms', 'area_ha')
    op.drop_column('farms', 'boundary')
