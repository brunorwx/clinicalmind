# app/models/db.py
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime,
    ForeignKey, JSON, Boolean, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base
import enum

class TrialStatus(str, enum.Enum):
    active = "active"
    completed = "completed"
    suspended = "suspended"

class Trial(Base):
    __tablename__ = "trials"
    id          = Column(String, primary_key=True)   # e.g. "TRIAL-ONC2024"
    name        = Column(String, nullable=False)
    status      = Column(SAEnum(TrialStatus), default=TrialStatus.active)
    created_at  = Column(DateTime, default=datetime.utcnow)
    patients    = relationship("Patient", back_populates="trial")
    documents   = relationship("Document", back_populates="trial")

class Patient(Base):
    __tablename__ = "patients"
    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id        = Column(String, ForeignKey("trials.id"), nullable=False)
    external_id     = Column(String, nullable=False)   # e.g. "P1042"
    arm             = Column(String)                   # "treatment" | "control"
    enrolled_date   = Column(DateTime)
    status          = Column(String, default="active")
    trial           = relationship("Trial", back_populates="patients")
    adverse_events  = relationship("AdverseEvent", back_populates="patient")
    lab_results     = relationship("LabResult", back_populates="patient")

class AdverseEvent(Base):
    __tablename__ = "adverse_events"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id  = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    grade       = Column(Integer, nullable=False)          # 1-5 (CTCAE)
    description = Column(Text, nullable=False)
    onset_day   = Column(Integer)
    resolved    = Column(Boolean, default=False)
    patient     = relationship("Patient", back_populates="adverse_events")

class LabResult(Base):
    __tablename__ = "lab_results"
    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id      = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    test_name       = Column(String, nullable=False)
    value           = Column(Float, nullable=False)
    unit            = Column(String)
    collected_at    = Column(DateTime, default=datetime.utcnow)
    patient         = relationship("Patient", back_populates="lab_results")

class Document(Base):
    __tablename__ = "documents"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id    = Column(String, ForeignKey("trials.id"), nullable=False)
    filename    = Column(String, nullable=False)
    s3_key      = Column(String, nullable=False)
    doc_type    = Column(String)   # "protocol" | "safety_report" | "ae_log"
    created_at  = Column(DateTime, default=datetime.utcnow)
    trial       = relationship("Trial", back_populates="documents")
    chunks      = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    trial_id    = Column(String, nullable=False)         # denormalized for fast filter
    chunk_index = Column(Integer, nullable=False)
    content     = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding   = Column(Vector(1536))
    meta        = Column(JSON, default={})
    document    = relationship("Document", back_populates="chunks")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(String, nullable=False)
    trial_id        = Column(String, nullable=False)
    question        = Column(Text, nullable=False)
    agents_invoked  = Column(ARRAY(String), default=[])
    tools_used      = Column(ARRAY(String), default=[])
    chunk_ids       = Column(ARRAY(String), default=[])
    created_at      = Column(DateTime, default=datetime.utcnow)

class ReviewFlag(Base):
    __tablename__ = "review_flags"
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id    = Column(String, nullable=False)
    user_id     = Column(String, nullable=False)
    question    = Column(Text, nullable=False)
    reason      = Column(Text, nullable=False)
    priority    = Column(String, default="medium")
    resolved    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)