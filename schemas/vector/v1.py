"""JSON schema definition for vector records (v1)."""

from core.models import VectorRecord

VECTOR_RECORD_SCHEMA = VectorRecord.model_json_schema()

__all__ = ["VECTOR_RECORD_SCHEMA"]
