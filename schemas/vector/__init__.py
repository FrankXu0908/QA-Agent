"""Versioned vector record schemas."""

from importlib import import_module
from typing import Any, Dict

_SCHEMAS = {
    "v1": "schemas.vector.v1:VECTOR_RECORD_SCHEMA",
}


def get_schema(version: str = "v1") -> Dict[str, Any]:
    """Return the JSON schema for the requested vector record version."""

    target = _SCHEMAS.get(version)
    if not target:
        raise ValueError(f"Unsupported schema version: {version}")
    module_path, attr = target.split(":")
    module = import_module(module_path)
    return getattr(module, attr)


__all__ = ["get_schema"]
