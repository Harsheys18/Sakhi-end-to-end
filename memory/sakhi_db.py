"""Compatibility wrapper for legacy imports.

Re-exports the PostgreSQL helpers from memory/db.py.
"""

from db import connect, init_db, put_in, get_from

__all__ = ["connect", "init_db", "put_in", "get_from"]
