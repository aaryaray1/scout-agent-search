"""Durable storage backends for ingested pages.

See base.py for the contract and why it stops short of being a
vector-search interface.
"""
from .base import ChunkStore, JsonChunkStore, StoreStats

__all__ = ["ChunkStore", "JsonChunkStore", "StoreStats"]
