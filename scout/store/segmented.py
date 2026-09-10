"""Append-only segmented storage for ingested pages.

An upsert writes one immutable segment plus a manifest entry, so it costs the
page rather than the store. Crash safety is the write ordering: segments
first, manifest last. See docs/design/storage.md.
"""
import logging
import os

import numpy as np

from ..index import (
    IndexCorruptError,
    load_ingested,
    read_json,
    read_npy,
    resolve_index_dir,
    write_json,
    write_npy,
)
from .base import ChunkStore, StoreStats, check_alignment, source_counts

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
# Bumped only for a change an older Scout could not read correctly, since
# misreading a store means serving the wrong chunks.
MANIFEST_FORMAT = 1

# A merge costs the size of the live set, so it has to buy back more than it
# costs: a real share of the store, and more than a handful of chunks.
_COMPACT_DEAD_RATIO = 0.3
_COMPACT_MIN_DEAD = 64
# A load opens two files per segment, so segments are capped independently of
# how much is dead. Merging here spreads one rewrite across that many ingests.
_COMPACT_MAX_SEGMENTS = 32


class SegmentedChunkStore(ChunkStore):
    """Ingested pages as an append-only run of segments plus a manifest.

    Nothing is held in memory between calls except the manifest: superseding
    a source is a line in it rather than a rewrite.
    """

    def __init__(self, index_dir=None):
        self._index_dir = index_dir
        # None until the manifest is read; [] means read and empty.
        self._entries = None
        self._next_id = 1
        # source -> (entry position, chunk count): which segment holds the
        # live version of that source.
        self._live = {}
        # Chunks physically present in segments on disk, live or not.
        self._written = 0

    # -- layout --------------------------------------------------------------

    @property
    def _dir(self):
        # Resolved per call so a monkeypatched INDEX_DIR redirects this too.
        return os.path.join(resolve_index_dir(self._index_dir), "ingested")

    def _path(self, name, ext):
        return os.path.join(self._dir, name + ext)

    # -- manifest ------------------------------------------------------------

    def _ensure_loaded(self):
        """Read the manifest before anything that depends on what is stored.

        An instance that has not read it knows nothing, and a store acting on
        that belief supersedes sources it cannot see.
        """
        if self._entries is not None:
            return
        manifest = self._read_manifest()
        if manifest is None:
            self._entries = []
            self._adopt_legacy_store()
        else:
            self._entries = manifest["entries"]
            self._next_id = manifest["next_id"]
            self._reindex()
        self._discard_unreferenced()

    def _read_manifest(self):
        """Return the manifest dict, or None when this store is new here."""
        path = os.path.join(self._dir, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        manifest = read_json(path, recoverable=False)
        # Valid JSON of the wrong shape is corruption, not an empty store:
        # reading it as empty supersedes pages still sitting beside it.
        if not isinstance(manifest, dict) or "entries" not in manifest:
            raise IndexCorruptError(f"'{path}' is not a Scout store manifest")
        if manifest.get("format") != MANIFEST_FORMAT:
            raise IndexCorruptError(
                f"'{path}' is manifest format {manifest.get('format')!r}, this "
                f"Scout reads format {MANIFEST_FORMAT}"
            )
        return manifest

    def _flush_manifest(self):
        write_json(
            os.path.join(self._dir, MANIFEST_NAME),
            {
                "format": MANIFEST_FORMAT,
                "next_id": self._next_id,
                "entries": self._entries,
            },
            indent=None,
        )

    def _reindex(self):
        """Replay the manifest into the live-source map and chunk count.

        Cheap to redo after every append: a merge bounds the entry list, and
        entries hold source names rather than chunks.
        """
        live, written = {}, 0
        for position, entry in enumerate(self._entries):
            sources = entry.get("sources") or {}
            # An upsert supersedes exactly the sources it carries, so it
            # stores no separate list; only a deletion needs one.
            for source in entry.get("removes", sources):
                live.pop(source, None)
            for source, count in sources.items():
                live[source] = (position, count)
                written += count
        self._live = live
        self._written = written

    def _adopt_legacy_store(self):
        """Import a pre-segment ingested store, if one is sitting there.

        Without this an upgrade starts empty and lets the next ingest
        supersede the only copy of every page already stored. The old files
        are left in place, so the upgrade stays reversible.
        """
        stored = load_ingested(self._index_dir)
        if not stored:
            return
        embeddings, chunks = stored
        logger.info(
            "importing %d chunk(s) from the pre-segment ingested store", len(chunks)
        )
        self.upsert(chunks, embeddings)

    def _discard_unreferenced(self):
        """Delete files the manifest doesn't name: debris from a crash
        between a segment write and the manifest write that would have
        committed it, holding nothing a reader could have seen."""
        if not os.path.isdir(self._dir):
            return
        referenced = {entry.get("segment") for entry in self._entries}
        for name in os.listdir(self._dir):
            if name == MANIFEST_NAME or os.path.splitext(name)[0] in referenced:
                continue
            self._discard(name)

    def _discard(self, name):
        """Remove one uncommitted file, treating failure as untidiness: it is
        already invisible to readers, so a read-only mount warns rather than
        making the store unopenable."""
        path = os.path.join(self._dir, name)
        logger.info("discarding uncommitted store file '%s'", name)
        try:
            os.unlink(path)
        except OSError as e:
            logger.warning("could not remove '%s': %s", path, e)

    # -- reading -------------------------------------------------------------

    def load(self):
        self._ensure_loaded()
        chunks, blocks = [], []
        for position, entry in enumerate(self._entries):
            part = self._read_live_part(position, entry)
            if part is None:
                continue
            chunks.extend(part[0])
            blocks.append(part[1])
        if not chunks:
            return [], None
        return chunks, np.vstack(blocks)

    def _read_live_part(self, position, entry):
        """Return this entry's still-live (chunks, embeddings), or None."""
        name = entry.get("segment")
        if name is None:  # a deletion: it removes, it stores nothing
            return None
        live = self._live_sources_in(position, entry)
        if not live:
            return None
        return self._read_segment_filtered(name, live, len(entry["sources"]))

    def _live_sources_in(self, position, entry):
        """Which of this entry's sources it still holds the live version of.

        Anything the live map points elsewhere for was superseded later, and
        is dead weight waiting for a merge.
        """
        return {
            source
            for source in entry["sources"]
            if self._live.get(source, (None,))[0] == position
        }

    def _read_segment_filtered(self, name, live, total_sources):
        """Read a segment, dropping the sources it no longer owns.

        The whole-segment case is separate because it is the common one and
        avoids building a mask the length of the segment.
        """
        chunks, embeddings = self._read_segment(name)
        if len(live) == total_sources:
            return chunks, embeddings
        keep = np.array([c["source"] in live for c in chunks], dtype=bool)
        return [c for c, alive in zip(chunks, keep) if alive], embeddings[keep]

    def _read_segment(self, name):
        """Read one segment, refusing it if its two halves disagree -- a
        drifted segment scores every chunk against the wrong vector."""
        chunks = read_json(self._path(name, ".json"), recoverable=False)
        embeddings = read_npy(self._path(name, ".npy"))
        if not isinstance(chunks, list) or len(embeddings) != len(chunks):
            raise IndexCorruptError(
                f"segment '{name}' is inconsistent: {len(embeddings)} embeddings "
                f"for {len(chunks) if isinstance(chunks, list) else '?'} chunks"
            )
        return chunks, embeddings

    # -- writing -------------------------------------------------------------

    def upsert(self, chunks, embeddings):
        if not len(chunks):
            return
        check_alignment(chunks, embeddings)
        self._ensure_loaded()
        counts = source_counts(chunks)
        self._append(self._write_segment(chunks, embeddings, counts))

    def delete_sources(self, sources):
        self._ensure_loaded()
        # Only sources actually held: an entry naming nothing live changes
        # nothing and is the one way this file could grow without bound.
        removes = sorted(s for s in set(sources) if s in self._live)
        if not removes:
            return
        self._append({"segment": None, "removes": removes})

    def _write_segment(self, chunks, embeddings, counts):
        """Write one immutable segment and return the entry that commits it.

        Returned rather than appended so nothing references the segment
        until it is fully on disk.
        """
        name = f"seg-{self._next_id:06d}"
        self._next_id += 1
        os.makedirs(self._dir, exist_ok=True)
        write_npy(self._path(name, ".npy"), np.asarray(embeddings))
        write_json(self._path(name, ".json"), list(chunks), indent=None)
        return {"segment": name, "sources": counts}

    def _append(self, entry):
        """Commit one manifest entry, then merge if that made it worth it."""
        self._entries.append(entry)
        self._reindex()
        self._flush_manifest()
        self._compact_if_worthwhile()

    # -- compaction ----------------------------------------------------------

    def _segment_count(self):
        return sum(1 for entry in self._entries if entry.get("segment"))

    def _live_chunks(self):
        return sum(count for _position, count in self._live.values())

    def _worth_compacting(self):
        if self._segment_count() > _COMPACT_MAX_SEGMENTS:
            return True
        dead = self._written - self._live_chunks()
        return dead >= _COMPACT_MIN_DEAD and dead >= _COMPACT_DEAD_RATIO * self._written

    def _compact_if_worthwhile(self):
        """Merge the live set into one segment and drop everything else.

        Same ordering as every other write: new segment, then the manifest
        naming it, then the files it replaced.
        """
        if not self._worth_compacting():
            return
        superseded = [e["segment"] for e in self._entries if e.get("segment")]
        chunks, embeddings = self.load()

        self._entries = (
            [self._write_segment(chunks, embeddings, source_counts(chunks))]
            if chunks
            else []
        )
        self._reindex()
        self._flush_manifest()
        self._discard_unreferenced()
        logger.info(
            "merged %d store segment(s) into %d, holding %d chunk(s)",
            len(superseded), self._segment_count(), len(chunks),
        )

    # -- reporting -----------------------------------------------------------

    def stats(self):
        self._ensure_loaded()
        live = self._live_chunks()
        return StoreStats(
            live_chunks=live,
            sources=len(self._live),
            dead_chunks=self._written - live,
            segments=self._segment_count(),
        )
