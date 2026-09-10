"""scout-ingest: build or refresh the embedding index for the docs corpus."""
import argparse
import logging
import sys

from scout.config import load_config
from scout.embeddings import EmbeddingModel
from scout.index import compute_corpus_hash, load_index, save_index
from scout.ingest import chunk_docs, load_markdown_docs
from scout.logsetup import CLI_FORMAT, configure_logging

logger = logging.getLogger("scout.cli")


def _build_parser(config):
    parser = argparse.ArgumentParser(description="Scout ingestion CLI")
    parser.add_argument(
        "path",
        nargs="?",
        default=config["docs_path"],
        help="Path to folder containing markdown docs",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild embeddings even if the index is up to date",
    )
    return parser


def _index_is_current(corpus_hash):
    cached = load_index()
    return bool(cached) and cached[2] == corpus_hash


def build_index(path, rebuild=False):
    """Embed every markdown chunk under `path` unless the cache matches.
    Returns the number of chunks in the index."""
    config = load_config()
    docs = chunk_docs(load_markdown_docs(path))
    if not docs:
        logger.error("no markdown documents found under '%s'", path)
        return 0

    logger.info("loaded %d chunks from %s", len(docs), path)
    corpus_hash = compute_corpus_hash(docs)
    if not rebuild and _index_is_current(corpus_hash):
        logger.info("index is up to date; nothing to rebuild")
        return len(docs)

    logger.info("building embeddings index...")
    embeddings = EmbeddingModel(config["vector_model"]).encode(
        [d["content"] for d in docs]
    )
    save_index(embeddings, docs, corpus_hash)
    logger.info("finished ingesting %d doc chunks", len(docs))
    return len(docs)


def main():
    config = load_config()
    configure_logging(config["log_level"], fmt=CLI_FORMAT)
    args = _build_parser(config).parse_args()
    # Exit non-zero on an empty corpus, so a build step catches it rather
    # than shipping an empty index that looks like a working one.
    return 0 if build_index(args.path, rebuild=args.rebuild) else 1


if __name__ == "__main__":
    sys.exit(main())
