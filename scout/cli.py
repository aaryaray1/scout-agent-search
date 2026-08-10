import argparse
from scout.index import load_index, save_index, compute_corpus_hash
from scout.ingest import load_markdown_docs, chunk_docs
from scout.embeddings import EmbeddingModel
from scout.config import load_config

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Scout ingestion CLI")
    parser.add_argument(
        "path",
        help="Path to folder containing markdown docs",
        default=config["docs_path"],
        nargs="?"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild embeddings even if index exists"
    )
    args = parser.parse_args()

    print(f"Loading markdown docs from: {args.path}")
    raw_docs = load_markdown_docs(args.path)
    print(f"Loaded {len(raw_docs)} markdown docs")
    docs = chunk_docs(raw_docs)
    print(f"Split into {len(docs)} chunks")

    embed_model = EmbeddingModel(config["vector_model"])
    current_hash = compute_corpus_hash(docs)

    cached = load_index()
    if cached:
        _embeddings, _metadata, cached_hash = cached
        if not args.rebuild and cached_hash == current_hash:
            print("Index is up to date. No rebuild needed.")
            return

    print("Building embeddings index...")
    embeddings = embed_model.encode([d["content"] for d in docs])
    print("Saving embeddings to disk...")
    save_index(embeddings, docs, current_hash)
    print(f"Finished ingesting {len(docs)} doc chunks.")


if __name__ == "__main__":
    main()
