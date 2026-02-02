from pathlib import Path

def load_markdown_docs(path="data/docs"):
    docs = []
    for file in Path(path).glob("*.md"):
        text = file.read_text(encoding="utf-8")
        docs.append({
            "source": file.name,
            "content": text
        })
    return docs

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

def chunk_docs(docs):
    chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["content"]):
            chunks.append({
                "id": doc["id"],
                "content": chunk,
                "source": doc["source"],
                "type": doc["type"]
            })
    return chunks
