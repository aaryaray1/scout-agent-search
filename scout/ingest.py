from pathlib import Path
import uuid

def load_markdown_docs(path="data/docs"):

    docs = []
    for file in Path(path).glob("*.md"):
        text = file.read_text(encoding="utf-8")
        docs.append({
            "source": file.name,
            "content": text,
            # Default type
            "type": "documentation"
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
        # Generate unique ID 
        if "id" not in doc:
            doc["id"] = str(uuid.uuid4())

        # Ensure 'type' exists
        doc_type = doc.get("type", "documentation")

        # Chunk content
        for chunk in chunk_text(doc["content"]):
            chunks.append({
                "id": doc["id"],
                "content": chunk,
                "source": doc["source"],
                "type": doc_type
            })

    return chunks
