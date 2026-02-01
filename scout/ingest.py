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
