import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
INPUT_FILE = Path("full_text_ocr.txt")
DB_PATH = "chroma_db"
MODEL_NAME = "nomic-embed-text"

def main():
    load_dotenv()
    if not INPUT_FILE.exists():
        print(f"❌ Error: Run ingest.py first to create {INPUT_FILE}")
        return

    # 1. Load and Parse
    print(f"📑 Loading {INPUT_FILE}...")
    text = INPUT_FILE.read_text(encoding="utf-8")

    # Split by section numbers (e.g., 12.12.010)
    # Using multiline anchor (^) to find section starts at beginning of lines
    section_pattern = r'(?m)^(\d+\.\d+\.\d+)\s+'
    splits = re.split(section_pattern, text)

    documents = []
    # Index 0 is often header text, content starts at index 1
    for i in range(1, len(splits), 2):
        sec_num = splits[i].strip()
        content = splits[i+1].strip()

        if content:
            documents.append(Document(
                page_content=content,
                metadata={"section": sec_num, "source": str(INPUT_FILE)}
            ))

    print(f"📄 Created {len(documents)} structured documents.")

    # 2. Rebuild Database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print(f"🧠 Generating embeddings with {MODEL_NAME}...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 3. Load into Chroma
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"✅ Database built at '{DB_PATH}'")

    # 4. Quick Test
    query = "emergency vehicle right of way"
    print(f"\n🔍 Testing Search: '{query}'")
    results = db.similarity_search(query, k=1)
    for doc in results:
        print(f"Found in Section {doc.metadata['section']}:")
        print(f"{doc.page_content[:150]}...")

if __name__ == "__main__":
    main()
