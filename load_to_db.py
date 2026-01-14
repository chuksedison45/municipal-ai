import os
import re
import shutil
from dotenv import load_dotenv
# --- CHANGED: Using HuggingFace instead of Google ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
OCR_TEXT_PATH = "full_text_ocr.txt"
DB_PATH = "chroma_db"

def main():
    # Load environment variables (kept for other potential needs)
    load_dotenv()

    print("🚀 Starting database loading process...")

    # 1. Load the OCR'd text
    if not os.path.exists(OCR_TEXT_PATH):
        print(f"❌ Error: Text file not found at '{OCR_TEXT_PATH}'")
        return

    print(f"📖 Loading text from '{OCR_TEXT_PATH}'...")
    with open(OCR_TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Parse sections with Regex
    print("📑 Parsing text into sections using Regex...")
    section_pattern = r'(\d+\.\d+\.\d+)'
    splits = re.split(section_pattern, text)

    documents = []
    # Combine the section number with its content
    for i in range(1, len(splits), 2):
        section_number = splits[i]
        content = splits[i+1]
        documents.append(
            Document(page_content=content.strip(), metadata={"section": section_number})
        )

    # 3. Fallback to chunking if Regex parsing is ineffective
    if len(documents) < 10:
        print("⚠️  Few sections found, using fallback chunking strategy...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.create_documents([text])

    print(f"📄 Created {len(documents)} documents.")

    # 4. Clear out the old database
    if os.path.exists(DB_PATH):
        print("🗑️  Removing existing database...")
        shutil.rmtree(DB_PATH)

    # 5. Initialize the embedding model and ChromaDB
    # --- CHANGED: Initializing local HuggingFace Model ---
    print("🧠 Initializing HuggingFace Embeddings (Local)...")
    # "all-MiniLM-L6-v2" is small, fast, and great for general purpose tasks
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"🗄️  Initializing ChromaDB at '{DB_PATH}'...")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 6. Add documents to the vector store
    print(f"⚡ Adding {len(documents)} documents to the database...")
    print("This will be much faster since it's running locally! ⚡")
    db.add_documents(documents)

    # Note: db.persist() is deprecated in newer Chroma versions as it autosaves
    print("✅ Documents added successfully.")

    # 7. Verify the database
    print("\n🔍 Verifying database...")
    try:
        collection_count = db._collection.count()
        print(f"✅ Database has {collection_count:,} documents!")

        # Run a test similarity search
        print("\nRunning a test search for 'fence height'...")
        test_results = db.similarity_search("fence height", k=3)

        if test_results:
            for doc in test_results:
                section = doc.metadata.get('section', 'Unknown')
                print(f"   📋 Result: Section {section} | {doc.page_content[:100]}...")
        else:
            print("❌ Test search returned no results.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")

    print("\n🎉 COMPLETE! Database is ready.")

# Run the main function
if __name__ == "__main__":
    main()
