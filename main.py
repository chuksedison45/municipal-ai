import os
from dotenv import load_dotenv

# Modern LangChain imports for v0.2/v0.3
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"  # Used to "read" the DB
LLM_MODEL = "llama3"                  # Used to "think" and answer

PROMPT_TEMPLATE = """
You are a legal assistant. Answer the question using ONLY the following context from the Municipal Code.
If the answer isn't in the context, say you don't know.

Context:
{context}

---

Question: {question}

Assistant Answer (Include Section Numbers):
"""

def main():
    load_dotenv()

    # 1. Initialize the Embedding Model (must match Lab 2)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 2. Load the existing ChromaDB
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}. Please run your loader script first.")
        return

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    print(f"🏛️  Municipal AI Ready (Model: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL})")
    print("Type 'exit' to quit.")
    print("-" * 50)

    while True:
        query_text = input("\n🔍 Legal Question: ")
        if query_text.lower() == 'exit':
            break

        # 3. Retrieve relevant sections using Embeddings
        # We fetch the top 3 most relevant matches
        results = db.similarity_search(query_text, k=3)

        if not results:
            print("⚠️ No matching sections found in the database.")
            continue

        # 4. Prepare Context for the LLM
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

        # 5. Format Prompt and Generate Answer via LLM
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        llm = OllamaLLM(model=LLM_MODEL)

        print("\n🤖 Thinking...")
        response = llm.invoke(prompt)

        # 6. Output Results
        print(f"\n📝 RESPONSE:\n{response}")

        # Show which sections were used as evidence
        sources = [doc.metadata.get("section", "N/A") for doc in results]
        print(f"\n📍 EVIDENCE: Sections {', '.join(set(sources))}")

if __name__ == "__main__":
    main()
