# from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableParallel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = "chroma_db"


def create_rag_chain(model="nomic-embed-text"):
    """Creates the RAG chain for the AI assistant."""
    print("🚀 Initializing RAG chain...")

    # Initialize components
    embeddings = OllamaEmbeddings(model=model)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # Create the Prompt Template
    prompt_template = """
    You are an expert assistant on El Paso municipal codes. Your task is to answer questions based ONLY on the following context.
    If the context does not contain the answer, state that the information is not available in the provided documents.
    Do not use any outside knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Initialize the LLM
    llm = OllamaLLM(model="llama3")

    # The chain that generates the answer
    answer_chain = prompt | llm | StrOutputParser()

    # The complete chain that returns sources
    rag_chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=answer_chain)
    )

    print("✅ RAG chain initialized.")
    return rag_chain


# This block is for testing the module directly
if __name__ == '__main__':
    rag_chain = create_rag_chain()
    print("\n--- Testing the RAG chain ---")
    question = "how tall can my fence be"
    response = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {[doc.metadata for doc in response['context']]}")

