import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Step 2 - Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# FIX: Delete and recreate the collection to avoid duplicate ID errors on re-ingestion.
# If you want to *append* instead of replace, remove these two lines and keep get_or_create_collection.
try:
    chroma_client.delete_collection(name="knowledge_base")
    print("Cleared existing 'knowledge_base' collection.")
except Exception:
    pass  # Collection didn't exist yet — that's fine

collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# Step 3 - Load and split the documents
def load_and_split_documents(docs_folder: str) -> list:
    """Load all .txt files from a folder and split into chunks."""
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )

    docs_path = Path(docs_folder)
    if not docs_path.exists():
        print(f"ERROR: The folder '{docs_folder}' does not exist.")
        print("Please create a 'docs/' folder next to this script and put your .txt files inside it.")
        return []

    txt_files = list(docs_path.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in '{docs_folder}/'.")
        return []

    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {txt_file.name}")

    return all_chunks

# Step 4 - Store the chunks in ChromaDB
def ingest_to_chromadb(chunks: list) -> None:
    """Store document chunks in ChromaDB."""
    documents = [chunk.page_content for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": chunk.metadata.get("source", "unknown")}
        for chunk in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(documents)} chunks in ChromaDB.")

    # FIX: Verify the chunks were actually saved
    stored_count = collection.count()
    print(f"Verification: ChromaDB now contains {stored_count} chunks.")

# Step 5 - The main runner
if __name__ == "__main__":
    print("Starting document ingestion...")
    print("Looking for .txt files in the 'docs/' folder...\n")
    chunks = load_and_split_documents("docs")

    if not chunks:
        print("\nIngestion aborted. Fix the issue above and try again.")
    else:
        ingest_to_chromadb(chunks)
        print(f"\nDone! Total chunks stored: {len(chunks)}")
        print("Your knowledge base is ready. You can now run rag_agent.py.")