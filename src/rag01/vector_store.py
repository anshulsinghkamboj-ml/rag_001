
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from rag01.chunking import chunk_documents
from rag01.ingest import extract_documents

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env")

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=NVIDIA_API_KEY,
)

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag01_documents"


def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def build_vector_store():
    store = get_vector_store()

    # Avoid adding the same documents on every run.
    if store._collection.count() > 0:
        print("Existing persisted collection found.")
        print("Stored documents:", store._collection.count())
        return store

    documents = extract_documents("data/raw/harry1.pdf")
    chunks = chunk_documents(documents)

    store.add_documents(chunks)

    print("Documents stored:", store._collection.count())
    print("Database directory:", CHROMA_DIR)

    return store


if __name__ == "__main__":
    store = build_vector_store()

    results = store.similarity_search(
        "Who are Mr. and Mrs. Dursley?",
        k=3,
    )

    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:300])
        print("Page:", doc.metadata.get("page"))
