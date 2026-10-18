

import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os

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


def retrieve_documents(question: str, k: int = 3):
    store = get_vector_store()
    return store.similarity_search(question, k=k)


if __name__ == "__main__":
    questions = [
        "Who are Mr. and Mrs. Dursley?",
        "What does Grunnings manufacture?",
        "Who is Dudley?",
        "Where do the Dursleys live?",
    ]

    for question in questions:
        print(f"\n{'=' * 70}")
        print("Question:", question)

        results = retrieve_documents(question, k=3)

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}")
            print("Page:", doc.metadata.get("page"))
            print("Content:", doc.page_content[:250].replace("\n", " "))
