
import os

from dotenv import load_dotenv
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


if __name__ == "__main__":
    documents = extract_documents("data/raw/harry1.pdf")
    chunks = chunk_documents(documents)

    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(vectors))
    print("Embedding dimensions:", len(vectors[0]))
    print("First vector sample:", vectors[0][:10])
