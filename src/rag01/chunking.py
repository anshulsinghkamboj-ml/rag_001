
import warnings

warnings.filterwarnings("ignore")


from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag01.ingest import extract_documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    documents = extract_documents("data/raw/harry1.pdf")
    chunks = chunk_documents(documents)

    print("Total pages:", len(documents))
    print("Total chunks:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nFirst chunk metadata:")
    print(chunks[0].metadata)

    print("\nFirst 10 chunk lengths:")
    for i, chunk in enumerate(chunks[:10]):
        print(f"Chunk {i}: {len(chunk.page_content)} characters")
