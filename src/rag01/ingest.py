import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*being sunset.*",
    category=DeprecationWarning,
)


from langchain_community.document_loaders import PyPDFLoader


def extract_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    docs = extract_documents("data/raw/harry1.pdf")

    print("Total pages:", len(docs))
    print("\nFirst page content:")
    print(docs[0].page_content[:1000])
    print("\nMetadata:")
    print(docs[0].metadata)
