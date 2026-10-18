import warnings

warnings.filterwarnings("ignore")

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from rag01.retrieval import retrieve_documents
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env")

#llm = ChatNVIDIA(
    #model="deepseek-ai/deepseek-v4.1-flash",
    #model='z-ai/glm-5.3-flash',
  #  model="nvidia/nemotron-3.5-lightning-30b-a3b",
 #   api_key=NVIDIA_API_KEY,
#)

llm = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0,
)

def generate_answer(question: str) -> str:
    documents = retrieve_documents(question, k=3)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')}, "
        f"Page: {doc.metadata.get('page', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    print("\n--- Retrieved Context ---")
    print(context)

    prompt = f"""
You are a question-answering assistant.
Answer the question using only the provided context.

Rules:
- If the context does not contain enough information, say you don't know
  based on the available context.
- Do not invent facts.
- Keep the answer concise.
- Cite supporting pages using the format [Page N].

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    question = "What does Grunnings manufacture?"

    answer = generate_answer(question)

    print("Question:", question)
    print("\nAnswer:", answer)
