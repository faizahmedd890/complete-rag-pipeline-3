from pathlib import Path
import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def load_pdf_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"

        documents.extend(pdf_docs)

    return documents


def load_employee_json():
    json_file = DATA_DIR / "employee_records.json"

    if not json_file.exists():
        return []

    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("employees", [data])
    else:
        records = []

    for employee in records:
        text = "\n".join(
            f"{key}: {value}"
            for key, value in employee.items()
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "employee_records.json",
                    "file_type": "json",
                    "employee_id": str(
                        employee.get("employee_id", "")
                    )
                }
            )
        )

    return documents


def ingest():
    print("Loading documents...")

    documents = []

    pdf_documents = load_pdf_documents()
    json_documents = load_employee_json()

    documents.extend(pdf_documents)
    documents.extend(json_documents)

    print(f"Loaded documents: {len(documents)}")

    if not documents:
        raise RuntimeError(
            f"No documents found in {DATA_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    existing_count = vectorstore._collection.count()

    if existing_count > 0:
        print(
            f"Chroma already contains {existing_count} documents."
        )
        print("Skipping ingestion.")
        return

    print("Creating embeddings and storing in Chroma...")

    vectorstore.add_documents(chunks)

    final_count = vectorstore._collection.count()

    print(f"Chroma documents: {final_count}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest()