
import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.config import DATA_DIR
from app.rag.vector_store import load_vectorstore


def load_pdf_documents():

    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):

        loader = PyPDFLoader(
            str(pdf_file)
        )

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

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as file:

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

    print(
        f"Loaded documents: {len(documents)}"
    )

    if not documents:

        raise RuntimeError(
            f"No documents found in {DATA_DIR}"
        )

    # Clean extracted PDF and JSON text
    for doc in documents:

        doc.page_content = re.sub(
            r"\s+",
            " ",
            doc.page_content
        ).strip()

    print("Document text cleaned successfully.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created chunks: {len(chunks)}"
    )

    if not chunks:
        raise RuntimeError("No document chunks created.")

    # Connect to Supabase
    vectorstore = load_vectorstore()

    # Replace old embeddings with new cleaned embeddings
    print(
        "Replacing old embeddings with cleaned documents..."
    )

    inserted_count = vectorstore.replace_documents(
        chunks
    )

    print(
        f"Supabase documents: {inserted_count}"
    )

    print(
        "Ingestion completed successfully."
    )


if __name__ == "__main__":

    ingest()
