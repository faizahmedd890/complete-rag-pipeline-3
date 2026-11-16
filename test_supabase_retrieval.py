
from app.rag.vector_store import load_vectorstore

vectorstore = load_vectorstore()

query = "What is the company leave policy?"

results = vectorstore.similarity_search(query, k=5)

print("\nRetrieved documents:\n")

for doc in results:
    print(doc.page_content)
    print()
