from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

question = input("Ask: ")

results = db.similarity_search(
    question,
    k=3
)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content)
for doc in results:
    print("\nSOURCE:")
    print(doc.metadata)

    print("\nCONTENT:")
    print(doc.page_content)

