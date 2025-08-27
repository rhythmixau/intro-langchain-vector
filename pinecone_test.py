import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.environ["PINECONE_DOC_INDEX_NAME"]  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print("--------------------------------")
print(f"Target Indexes: {index_name}")
print(f"Pinecone Indexes: {existing_indexes}")

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
else:
    print("Index already exists")

index = pc.Index(index_name, host=os.environ["PINECONE_HOST"])
print("--------------------------------")
print(index.describe_index_stats())
print("--------------------------------")

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vector_store = PineconeVectorStore(index=index, embedding=embedding)
print(
    vector_store.get_pinecone_index(
        os.environ["PINECONE_DOC_INDEX_NAME"]
    ).describe_index_stats()
)
