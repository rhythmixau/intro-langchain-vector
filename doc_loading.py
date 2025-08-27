import os
from typing import List
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
import chromadb
import ssl
import certifi


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

chroma_client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)
# collection = chroma_client.get_or_create_collection("langchain-docs")

def ingest_docs():
    try:
        loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

        raw_documents = loader.load()
        print(f"Loaded {len(raw_documents)} documents")

        if len(raw_documents) > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            split_documents = text_splitter.split_documents(raw_documents)
            print(f"Split {len(split_documents)} documents")
            
            if len(split_documents) > 0:
                for doc in split_documents:
                    new_url = doc.metadata["source"]
                    new_url = new_url.replace("langchain-docs", "https:/")
                    doc.metadata.update({"source": new_url})

                print(f"Going to add {len(split_documents)} documents to Chroma DB")

                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="langchain-docs",
                    embedding_function=embeddings,
                    chroma_cloud_api_key=os.environ["CHROMA_API_KEY"],
                    tenant=os.environ["CHROMA_TENANT"],
                    database=os.environ["CHROMA_DATABASE"],
                )

                vectorstore.add_documents(
                    documents=split_documents,
                    ids=[str(uuid.uuid4()) for _ in range(len(split_documents))],
                )
                print("***Loading to vectorstore done***")
                # collection.add(
                #     documents=split_documents,
                #     ids=[str(uuid.uuid4()) for _ in range(len(split_documents))],)

    except Exception as e:
        print(f"Failed to load documents. Error: {e}")

if __name__ == "__main__":
    ingest_docs()