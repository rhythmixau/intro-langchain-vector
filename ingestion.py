import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    print("Ingestion started")
    loader = TextLoader("data/medium-blog1.txt")
    docs = loader.load()

    print("Splitting documents")

    print("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    print(f"created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting chunks into Pinecone")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )

    print("Ingestion completed")
