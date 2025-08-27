import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import chromadb


INDEX_NAME = os.environ["PINECONE_DOC_INDEX_NAME"]

def run_llm(query: str):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma_client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="chroma_db",
        embedding_function=embedding,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=documents_chain,
    )

    result = qa.invoke({"input": query})
    return result

    # chain = create_stuff_documents_chain(llm, retriever)
    # response = chain.invoke({"input": query})
    # return response

if __name__ == "__main__":
    print("Running the LLM...")
    res = run_llm("What is the definition of Langchain?")
    print(res["answer"])