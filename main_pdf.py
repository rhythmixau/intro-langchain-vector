import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

if __name__ == "__main__":
    print("Hello from main_pdf.py")

    loader = PyPDFLoader(file_path="data/sfc339entire.pdf")
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_splitted = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs_splitted, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(model="gpt-3.5-turbo-instruct"), retrieval_qa_chat_prompt) #gpt-3.5-turbo-instruct
    retrieval_chain = create_retrieval_chain(retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke(input ={"input": "how do kea birds got exposed to lead poisoning?"})
    print(result["answer"])
    