import os

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def main():
    print("Hello from intro-to-vector!")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    query = "What is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm

    # Ask the LLM to answer the question without the context from Pinecone vector store
    # result = chain.invoke(input ={})
    # print(result)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    print("Querying Pinecone")
    # result = retrieval_chain.invoke(input ={"input": query})
    # print(result)

    template = """Use the following pieces of retrieved context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible.
    Always say, "Thank you for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template=template)
    custom_rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )
    result = custom_rag_chain.invoke(query)
    print(result)


if __name__ == "__main__":
    main()
