import os
import asyncio
import ssl
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


if __name__ == "__main__":
    print("Hello from main_doc_helper.py")
