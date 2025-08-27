import os
import asyncio
import ssl
import certifi
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning
from rich.console import Console

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

console = Console()

# If using Pinecone
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# index = pc.Index(os.environ["PINECONE_DOC_INDEX_NAME"], host=os.environ["PINECONE_HOST"])

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
# vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore = Chroma(
    collection_name="chroma_db",
    embedding_function=embeddings,
    chroma_cloud_api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
)
tavily_extract = TavilyExtract(
    credentails=os.getenv("TAVILY_API_KEY"), max_num_results=50
)
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks


# async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
#     """Extract content from a batch of URLs."""
#     try:
#         console.print(f"🔍 Extracting batch {batch_num} of {len(urls)}", style="bold blue")
#         docs = await tavily_extract.ainvoke(input={"urls": urls})
#         results = docs.get("results", [])
#         console.print(f"✅ Extracted {len(results)} documents from batch {batch_num}", style="bold green")
#         return results
#     except Exception as e:
#         console.print(f"❌ Error extracting batch {batch_num}: {e}")
#         return []


async def extract_batch2(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """extract_batch2 - Extract documents from a batch of URLs."""

    try:
        log_info(f"🔍 Extracting batch {batch_num} of {len(urls)}", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"✅ Extracted {len(docs)} documents from batch {batch_num}")
        return docs
    except Exception as e:
        console.print(
            f"❌ TavilyExtract Error: Failed to extract batch {batch_num}: {e}"
        )
        return []


async def async_extract(url_batches: List[List[str]]) -> List[Dict[str, Any]]:
    """Extract documents from a list of URL batches."""
    log_header(f"Extracting {len(url_batches)} batches of {len(url_batches[0])} urls")
    log_info(f"🛠 TavilyExtract: Starting extraction process")
    tasks = [extract_batch2(batch, i) for i, batch in enumerate(url_batches, start=1)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    # print("Batch Results: ")
    # print(batch_results)
    # print("--------------------------------")
    all_pages = []
    failed_batches = 0
    for result in batch_results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract Error: {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"✅ Successfully extracted {len(all_pages)} documents from {len(url_batches)} batches"
    )
    log_info(f"💡 Note: {failed_batches} batches failed to extract")
    return all_pages


async def index_documents_async(docs: List[Document], batch_size: int = 100) -> None:
    """Index documents asynchronously in batches."""

    log_header("Indexing documents")
    log_info(
        f"🔍 Indexing {len(docs)} documents in batches of {batch_size}", Colors.CYAN
    )

    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

    log_info(f"🔍 Indexing {len(batches)} batches", Colors.CYAN)

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            log_info(f"🔍 Indexing batch {batch_num} of {len(batches)}", Colors.CYAN)
            await vectorstore.aadd_documents(batch)
            log_success(
                f"✅ Successfully indexed batch {batch_num}/{len(batches)} documents"
            )
        except Exception as e:
            log_error(f"❌ Error indexing batch {batch_num}: {e}")
            return False
        return True

    tasks = [add_batch(batch, i) for i, batch in enumerate(batches, start=1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful_batches = sum(1 for result in results if result is True)
    successful = True if successful_batches == len(batches) else False

    if successful:
        log_success(
            f"VectorStore indexing: All batches indexed successfully ({successful_batches}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore indexing: Failed to index {len(batches) - successful_batches} batches"
        )


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("Starting document ingestion process")
    log_info(
        "TavilyCrawl: Starting Tavily Crawl on https://python.langchain.com/",
        Colors.PURPLE,
    )

    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(f"Successfully mapped {len(site_map["results"])} urls!")

    ## START
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)

    console.print(
        f"Processing {len(site_map["results"])} urls in {len(url_batches)} batches",
        style="bold blue",
    )

    all_docs = await async_extract(url_batches)
    log_success(f"✅ Successfully extracted content from {len(all_docs)} urls")

    log_header("Docuemnt Chunking Phase")
    log_info(
        f"🔍 Chunking {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"✅ Successfully split {len(all_docs)} documents into {len(splitted_docs)} chunks"
    )

    await index_documents_async(splitted_docs, batch_size=250)

    log_header("PIPELINE COMPLETED")
    log_success("🎉 All tasks completed successfully!")
    log_info(
        "💡 Note: You can now use the vectorstore for retrieval and querying",
        Colors.YELLOW,
    )
    log_info(f"URLs mapped: {len(site_map["results"])}", Colors.BLUE)
    log_info(f"Documents extracted: {len(all_docs)}", Colors.BLUE)
    log_info(f"Documents indexed: {len(splitted_docs)}", Colors.BLUE)
    ## END


if __name__ == "__main__":
    asyncio.run(main())
