from rag import DocumentLoader, ChromaRag
from rag.chunking import recursive_split

from langchain_openai import OpenAIEmbeddings
from pathlib import Path
rag = ChromaRag(
    persist_directory=r"chroma",
    embedding_fun=OpenAIEmbeddings(model="text-embedding-3-large"),
)
file_path = "medications"
document_list = DocumentLoader.from_files(Path(file_path))
document_list = recursive_split(document_list, chunk_size=2000)
vector_ids = rag.save_documents(documents=document_list)
print(f"Saved {len(vector_ids)} documents.")