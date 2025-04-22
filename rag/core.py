import abc
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional, Union

from dotenv import load_dotenv
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logging.basicConfig(level=logging.INFO)
load_dotenv()


class DocumentLoader:
    @staticmethod
    def from_files(file_paths: Union[Sequence[Union[Path, str]], Path, str], **kwargs) -> List[Document]:
        """
        Load documents from local files or directories.
        :param file_paths: A single file/directory path (Path or str) or a list of file/directory paths.
        :return: List of documents. All source of documents will be a str since TextLoader will convert Path to string.
        """
        # If file_paths is a single file/directory, wrap it in a list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        # Gather all file paths, recursively if a directory is encountered
        all_files = []
        for file_path in file_paths:
            path_obj = Path(file_path)
            if path_obj.is_dir():
                # Recursively gather all files in the directory
                for sub_file in path_obj.rglob('*'):
                    if sub_file.is_file():
                        all_files.append(sub_file)
            elif path_obj.is_file():
                all_files.append(path_obj)
            else:
                raise ValueError(f'Path "{file_path}" is not a valid file or directory.')
        
        # Process each file using the appropriate loader based on its extension
        documents = []
        for file in all_files:
            file_path_str = str(file)
            extension = file.suffix
            match extension:
                case '.csv':
                    loader = CSVLoader(file_path=file_path_str, csv_args=kwargs)
                    documents += loader.load()
                case '.txt':
                    loader = TextLoader(file_path=file_path_str, autodetect_encoding=True)
                    documents += loader.load()
                case '.docx':
                    loader = Docx2txtLoader(file_path=file_path_str)
                    documents += loader.load()
                case '.pdf':
                    loader = PyPDFLoader(file_path=file_path_str)
                    documents += loader.load()
                case '.pptx':
                    loader = UnstructuredPowerPointLoader(file_path=file_path_str)
                    documents += loader.load()
                case '.html':
                    loader = UnstructuredHTMLLoader(file_path=file_path_str)
                    documents += loader.load()
                case '.md':
                    loader = TextLoader(file_path=file_path_str, autodetect_encoding=True)
                    documents += loader.load()
                case _:
                    raise ValueError(f'File extension "{extension}" is not supported.')
        return documents
    def from_str(text: str, metadata: dict = None) -> Document:
        """
        Load a document from str
        :param text: Text string
        :param metadata: Metadata of the text
        :return: Document object
        """
        if metadata is None:
            metadata = {}
        return Document(page_content=text, metadata=metadata)


class AbstractRag(metaclass=abc.ABCMeta):
    def __init__(self, vector_store: VectorStore, reranker_name: Optional[str] = None,
                 *, max_size: int = sys.maxsize):
        if reranker_name is not None and reranker_name not in {'cohere', 'bge'}:
            raise ValueError("Unknown reranker type, only support 'cohere' and 'bge' if provided.")

        self.vectorstore = vector_store
        self.reranker_name = reranker_name  # If None, reranking is disabled
        self.max_size = max_size

    def save_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Save documents to the vector database
        :param documents: Documents to save
        :param ids: IDs of the documents
        :return: ID sequence of the saved documents
        """
        if ids is None:
            return self.vectorstore.add_documents(documents=documents)
        else:
            return self.vectorstore.add_documents(documents=documents, ids=ids)

    def update_by_ids(self, ids: List[str], documents: List[Document]) -> None:
        raise NotImplementedError

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def delete_by_ids(self, ids: List[str]):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Search for similar documents
        :param query: Query string
        :param k: Number of documents to return. -1 means return all
        :return: Top k similar documents
        """
        if k == -1:
            k = self.max_size
        return self.vectorstore.similarity_search(query, k)

    def rerank(self, documents: List[Document], query: str, k: int) -> Sequence[Document]:
        """
        Rerank the documents based on the query if a reranker is specified.
        If no reranker is provided, return the original documents (or top k if specified).
        :param documents: Documents to rerank
        :param query: Query string
        :param k: Number of documents to return. -1 means return all
        :return: Reranked documents or original documents if reranker is not set.
        """
        if k == -1:
            top_n = len(documents)
        else:
            top_n = k

        # If no reranker is specified, simply return the top_n documents without reranking.
        if self.reranker_name is None:
            logging.info("Reranking is disabled; returning original documents.")
            return documents[:top_n]

        match self.reranker_name:
            case 'bge':
                model = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')
                reranker = CrossEncoderReranker(model=model, top_n=top_n)
            case 'cohere':
                reranker = CohereRerank(model='rerank-english-v3.0', top_n=top_n)
            case _:
                logging.warning(f"Unknown reranker type {self.reranker_name}, using default 'cohere'")
                reranker = CohereRerank(model='rerank-english-v3.0', top_n=top_n)

        return reranker.compress_documents(documents=documents, query=query)


class ChromaRag(AbstractRag):
    def __init__(self, persist_directory: Optional[Path], embedding_fun: Embeddings, reranker_name: Optional[str] = None,
                 max_size: int = sys.maxsize):

        if persist_directory is not None:
            if isinstance(persist_directory, str):
                try:
                    persist_directory = Path(persist_directory)
                except:
                    raise FileExistsError
            if persist_directory.is_dir():
                logging.info(f"Chroma directory {persist_directory} already exists, loading existing Chroma instance")
            else:
                logging.info(f"Chroma directory {persist_directory} does not exist, creating new Chroma instance")
                persist_directory.mkdir(parents=True)

        chroma_dir_str = str(persist_directory) if persist_directory is not None else None
        vectorstore = Chroma(persist_directory=chroma_dir_str, embedding_function=embedding_fun)

        super().__init__(vectorstore, reranker_name, max_size=max_size)
        self.vectorstore = vectorstore

    def update_by_ids(self, ids: List[str], documents: List[Document]):
        self.vectorstore.update_documents(ids, documents)

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        return self.vectorstore.get(ids)
    
    def delete_by_ids(self, ids: List[str]):
        self.vectorstore.delete(ids)

    def reset(self):
        self.vectorstore.reset_collection()