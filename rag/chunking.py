"""
Chunking functions for splitting documents into smaller chunks.
"""
from typing import List, Literal, Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

ChunkingFunc = Callable[[List[Document]], List[Document]]


def semantic_split(documents: List[Document], **kwargs) -> List[Document]:
    default_params = {
        "embeddings": OpenAIEmbeddings(model="text-embedding-3-large"),
        "buffer_size": 1,
        "add_start_index": False,
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": None,
        "number_of_chunks": None,
        "sentence_split_regex": r"(?<=[.。!！?？])\s+"
    }

    default_params.update(kwargs)

    chunker = SemanticChunker(**default_params)
    split_docs = []
    for doc in documents:
        split_docs.extend(chunker.split_documents([doc]))
    return split_docs


def recursive_split(documents: List[Document], **kwargs) -> List[Document]:
    default_params = {
        "separators": ["\n\n", "\n", " ", ""],
        "keep_separator": False,
        "is_separator_regex": False,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "length_function": len,
    }

    default_params.update(kwargs)

    splitter = RecursiveCharacterTextSplitter(**default_params)
    split_docs = []
    for doc in documents:
        split_docs.extend(splitter.split_documents([doc]))
    return split_docs
