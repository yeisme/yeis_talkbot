"""
RAG (Retrieval-Augmented Generation) 工具模块

提供文档存储、检索、向量化等功能，支持多种向量数据库后端。
"""

from .abc import VectorStore

try:
    from .chroma import ChromaVectorStore

    __all__ = ["VectorStore", "ChromaVectorStore"]
except ImportError:
    __all__ = ["VectorStore"]
