import logging
from typing import List, Mapping, Union

import chromadb
from chromadb.api import ClientAPI
from langchain_core.documents import Document

from src.yeis_talkbot.configs import AppConfig

from .abc import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.client: ClientAPI
        # 初始化 ChromaDB 特定的客户端
        if config.Tools.rag.vs_path:
            self.client = chromadb.PersistentClient(path=config.Tools.rag.vs_path)
            logger.info(
                f"初始化 ChromaDB 使用持久化存储路径: {config.Tools.rag.vs_path}"
            )
        else:
            # 如果没有指定路径，则使用默认的内存存储
            self.client = chromadb.Client()
            logger.info("初始化 ChromaDB 使用内存存储")

    def create_collection(self, name: str = ""):
        if not name:
            name = self.config.Tools.rag.vs_collection
        if self.embeddings is None:
            raise ValueError("Embeddings model is not set or initialized correctly.")

        get_collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embeddings,  # type: ignore
        )

        logger.info(f"获取或创建集合: {name}")

        self.collection_list[name] = get_collection
        return True

    def delete_collection(self, name: str):
        if name not in self.collection_list:
            logger.warning(f"集合 {name} 不存在，无法删除。")
            return False

        self.client.delete_collection(name)
        del self.collection_list[name]
        logger.info(f"删除集合: {name}")
        return True

    def upsert(self, collection_name: str, documents: List[Document]) -> List[str]:
        if collection_name not in self.collection_list:
            logger.warning(f"集合 {collection_name} 不存在，无法插入文档。")
            return []

        collection: chromadb.Collection = self.collection_list[collection_name]

        # 准备要插入的数据
        docs_to_upsert = [doc.page_content for doc in documents]
        metadatas: List[Mapping[str, Union[str, int, float, bool, None]]] = [
            doc.metadata for doc in documents
        ]
        ids = [
            doc.metadata.get("id") or f"{collection_name}_{i}"
            for i, doc in enumerate(documents)
        ]

        collection.upsert(
            documents=docs_to_upsert,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"向集合 {collection_name} 更新/插入了 {len(ids)} 个文档")
        return ids

    def query(
        self,
        collection_name: str,
        query_texts: List[str],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> List[Document]:
        if collection_name not in self.collection_list:
            logger.warning(f"集合 {collection_name} 不存在，无法查询。")
            return []

        collection: chromadb.Collection = self.collection_list[collection_name]

        results = collection.query(
            query_texts=query_texts,
            n_results=top_k,
            where=filters,
        )

        documents = []
        if results and results["documents"]:
            for i, docs in enumerate(results["documents"]):
                for j, doc_content in enumerate(docs):
                    metadata = (
                        dict(results["metadatas"][i][j])
                        if results["metadatas"]
                        and results["metadatas"][i]
                        and results["metadatas"][i][j]
                        else {}
                    )
                    if (
                        results["distances"]
                        and results["distances"][i]
                        and results["distances"][i][j] is not None
                    ):
                        metadata["score"] = results["distances"][i][j]
                    documents.append(
                        Document(page_content=doc_content, metadata=metadata)
                    )

        return documents

    def delete(self, collection_name: str, ids: List[str]):
        if collection_name not in self.collection_list:
            logger.warning(f"集合 {collection_name} 不存在，无法删除文档。")
            return

        collection: chromadb.Collection = self.collection_list[collection_name]
        collection.delete(ids=ids)
        logger.info(f"从集合 {collection_name} 中删除了 {len(ids)} 个文档")
