from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    content: str
    vector: List[float]
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None


class VectorStore(ABC):
    """
    矢量数据库操作的抽象基类 (ABC)。

    这个接口定义了与 RAG 后端矢量存储交互所需的标准方法。
    任何具体的矢量数据库实现（如 ChromaDB, Pinecone, Qdrant）都应该继承这个类
    并实现其所有抽象方法。
    """

    @abstractmethod
    def create_collection(
        self, name: str, vector_dimension: int, distance_metric: str = "cosine"
    ):
        """
        创建一个新的集合（或索引）来存储向量。

        Args:
            name (str): 集合的唯一名称。
            vector_dimension (int): 集合中向量的维度。
            distance_metric (str): 计算向量相似度时使用的距离度量。
                                   常见的有 'cosine', 'euclidean' (l2), 'dot_product'。
                                   默认为 'cosine'。
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str):
        """
        删除一个集合及其所有内容。

        Args:
            name (str): 要删除的集合的名称。
        """
        pass

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """
        检查具有给定名称的集合是否存在。

        Args:
            name (str): 要检查的集合的名称。

        Returns:
            bool: 如果集合存在，则为 True，否则为 False。
        """
        pass

    @abstractmethod
    def upsert(self, collection_name: str, documents: List[Document]):
        """
        向指定的集合中插入或更新一批文档。

        如果文档ID已存在，则更新它；如果不存在，则插入。
        这是一个批量操作，以获得更好的性能。

        Args:
            collection_name (str): 目标集合的名称。
            documents (List[Document]): 要插入或更新的文档列表。
        """
        pass

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        在指定集合中执行相似度搜索。

        Args:
            collection_name (str): 要查询的集合的名称。
            query_vector (List[float]): 用于查询的输入向量。
            top_k (int): 要返回的最相似结果的数量。默认为 5。
            filters (Optional[Dict[str, Any]]): 用于在搜索前过滤数据的元数据条件。
                                               例如: {'source': 'file.pdf'}
                                               默认为 None，表示不进行过滤。

        Returns:
            List[Document]: 一个按相似度降序排列的文档列表。
                            每个文档应包含其相似度得分。
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]):
        """
        根据文档ID从指定集合中删除一个或多个文档。

        Args:
            collection_name (str): 目标集合的名称。
            ids (List[str]): 要删除的文档ID列表。
        """
        pass
