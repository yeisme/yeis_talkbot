from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.yeis_talkbot.configs import AppConfig


class VectorStore(ABC):
    """矢量数据库操作的抽象基类 (Abstract Base Class)。

    该接口遵循策略设计模式，定义了与任何 RAG (Retrieval-Augmented Generation)
    后端矢量存储进行交互所需的标准方法。通过依赖此抽象而非具体实现，
    应用程序可以轻松地在不同的矢量数据库之间切换（例如 ChromaDB, Pinecone, Qdrant），
    而无需修改核心业务逻辑。

    任何具体的矢量数据库实现都应该继承这个类，并实现其所有抽象方法。
    """

    def __init__(self, config: AppConfig):
        """初始化矢量存储基类。

        子类在实现时应首先调用 `super().__init__(config)`，然后初始化
        其特定的数据库客户端并赋值给 `self.client`。

        Args:
            config (AppConfig): 应用程序配置，包含矢量存储的相关参数。
        """
        self.config = config
        self.client = None  # 子类应在初始化时赋值
        self.collection_list: Dict[str, Any] = {}

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """获取 HuggingFace 的嵌入模型实例。

        这是一个便利属性，用于在所有子类中统一嵌入模型的创建方式。
        模型配置从 `AppConfig` 中读取。

        Returns:
            HuggingFaceEmbeddings: 用于生成文本嵌入的模型实例。
        """
        return HuggingFaceEmbeddings(
            model_name=self.config.Tools.rag.embedding_model,
            model_kwargs={"device": self.config.Tools.rag.embedding_device},
            normalize_embeddings=self.config.Tools.rag.embedding_normalize,
        )

    @abstractmethod
    def create_collection(self, name: str) -> bool:
        """创建一个新的集合（或索引）来存储向量。

        如果同名集合已存在，此方法可能会抛出异常，具体行为取决于底层数据库的实现。

        Args:
            name (str): 集合的唯一名称。

        Raises:
            Exception: 如果集合已存在或创建失败。
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """删除一个集合及其所有内容。

        Args:
            name (str): 要删除的集合的名称。

        Raises:
            Exception: 如果集合不存在或删除失败。
        """
        pass

    @abstractmethod
    def upsert(self, collection_name: str, documents: List[Document]) -> List[str]:
        """向指定的集合中插入或更新一批文档。

        这是一个“更新或插入”操作。如果文档ID已存在，则更新它；如果不存在，则插入新文档。
        建议实现为批量操作以获得更好的性能。

        Args:
            collection_name (str): 目标集合的名称。
            documents (List[Document]): 要插入或更新的 LangChain 文档列表。

        Returns:
            List[str]: 成功插入或更新的文档的ID列表。

        Raises:
            Exception: 如果目标集合不存在或操作失败。
        """
        pass

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_texts: List[str],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """在指定集合中执行相似度搜索。

        Args:
            collection_name (str): 要查询的集合的名称。
            query_texts (List[str]): 用于查询的输入文本列表。
            top_k (int): 要返回的最相似结果的数量。默认为 5。
            filters (Optional[Dict[str, Any]]): 用于在搜索前过滤数据的元数据条件。
                                               例如: `{'source': 'file.pdf'}`。
                                               默认为 None，表示不进行过滤。

        Returns:
            List[Document]: 一个按相似度降序排列的文档列表。每个返回的 `Document`
                            对象的 `metadata` 字段中应包含一个 `score` 键，
                            表示其与查询向量的相似度得分。

        Raises:
            Exception: 如果目标集合不存在或查询失败。
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]):
        """根据文档ID从指定集合中删除一个或多个文档。

        Args:
            collection_name (str): 目标集合的名称。
            ids (List[str]): 要删除的文档ID列表。

        Raises:
            Exception: 如果目标集合不存在或删除操作失败。
        """
        pass

    def collection_exists(self, name: str) -> bool:
        """检查集合是否存在。

        Args:
            name (str): 集合的名称。

        Returns:
            bool: 如果集合存在则返回 True，否则返回 False。
        """
        return name in self.collection_list

    def get_collection(self, name: str):
        """获取指定名称的集合。

        Args:
            name (str): 集合的名称。

        Returns:
            Collection: 如果集合存在则返回对应的集合对象，否则返回 None。
        """
        return self.collection_list.get(name, None)

    def list_collections(self) -> list:
        """列出所有已创建的集合名称。

        Returns:
            list: 包含所有集合名称的列表。
        """
        return list(self.collection_list.keys())
