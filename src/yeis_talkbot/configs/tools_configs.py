from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json


class RAGConfig(BaseModel):
    embedding_normalize: bool = Field(True, description="是否对 Embedding 做 L2 归一化")
    embedding_device: str = Field("cuda", description="cpu / cuda / mps / auto")
    embedding_model: str = Field(
        "BAAI/bge-large-zh-v1.5", description="Embedding 模型名称(HuggingFace Hub)"
    )

    vs_type: str = Field("chroma", description="chroma / milvus / faiss / weaviate")
    vs_collection: str = Field("default", description="向量库 collection / index 名称")
    vs_top_k: int = Field(5, description="检索返回条数")
    vs_score_threshold: float = Field(0.0, description="相似度阈值，<=0 表示不启用")
    rerank_model: Optional[str] = Field(None, description="重排序模型；None 表示不使用")


class ToolsConfig(BaseModel):
    mcp_file_path: str = Field(
        "mcp.json", description="Path to the MCP configuration文件"
    )

    rag: RAGConfig = Field(
        default_factory=lambda: RAGConfig(),  # type: ignore
        description="Configuration for the RAG tool",
    )

    def load_mcp_config(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """从 JSON 文件加载工具配置。"""
        if file_path is None:
            file_path = self.mcp_file_path
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})
