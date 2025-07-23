from pydantic import BaseModel
from typing import Dict, Any
import json


class ToolsConfig(BaseModel):
    @staticmethod
    def load_mcp_config(file_path: str = "mcp.json") -> Dict[str, Any]:
        """从 JSON 文件加载工具配置。"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})
