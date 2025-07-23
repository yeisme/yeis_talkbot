from typing import Dict, Any
from pydantic import BaseModel, Field
import json


class ToolsConfig(BaseModel):
    mcp_file_path: str = Field(
        "mcp.json", description="Path to the MCP configuration file"
    )

    @staticmethod
    def load_mcp_config(file_path: str = mcp_file_path) -> Dict[str, Any]:
        """从 JSON 文件加载工具配置。"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})
