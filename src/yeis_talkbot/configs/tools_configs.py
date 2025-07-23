from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json


class ToolsConfig(BaseModel):
    mcp_file_path: str = Field(
        "mcp.json", description="Path to the MCP configuration文件"
    )

    def load_mcp_config(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """从 JSON 文件加载工具配置。"""
        if file_path is None:
            file_path = self.mcp_file_path
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})
