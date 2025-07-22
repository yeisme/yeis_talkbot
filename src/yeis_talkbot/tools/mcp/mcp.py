import logging
from typing import List

from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore

from ...configs import AppConfig

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    一个通过 MCP 客户端提供 LangChain 工具的类。

    这个类封装了与 MultiServerMCPClient 交互的细节，
    并缓存了初始化后的客户端和工具列表。
    """

    def __init__(self, app_config: AppConfig):
        """
        使用应用配置初始化工具提供者。

        Args:
            app_config: 应用的全局配置。
        """
        self._app_config = app_config
        self._mcp_client: MultiServerMCPClient | None = None
        self._tools: List[BaseTool] = []
        self._initialized = False

    async def initialize(self) -> None:
        """
        异步初始化 MCP 客户端并加载工具。
        这个方法可以被调用多次，但只会实际执行一次。
        """
        if self._initialized:
            return

        logger.info("正在初始化 MCPToolProvider...")
        mcp_config = self._app_config.Tools.load_mcp_config()
        if not mcp_config:
            logger.warning("MCP 配置为空，无法加载任何工具。")
            self._initialized = True
            return

        self._mcp_client = MultiServerMCPClient(mcp_config)
        self._tools = await self._mcp_client.get_tools()

        if not self._tools:
            logger.warning("没有从 MCP 加载到任何工具，请检查 MCP 配置文件。")
        else:
            logger.info(f"从 MCP 配置中成功加载了 {len(self._tools)} 个工具。")

        self._initialized = True

    async def list_tools(self) -> List[BaseTool]:
        """
        获取已加载的工具列表。

        如果尚未初始化，会自动进行初始化。

        Returns:
            一个 BaseTool 对象的列表。
        """
        if not self._initialized:
            await self.initialize()
        return self._tools

    def get_client(self) -> MultiServerMCPClient | None:
        """
        获取已初始化的 MCP 客户端实例。

        Returns:
            MultiServerMCPClient 实例，如果未初始化则返回 None。
        """
        return self._mcp_client
