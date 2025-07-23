import pytest
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, AsyncMock, patch

from langchain_core.tools.base import BaseTool

from src.yeis_talkbot.tools.mcp.mcp import MCPToolProvider
from src.yeis_talkbot.configs import AppConfig, ToolsConfig


class MockTool(BaseTool):
    """用于测试的模拟工具"""

    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


@pytest.fixture
def mock_mcp_config() -> Dict[str, Any]:
    """提供模拟的MCP配置"""
    return {
        "test-server": {
            "command": "test-command",
            "args": ["arg1", "arg2"],
            "transport": "stdio",
        },
        "http-server": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
    }


@pytest.fixture
def mock_app_config(mock_mcp_config: Dict[str, Any]) -> Any:
    """提供模拟的应用配置"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"mcpServers": mock_mcp_config}, f)
        temp_file = f.name

    try:
        mock_config = Mock(spec=AppConfig)
        mock_tools_config = Mock(spec=ToolsConfig)
        mock_tools_config.load_mcp_config.return_value = mock_mcp_config
        mock_config.Tools = mock_tools_config
        yield mock_config
    finally:
        Path(temp_file).unlink()


@pytest.fixture
def mock_tools() -> List[BaseTool]:
    """提供模拟的工具列表"""
    return [MockTool(), MockTool(name="mock_tool_2", description="Another mock tool")]


class TestMCPToolProvider:
    """MCPToolProvider的测试类"""

    def test_init(self, mock_app_config: Mock) -> None:
        """测试MCPToolProvider的初始化"""
        provider = MCPToolProvider(mock_app_config)

        assert provider._app_config == mock_app_config
        assert provider._mcp_client is None
        assert provider._tools == []
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试成功初始化"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            await provider.initialize()

            assert provider._initialized is True
            assert provider._mcp_client == mock_client
            assert provider._tools == mock_tools
            mock_client_class.assert_called_once()
            mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_config(self, mock_app_config: Mock) -> None:
        """测试无配置时的初始化"""
        mock_app_config.Tools.load_mcp_config.return_value = {}
        provider = MCPToolProvider(mock_app_config)

        await provider.initialize()

        assert provider._initialized is True
        assert provider._mcp_client is None
        assert provider._tools == []

    @pytest.mark.asyncio
    async def test_initialize_no_tools(self, mock_app_config: Mock) -> None:
        """测试没有工具时的初始化"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = []
            mock_client_class.return_value = mock_client

            await provider.initialize()

            assert provider._initialized is True
            assert provider._mcp_client == mock_client
            assert provider._tools == []

    @pytest.mark.asyncio
    async def test_initialize_multiple_calls(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试多次调用initialize只执行一次"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            # 第一次调用
            await provider.initialize()
            # 第二次调用
            await provider.initialize()

            # 验证只被调用了一次
            mock_client_class.assert_called_once()
            mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_with_initialization(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试list_tools方法会触发初始化"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            tools = await provider.list_tools()

            assert tools == mock_tools
            assert provider._initialized is True

    @pytest.mark.asyncio
    async def test_list_tools_already_initialized(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试已初始化的情况下调用list_tools"""
        provider = MCPToolProvider(mock_app_config)
        provider._initialized = True
        provider._tools = mock_tools

        tools = await provider.list_tools()

        assert tools == mock_tools

    def test_get_client_before_initialization(self, mock_app_config: Mock) -> None:
        """测试初始化前获取客户端"""
        provider = MCPToolProvider(mock_app_config)

        client = provider.get_client()

        assert client is None

    @pytest.mark.asyncio
    async def test_get_client_after_initialization(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试初始化后获取客户端"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            await provider.initialize()
            client = provider.get_client()

            assert client == mock_client

    def test_is_initialized_false(self, mock_app_config: Mock) -> None:
        """测试未初始化状态"""
        provider = MCPToolProvider(mock_app_config)

        assert provider.is_initialized() is False

    @pytest.mark.asyncio
    async def test_is_initialized_true(
        self, mock_app_config: Mock, mock_tools: List[BaseTool]
    ) -> None:
        """测试已初始化状态"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            await provider.initialize()

            assert provider.is_initialized() is True

    @pytest.mark.asyncio
    async def test_initialize_with_exception(self, mock_app_config: Mock) -> None:
        """测试初始化过程中出现异常"""
        provider = MCPToolProvider(mock_app_config)

        with patch(
            "src.yeis_talkbot.tools.mcp.mcp.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await provider.initialize()

            assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_real_mcp_config_loading(self) -> None:
        """测试真实的MCP配置加载"""
        # 创建临时配置文件
        mcp_config = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                    "transport": "stdio",
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mcp_config, f)
            temp_file = f.name

        try:
            # 创建真实的配置对象
            tools_config = ToolsConfig(mcp_file_path=temp_file)
            loaded_config = tools_config.load_mcp_config()

            assert loaded_config == mcp_config["mcpServers"]
        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
