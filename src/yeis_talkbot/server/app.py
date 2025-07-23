from ..configs import AppConfig

app_config = AppConfig.from_yaml("configs/config.yaml")
mcp_config = app_config.Tools.load_mcp_config()
