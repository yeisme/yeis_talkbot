from src.yeis_talkbot.configs import AppConfig, EdgeTTSConfig


def test_app_load_config():
    app: AppConfig = AppConfig.from_yaml("configs/config.yaml")

    assert hasattr(app, "TTS")
    assert app.TTS.out_path == "tmp/tts/"
    assert app.TTS.index_tts["config"] == "checkpoints/checkpoints-config.yaml"
    assert app.TTS.edge_tts["config"] == "configs/edge-tts.yaml"

    assert hasattr(app, "ASR")
    assert app.ASR.FunASR["model"] == "paraformer-zh-streaming"

    assert hasattr(app, "VAD")
    assert app.VAD.FunASR["model"] == "fsmn-vad"

    assert hasattr(app, "LLM")
    assert app.LLM.model == "deepseek/deepseek-chat-v3-0324:free"
    assert app.LLM.base_url == "https://openrouter.ai/api/v1"
    assert app.LLM.temperature == 0.7
    assert app.LLM.max_tokens is not None
    assert app.LLM.streaming is True
    assert app.LLM.timeout == 20

    assert hasattr(app, "Tools")
    assert app.Tools.mcp_file_path == "mcp.json"
    assert hasattr(app.Tools, "rag")
    assert app.Tools.rag.embedding_normalize is True
    assert app.Tools.rag.embedding_device == "cuda"
    assert app.Tools.rag.embedding_model == "BAAI/bge-large-zh-v1.5"
    assert app.Tools.rag.vs_type == "chroma"
    assert app.Tools.rag.vs_collection == "default"
    assert app.Tools.rag.vs_top_k == 5
    assert app.Tools.rag.vs_score_threshold == 0.0
    assert app.Tools.rag.rerank_model is None


def test_edge_tts_config():
    edge_tts_config = EdgeTTSConfig.from_yaml("configs/edge-tts.yaml")

    assert edge_tts_config.voice == "zh-CN-XiaoxiaoNeural"
    assert edge_tts_config.rate == "+0%"
    assert edge_tts_config.volume == "+0%"


def test_load_mcp_config():
    app: AppConfig = AppConfig.from_yaml("configs/config.yaml")

    mcp_config = app.Tools.load_mcp_config()
    assert isinstance(mcp_config, dict)
