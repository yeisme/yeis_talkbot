from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid
from typing import Literal, Any


class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.event_name = self.__class__.__name__


class TTSEvent(BaseEvent):
    text: str = ""
    audio_path: str = ""
    status: Literal["pending", "processing", "completed", "failed"] = "pending"


class ASREvent(BaseEvent):
    audio_path: str = ""
    text: str = ""
    status: Literal["pending", "processing", "completed", "failed"] = "pending"


class BaseHandler:
    async def handle_event(self, event: BaseEvent) -> None:
        raise NotImplementedError("Subclasses must implement this method")


class TTSHandler(BaseHandler):
    """
    TTS 事件处理器基类

    初始化需要 TTS 实例
    """

    def __init__(self, tts: Any) -> None:
        self.tts = tts

    async def handle_event(self, event: BaseEvent) -> None:
        raise NotImplementedError("Subclasses must implement this method")


class ASRHandler(BaseHandler):
    """
    ASR 事件处理器基类

    初始化需要 ASR 实例
    """

    def __init__(self, asr: Any) -> None:
        self.asr = asr

    async def handle_event(self, event: BaseEvent) -> None:
        raise NotImplementedError("Subclasses must implement this method")
