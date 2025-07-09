from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class BaseEvent:
    """事件的基础结构"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_name: str = field(init=False)

    def __post_init__(self):
        self.event_name = self.__class__.__name__
