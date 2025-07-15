from typing import Callable, Type, Awaitable
from .event import BaseEvent

import logging

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        # 存储事件类型与其对应的处理器列表
        self._subscribers: dict[
            Type[BaseEvent], list[Callable[[BaseEvent], Awaitable[None]]]
        ] = {}

    def subscribe(
        self,
        event_type: Type[BaseEvent],
        handler: Callable[[BaseEvent], Awaitable[None]],
    ):
        """订阅一个事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.info(f"订阅事件: {event_type.__name__}")

    def unsubscribe(
        self,
        event_type: Type[BaseEvent],
        handler: Callable[[BaseEvent], Awaitable[None]],
    ):
        """取消订阅一个事件处理器"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.info(f"取消订阅事件: {event_type.__name__}")
            except ValueError:
                logger.warning(
                    f"处理器未找到: {handler}，无法取消订阅 {event_type.__name__}"
                )

    async def publish(self, event: BaseEvent):
        """发布一个事件"""
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                await handler(event)
        logger.info(f"发布事件: {event_type.__name__}")


event_bus = EventBus()
