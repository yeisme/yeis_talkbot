from typing import Callable, Type, Awaitable
from .event import BaseEvent

import asyncio


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

    async def publish(self, event: BaseEvent):
        """发布一个事件"""
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                await handler(event)

    async def publish2all(self, event: BaseEvent):
        """发布一个事件并并发执行所有处理器。"""
        event_type = type(event)
        if event_type in self._subscribers:
            tasks = [handler(event) for handler in self._subscribers[event_type]]
            await asyncio.gather(*tasks)  # Run all handlers concurrently


event_bus = EventBus()
