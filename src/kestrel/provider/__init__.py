"""Vendor-neutral provider surface: normalized events, the client protocol, and typed errors."""

from kestrel.provider.base import Effort, Message, ProviderClient, ToolSchema
from kestrel.provider.errors import (
    AuthError,
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from kestrel.provider.events import (
    StopEvent,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolCallEvent,
    UsageEvent,
    validate_stream_order,
)

__all__ = [
    "AuthError",
    "ContextOverflowError",
    "Effort",
    "Message",
    "ProviderClient",
    "ProviderError",
    "RateLimitError",
    "ServerError",
    "StopEvent",
    "StopReason",
    "StreamEvent",
    "TextDelta",
    "ToolCallEvent",
    "ToolSchema",
    "UsageEvent",
    "validate_stream_order",
]
