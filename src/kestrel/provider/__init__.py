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
from kestrel.provider.litellm_client import LiteLLMClient
from kestrel.provider.retry import RetryPolicy, complete_with_retry

__all__ = [
    "AuthError",
    "ContextOverflowError",
    "Effort",
    "LiteLLMClient",
    "Message",
    "ProviderClient",
    "ProviderError",
    "RateLimitError",
    "RetryPolicy",
    "ServerError",
    "StopEvent",
    "StopReason",
    "StreamEvent",
    "TextDelta",
    "ToolCallEvent",
    "ToolSchema",
    "UsageEvent",
    "complete_with_retry",
    "validate_stream_order",
]
