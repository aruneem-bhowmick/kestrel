"""Typed provider error taxonomy shared by every backend adapter.

Retries, backoff, and failover between backends are deliberately absent
from this module -- they belong to the caller, once one exists that
needs them. What every adapter must do today is map its own SDK's
exceptions onto these four types, so the rest of the codebase can react
to *kinds* of failure (auth, rate limit, context overflow, server error)
without importing or naming a specific vendor's exception classes.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every typed provider failure.

    Attributes:
        model_id: The registry id active when the failure occurred.
        backend: The backend name active when the failure occurred.
    """

    def __init__(self, message: str, *, model_id: str, backend: str) -> None:
        """Fold ``model_id``/``backend`` into the rendered message.

        Embedding both in ``str(self)`` means a bare ``print(exc)`` at a
        REPL or log call site is already self-identifying, even after a
        ``/model`` hot-swap has made "the active model" ambiguous without
        that context.
        """
        super().__init__(f"{message} [{backend}/{model_id}]")
        self.model_id = model_id
        self.backend = backend


class AuthError(ProviderError):
    """Credentials were missing, empty, or rejected by the backend."""


class RateLimitError(ProviderError):
    """The backend throttled the request.

    Attributes:
        retry_after_s: Seconds to wait before retrying, when the backend
            supplied one (e.g. a ``Retry-After`` header); ``None`` when
            it did not.
    """

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        backend: str,
        retry_after_s: float | None = None,
    ) -> None:
        """Store ``retry_after_s`` alongside the standard error fields."""
        super().__init__(message, model_id=model_id, backend=backend)
        self.retry_after_s = retry_after_s


class ContextOverflowError(ProviderError):
    """The request exceeded the model's context window."""


class ServerError(ProviderError):
    """The backend failed for a reason outside the caller's control.

    Attributes:
        status: The HTTP status code reported by the backend, when one is
            available; ``None`` for failures with no status (e.g. a
            connection timeout).
    """

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        backend: str,
        status: int | None = None,
    ) -> None:
        """Store ``status`` alongside the standard error fields."""
        super().__init__(message, model_id=model_id, backend=backend)
        self.status = status
