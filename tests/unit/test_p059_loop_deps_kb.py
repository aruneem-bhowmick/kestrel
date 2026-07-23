"""Unit tests for `kestrel.agent.loop.LoopDeps.kb`: left unset, it is
`None`, identical to every field predating it; when set, a dispatched
tool call's own `context` carries it through to whichever tool executor
declares a `kb` keyword-only parameter -- proven with a small
test-only stub tool registered via a monkeypatched
`kestrel.tools.registry._TOOLS`/`_BY_NAME`, since no real tool declares
that parameter yet. This suite otherwise mirrors
`test_p045_dispatch_call_refusal.py`'s own direct `_dispatch_tool_call`
scripting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

import kestrel.agent.loop as loop_module
import kestrel.tools.registry as registry_module
from kestrel.agent.loop import LoopDeps
from kestrel.config import KbConfig
from kestrel.cost.meter import CostMeter
from kestrel.kb.service import KbService
from kestrel.managers.approval import ApprovalManager
from kestrel.managers.undo import UndoManager
from kestrel.provider.base import Effort, Message, ToolSchema
from kestrel.provider.events import StreamEvent, ToolCallEvent
from kestrel.registry.model import ModelEntry, Registry
from kestrel.tools.registry import _ToolBinding

pytestmark = [pytest.mark.p059, pytest.mark.unit]

_MODEL_ID = "glm-5.2"
_STUB_TOOL_NAME = "kb_probe"


class _UnusedClient:
    """Stands in for `ProviderClient` in a `LoopDeps` built only to
    exercise `_dispatch_tool_call` directly -- that function never
    reads `deps.client`, so calling this would be a test-authoring
    mistake."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        model_id: str,
        effort: Effort,
        stream: bool = True,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Never called; raises if it somehow is."""
        raise AssertionError("_dispatch_tool_call must never call deps.client")
        yield  # pragma: no cover -- makes this an async generator


class _UnusedEmbeddingClient:
    """Stands in for `EmbeddingClient` in a `KbService` this suite only
    ever compares by identity -- raises if `embed` is somehow actually
    called."""

    async def embed(
        self, texts: Sequence[str], *, model_id: str
    ) -> tuple[tuple[float, ...], ...]:
        """Never called; raises if it somehow is."""
        raise AssertionError(
            "this stub KbService's embedding_client must never be used"
        )


class _StubArgs:
    """Empty argument object -- the stub tool's own parser always
    returns one of these, regardless of what a call's own JSON names."""


class _StubToolError(Exception):
    """The stub tool's own error type -- declared to satisfy
    `_ToolBinding`'s own shape, never actually raised."""


def _parse_stub_args(raw: str) -> _StubArgs:
    """Ignore `raw` entirely and return a fresh `_StubArgs` -- this stub
    tool takes no real arguments."""
    return _StubArgs()


_STUB_SCHEMA = ToolSchema(
    name=_STUB_TOOL_NAME,
    description="test-only stub declaring a keyword-only kb parameter",
    parameters={"type": "object", "properties": {}},
)


def _registry() -> Registry:
    """A single-entry `Registry`, matching this suite's siblings."""
    entry = ModelEntry(
        id=_MODEL_ID,
        backend="openrouter",
        provider_model="z-ai/glm-5.2",
        api_key_env="OPENROUTER_API_KEY",
        context_window=200_000,
        max_output=16_384,
        usd_per_mtok_input=Decimal("0.60"),
        usd_per_mtok_output=Decimal("2.20"),
        usd_per_mtok_cached=Decimal("0.11"),
        supports_tools=True,
        supports_cache=True,
    )
    return Registry(models={_MODEL_ID: entry}, source=None)


def _build_deps(repo_root: Path, *, kb: KbService | None) -> LoopDeps:
    """Assemble a `LoopDeps` bundle scoped to `repo_root`, varying only
    `kb` across this suite's cases."""
    return LoopDeps(
        client=_UnusedClient(),
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=repo_root,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=repo_root),
        meter=CostMeter(),
        kb=kb,
    )


def _fake_kb_service(repo_root: Path) -> KbService:
    """A `KbService` this suite only ever compares by identity -- never
    calls `search`/`add_note`, so its own collaborators need not be
    real."""
    return KbService(
        repo_root=repo_root,
        config=KbConfig(),
        embedding_client=_UnusedEmbeddingClient(),
        embedding_model_id="fake-embed",
        embedding_dim=4,
    )


@pytest.fixture
def stub_kb_calls(monkeypatch: pytest.MonkeyPatch) -> list[KbService | None]:
    """Register `_STUB_TOOL_NAME` in the shared tool registry for the
    duration of one test, its executor appending whatever it receives
    as `kb` to the returned list -- the one seam this suite needs to
    prove `deps.kb` actually reaches a dispatched tool call's own
    `context`, since no real tool declares that parameter yet.
    `monkeypatch` restores the original `_TOOLS`/`_BY_NAME` once the
    test ends, so no other suite ever sees this stub tool registered.
    """
    received: list[KbService | None] = []

    def _stub_execute(
        args: _StubArgs, *, repo_root: Path, kb: KbService | None = None
    ) -> str:
        """Record `kb` and return a fixed, otherwise-uninteresting
        result string."""
        received.append(kb)
        return "stub-ok"

    binding = _ToolBinding(
        _STUB_SCHEMA, _parse_stub_args, _stub_execute, _StubToolError
    )
    monkeypatch.setattr(registry_module, "_TOOLS", (*registry_module._TOOLS, binding))
    monkeypatch.setattr(
        registry_module,
        "_BY_NAME",
        {**registry_module._BY_NAME, _STUB_TOOL_NAME: binding},
    )
    return received


@pytest.mark.sanity
def test_kb_left_at_its_default_is_none(tmp_path: Path) -> None:
    """Given `LoopDeps` built without naming `kb` at all, when read,
    then it is `None` -- identical to every field predating it."""
    deps = LoopDeps(
        client=_UnusedClient(),
        registry=_registry(),
        model_id=_MODEL_ID,
        repo_root=tmp_path,
        approval=ApprovalManager(),
        undo=UndoManager(repo_root=tmp_path),
        meter=CostMeter(),
    )

    assert deps.kb is None


def test_dispatch_tool_call_threads_deps_kb_into_the_dispatched_context(
    tmp_path: Path, stub_kb_calls: list[KbService | None]
) -> None:
    """Given `deps.kb` set to a real `KbService`, when a call naming the
    stub tool (whose executor declares a keyword-only `kb` parameter) is
    dispatched, then that executor receives `deps.kb` exactly, by
    identity -- not a copy, not `None`."""
    kb = _fake_kb_service(tmp_path)
    deps = _build_deps(tmp_path, kb=kb)
    event = ToolCallEvent(id="call-1", name=_STUB_TOOL_NAME, arguments_json="{}")

    result = loop_module._dispatch_tool_call(event, deps=deps, turn_id=1, task_id="t-1")

    assert result.content == "stub-ok"
    assert stub_kb_calls == [kb]
    assert stub_kb_calls[0] is kb


def test_dispatch_tool_call_threads_a_none_kb_when_unset(
    tmp_path: Path, stub_kb_calls: list[KbService | None]
) -> None:
    """Given `deps.kb` left at its default `None`, when a call naming
    the stub tool is dispatched, then that executor receives `None` --
    the knowledge base being disabled is never silently mistaken for a
    missing context entry."""
    deps = _build_deps(tmp_path, kb=None)
    event = ToolCallEvent(id="call-2", name=_STUB_TOOL_NAME, arguments_json="{}")

    result = loop_module._dispatch_tool_call(event, deps=deps, turn_id=1, task_id="t-1")

    assert result.content == "stub-ok"
    assert stub_kb_calls == [None]
