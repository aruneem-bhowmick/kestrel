# kestrel

## Install (dev)

```sh
git clone https://github.com/aruneem-bhowmick/kestrel.git
cd kestrel
uv sync
uv run pytest -m "not live and not e2e"
```

## Configuration

Kestrel reads settings from `kestrel.toml`. On startup it checks, in
order, an explicit `--config` path, `$KESTREL_CONFIG`, `./kestrel.toml`,
and a per-user config directory, stopping at the first one it finds; if
none exist it falls back to built-in defaults. Files are never merged
across these layers. See `src/kestrel/data/kestrel.default.toml` for every
recognized key and its default value. Secrets (API keys, tokens,
passwords) must never be placed in this file -- set them as environment
variables instead.

## Models

Available models live in `models.toml`, an array of `[[models]]` tables
each naming a Kestrel-stable id, backend, provider-side model name, and
USD-per-million-token rates. On startup Kestrel checks, in order, an
explicit registry path passed to `load_registry()`, `./models.toml`, and
a per-user config directory, stopping at the first one it finds; if none
exist it falls back to the registry bundled with the package
(`src/kestrel/data/models.default.toml`), which ships two GLM-5.2 routes
(OpenRouter and Z.ai direct). Files are never merged across these layers.
Every entry is validated at load time; misconfigured entries fail with a
message naming the file, the entry, and the field at fault.

## Provider layer

Every backend adapter implements one interface,
`kestrel.provider.ProviderClient.complete()`, and streams a normalized
event sequence rather than its own wire format: zero or more
`TextDelta`/`ToolCallEvent` events, then exactly one `UsageEvent`, then
exactly one `StopEvent` as the final event --
`kestrel.provider.validate_stream_order()` checks a sequence against this
grammar. On failure, a call raises a typed `ProviderError` subclass
(`AuthError`, `RateLimitError`, `ContextOverflowError`, `ServerError`)
naming the active model id and backend, instead of emitting a stop event.
No call site outside an adapter names a vendor.

`kestrel.provider.LiteLLMClient` is the first concrete adapter: it routes
any registry entry with `backend = "openrouter"` through LiteLLM's
OpenAI-compatible streaming interface, normalizing OpenRouter's chunks and
errors into that same grammar and taxonomy. It reads its API key from the
entry's `api_key_env`, never from a config file, and fails with `AuthError`
before making any network call if that variable is unset or empty.
Integration tests redirect it to a local mock server via the
`KESTREL_OPENROUTER_BASE_URL` environment variable instead of the real
OpenRouter endpoint; this variable is inert unless set and has no effect
outside test runs. A registry entry with `backend = "zai"` routes through
the same client's OpenAI-compatible path against the entry's own
`endpoint` directly -- no environment-variable redirection, since the
registry itself already names where to call.

## Cost

Every completed turn is priced from the active model's own registry rates
using `kestrel.cost.compute_turn_cost`, which turns a `UsageEvent`'s
input/output/cached token counts into a `Decimal` USD amount (six decimal
places, `ROUND_HALF_EVEN`). `kestrel.cost.CostMeter` accumulates these
across a session, and `kestrel.cost.format_cost_line` renders the exact
line printed after each turn: input and output token counts, the turn's
own cost, and the running session total, each rounded to four decimal
places for display. A cache-hit count is only shown when nonzero, and a
turn with no usage at all still prints `$0.0000` rather than nothing --
a missing cost is meant to be noticed, not hidden.

## Flight check

`kestrel doctor` runs eight read-only checks and prints one aligned line
per check, in order: the interpreter version, whether the configuration
file loads, whether the model registry it points at loads, whether the
configured default model exists in that registry, and whether its
credential environment variable is set. Two more checks are placeholders
for integrations that do not exist in this codebase yet (a sandboxed
tool-execution environment, and an Ollama backend) and always report
`SKIP`. A typical all-green run against a fresh checkout looks like:

```text
OK    python-version  3.12
OK    config          ./kestrel.toml
OK    registry        2 models
OK    default-model   glm-5.2
OK    api-key         OPENROUTER_API_KEY
SKIP  endpoint        pass --live
SKIP  sandbox         sandboxed tool execution is not implemented
SKIP  ollama          the Ollama backend is not implemented
```

Each of the first five checks depends on the one before it; if one fails,
every check after it reports `SKIP` naming that same original failure
(`blocked by: <check>`) instead of re-deriving the same root cause under
a different name. A `FAIL` line's detail is the remedy -- a missing
config file names its path, a broken registry names the offending entry
and field, an unset credential names the environment variable (never its
value). `kestrel doctor` exits `0` unless at least one check `FAIL`s;
`SKIP` never affects the exit code.

Pass `--config PATH` (before or after `doctor`) to check a specific
configuration instead of the one that would otherwise be resolved. Pass
`--live` to also run the sixth check: a real, budget-capped (one output
token) completion against the configured default model, confirming the
backend actually answers. This is the only check that spends money or
touches the network, so it never runs unless explicitly requested --
never pass `--live` in an automated or CI context.
