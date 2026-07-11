# kestrel

<p align="center">
  <img src="assets/kestrel-mascot.svg" alt="Pixel-art kestrel perched on a branch, blinking and occasionally shaking">
</p>

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

`kestrel.provider.complete_with_retry()` wraps any `ProviderClient.complete()`
call with bounded exponential backoff and full jitter, retrying only
`RateLimitError` and `ServerError` -- never `AuthError` or
`ContextOverflowError`, since neither improves on a later attempt with the
same inputs -- and only when the failure occurs before any event has
reached the caller, so a partially-consumed stream is never replayed.
A `RateLimitError` carrying an explicit `retry_after_s` is honored
verbatim in place of the computed delay. `RetryPolicy` controls the
attempt count and backoff bounds; retrying across multiple models or
backends is out of scope for this wrapper.

## Tools

Kestrel offers a model a small, fixed set of capabilities -- read,
search, edit, and execute -- rather than an open-ended plugin surface.
Every tool declares its own `ToolSchema` and returns its result already
wrapped by `kestrel.security.frame_untrusted`, so a model never has to
guess whether a byte it receives back is data or an instruction.
`kestrel.tools.registry` is where all of them are wired together:
`all_schemas()` collects every tool's schema for a provider call's own
`tools=` argument, and `dispatch()` routes one returned tool call to
its bound parser and executor, framing an unknown tool name, a
malformed argument payload, or a caught tool error into an ordinary
result rather than letting any of them crash the calling loop. Adding a
tool means adding it to that module's own registration list -- nothing
else needs to change.

- **`read_file`** -- reads a UTF-8 text file, or a 1-indexed inclusive
  line range within it, from a repo-relative path. Refuses a path that
  escapes the repository root (including by following a symlink out of
  it), a missing file, a directory, and binary content; a whole-file read
  with no line range given is capped at 64 KiB, truncated with a note
  naming how much was cut.
- **`search`** -- runs a regex pattern against file contents under the
  repo (or a repo-relative subdirectory scope) via `rg`, and returns
  matched lines in deterministic file order, capped at a caller-supplied
  result count. Requires `rg` (ripgrep) on `PATH`; refuses a scope that
  escapes the repository root and a pattern `rg` rejects as invalid
  regex. A search that matches nothing is a normal result, not an error.
- **`execute`** -- runs a command (an argv list, never a shell string)
  in a `bwrap` sandbox scoped to the repo: the rest of the filesystem is
  mounted read-only, the repo and a fresh scratch directory are
  read-write, and the network namespace is unshared, so the command
  always runs with networking disabled. Returns the command's stdout,
  stderr, and exit code;
  a command that runs past its timeout is reported back as timed out
  rather than left to hang. Requires `bwrap` (bubblewrap) on `PATH`. A
  command recognized as `delete`, `chmod`, or a force-flagged
  `git push` is gated behind interactive approval before it runs (see
  [Approval](#approval)).
- **`edit_file`** -- replaces one exact, unique occurrence of an anchor
  string in a UTF-8 text file with new text. An anchor that is absent,
  or that occurs more than once, is refused rather than guessed at, and
  the file is left untouched either way. Every real edit is written to
  disk and then recorded to the undo journal (see [State](#state))
  before the call returns; passing `dry_run: true` instead returns a
  unified diff of the change without writing anything or touching the
  journal. Does not create new files -- an anchor checked against a
  path with no file on disk is refused the same way a missing anchor
  is, never treated as a request to create that file.

More tools land here as they're implemented.

## State

Kestrel writes its own runtime state under `<target-repo>/.kestrel/`: an
append-only undo journal today (`kestrel.managers.UndoManager`), with
artifacts and session logs joining it later. Target repos should
gitignore that path.

## Approval

`kestrel.managers.ApprovalManager` gates five kinds of destructive
action -- `delete`, `force_push`, `chmod`, `network_on`, and
`out_of_repo_write` -- behind interactive approval. The last two are
recognized but dormant this release: `execute`'s sandbox always runs
with networking disabled, and `edit_file` refuses an out-of-repo write
outright rather than offering it as an approvable escalation, so
neither has a live call path into the manager yet. `execute` is the
first live caller, classifying its command against a small pattern
table (`rm`/`rmdir`, `chmod`, and a force-flagged `git push`) and
calling `ApprovalManager.check()` for any match before the command
reaches the sandbox; a command outside that table runs unchecked.

Every check is either short-circuited -- the action's kind is in the
per-repo allowlist, or was already approved `"always"` earlier in the
session -- or handed to a decision function, which defaults to a plain
terminal prompt printing the action's summary and exact detail, then
reading one reply: `y`/`yes` approves just that one request (`"once"`),
`always` approves it and every later request of the same kind for the
rest of the session, and anything else -- including an empty reply --
denies it, raising `ApprovalDenied`.

Pre-approve specific kinds for an entire session via
`[managers.approval] allowlist` in `kestrel.toml`:

```toml
[managers.approval]
allowlist = ["delete"]
```

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

`kestrel doctor` runs eight checks and prints one aligned line per check,
in order: (1) `python-version` checking the interpreter version, (2) `config`
checking whether the configuration file loads, (3) `registry` checking
whether the model registry loads, (4) `default-model` checking whether the
default model exists in the registry, (5) `api-key` checking whether its
credential environment variable is set, (6) `endpoint` probing the default
model's live completion API, (7) `sandbox` checking whether the `execute`
tool's `bwrap` sandbox is usable, and (8) `ollama` as a placeholder for the
unimplemented Ollama backend. By default, the run is read-only and skips
the `endpoint` check; passing `--live` runs the endpoint check, which performs
a networked, potentially billable completion. A typical all-green run against
a fresh checkout looks like:

```text
OK    python-version  3.12
OK    config          ./kestrel.toml
OK    registry        2 models
OK    default-model   glm-5.2
OK    api-key         OPENROUTER_API_KEY
SKIP  endpoint        pass --live
OK    sandbox         bwrap
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

## Jetson quickstart

Deploying to an NVIDIA Jetson Orin Nano? See
[`docs/provisioning-jetson.md`](docs/provisioning-jetson.md) for the full
flash-to-REPL walkthrough, and run `scripts/jetson-flightcheck.sh` (add
`--ci-mode` off-device) to confirm the board's environment preconditions
before installing Kestrel itself.
