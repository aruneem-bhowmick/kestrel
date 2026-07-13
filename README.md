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

## Project memory

Kestrel reads an optional `KESTREL.md` from the target repo's own root
-- free-form notes and conventions written by that repo's maintainers,
in the same per-repo-memory tradition as `CLAUDE.md`. Unlike
`kestrel.toml`/`models.toml`, it has exactly one location and no search
precedence: a repo either has one or it doesn't, and having none is a
normal outcome, not an error. Because it's authored by the repo's own
maintainers rather than fetched at runtime by a tool, its text is
trusted project memory -- it is never run through the untrusted-content
framing every `read_file`/`search`/`execute` result goes through.

A `KESTREL.md` may also carry one fenced ` ```kestrel-verify ` block: a
small TOML table naming up to three commands the repo wants to be checked,
any of which may be omitted:

```kestrel-verify
lint = "ruff check ."
build = "true"
test = "pytest -q"
```

The `verify` tool (see [Tools](#tools)) runs whichever of these are
configured and reports pass/fail back to the model.

## Models

Available models live in `models.toml`, an array of `[[models]]` tables
each naming a Kestrel-stable id, backend, provider-side model name, and
USD-per-million-token rates. On startup Kestrel checks, in order, an
explicit registry path passed to `load_registry()`, `./models.toml`, and
a per-user config directory, stopping at the first one it finds; if none
exist it falls back to the registry bundled with the package
(`src/kestrel/data/models.default.toml`), which ships two full-size
GLM-5.2 routes (OpenRouter and Z.ai direct) plus a smaller, `"cheap"`-
tagged OpenRouter route for the agent loop's own budget degradation to
fall back to. Files are never merged across these layers.
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

Both the agent loop and the REPL send the exact same leading messages,
byte-for-byte, on every turn of one task or session:
`kestrel.provider.cache.build_stable_prefix()` renders one system-role
message from the fixed system prompt, folding in the target repo's
`KESTREL.md` (loaded once, up front, and never reloaded mid-task) when
one exists. Sending an identical prefix turn over turn is what lets a
cache-capable backend actually reuse its cached compute instead of
reprocessing the whole thing from scratch on every call.
`mark_cache_breakpoints()` sits next to it as a dormant extension point:
no backend wired up today needs an explicit marker for where that
prefix ends, so it is currently a no-op, but a future adapter for a
backend that does need one can read `Message.cache_breakpoint` off the
last message this function annotates.

## Tools

Kestrel offers a model a small, fixed set of capabilities -- read,
search, edit, execute, and verify -- rather than an open-ended plugin
surface.
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
- **`verify`** -- runs whichever of the repo's own configured
  lint/build/test commands (see [Project memory](#project-memory)) apply,
  in that fixed order, through the same sandbox `execute` uses. Every
  configured command always runs to completion, even after an earlier
  one fails, so every check's own result comes back together. Returns a
  short pass/fail summary per command -- never a command's own
  stdout/stderr -- and separately renders and persists the full
  per-command output as markdown to
  `.kestrel/artifacts/verification-<task_id>-<turn_id>.md` (a numeric
  suffix is appended if that name is already taken, so calling `verify`
  more than once in the same task and turn never overwrites an earlier
  report). Pass `only` (an array of `"lint"`/`"build"`/`"test"`) to
  restrict a call to a
  subset of what's configured. Refuses to run when the repo has no
  KESTREL.md, or when nothing it configures matches what was asked for.
  Reloads KESTREL.md fresh on every call, since a prior turn's
  `edit_file` may have just changed it.

More tools land here as they're implemented.

## State

Kestrel writes its own runtime state under `<target-repo>/.kestrel/`: an
append-only undo journal (`kestrel.managers.UndoManager`), under
`.kestrel/artifacts/`, persisted verification reports from the `verify`
tool, and under `.kestrel/sessions/`, one append-only JSONL journal per
task (`kestrel.managers.SessionManager`) recording every turn's message
deltas, priced cost, and latest verification report. Target repos should
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

`CostMeter.cache_hit_ratio` divides the session's total cached input
tokens by its total input tokens, entirely in `Decimal` arithmetic, and
returns `None` before any turn with real usage has been recorded.
`CostMeter.cache_alert` turns that ratio into a one-line warning naming
the measured percentage, but only once a session is long enough and the
active model is cache-capable: the active model's `supports_cache` must
be `True`, at least three turns must have been recorded, and the ratio
itself must sit below 50%. A cache-incapable model or a session with
only one or two turns never warns -- a short session never had a prior
turn's prefix to hit against, and a backend without cache support was
never expected to hit at all. The REPL's `/cost` command prints this
warning, when present, after the per-turn table.

## Budget

`kestrel.managers.BudgetManager` classifies how much has been spent
against three independently configurable USD caps -- session, day, and
month -- returning whether each is under budget (`OK`), past a soft
warn/degrade threshold (`SOFT`), or past its hard halt threshold
(`HARD`). It is a pure classifier: it never reads a file or tracks
spend itself, only turns three already-computed `Decimal` totals
(session/day/month spend, however a caller chooses to compute them)
into a `BudgetEvent` naming the worst status across all three and,
when more than one cap ties at that status, the first one in
session/day/month priority order.

A cap of `None` never trips, regardless of spend. Any configured cap
trips `HARD` once spend reaches it, and `SOFT` once spend reaches
`cap * soft_threshold` (0.8 by default, i.e. 80% of the cap). Configure
caps and the soft-threshold fraction per repo via `[managers.budget]`
in `kestrel.toml`:

```toml
[managers.budget]
session_usd = 5.00
day_usd = 20.00
month_usd = 100.00
soft_threshold = 0.8
```

Any of the three caps may be left unset for "no cap" on that scope.

Setting `LoopDeps.budget` wires this classifier into the agent loop
itself -- see "Agent loop" below for how a `SOFT` or `HARD` result
actually changes a running task.

## Agent loop

`kestrel.agent.run_task` drives one task to completion through a
tool-calling loop, distinct from the plain-chat REPL: each turn calls
the model with the full tool set offered, sanity-checks what it
proposes via an injectable self-critique function, dispatches every
requested tool call through `kestrel.tools.registry.dispatch` (turning
a denied approval into a framed refusal instead of raising), and folds
the turn's usage into a running total. A turn that requests no tools at
all is the task's natural completion.

Every collaborator a task needs -- the provider client, the model
registry, the approval and undo managers, the cost meter, and the
task's own limits -- arrives through one `LoopDeps` bundle rather than
global state. `LoopLimits` bounds a task with three hard caps (turn
count, cumulative tokens, wall-clock time); crossing any of them ends
the task with the matching `TerminationReason` rather than running
unbounded, and a `KeyboardInterrupt` mid-task ends it the same way,
keeping whatever turns and cost had already accumulated. A
`ContextOverflowError` raised while streaming a turn also ends the
task outright -- there is no compaction yet to recover the window and
retry.

Not yet implemented: mode switching (every call runs at a single,
fixed effort level), a real self-critique model call (the default
always approves), artifact persistence, and subagents.

Whether the model's own say-so is enough to end a task is configurable
via `LoopDeps.require_verification` (default `False`, preserving the
behavior above unchanged). Set it `True` and a turn that requests no
tools only completes the task once the most recent report in
`LoopDeps.verification_reports` -- filled in as the `verify` tool runs
during the task -- actually passed; otherwise the loop folds a nudge to
call `verify` into history and keeps going, the same shape the
self-critique-skip path already uses. This never adds a new way for a
task to end: an unbounded task still stops only via the turn, token, or
wall-clock cap, exactly as it always could.

Setting `LoopDeps.session` to a `kestrel.managers.SessionManager` makes
a task resumable: every real turn is journaled as it completes, and
`kestrel.agent.resume_task(task_id, deps)` reconstructs a prior task's
history, cost meter, and last verification report from that journal and
continues driving it -- picking the turn-cap counter up from where the
journal left off, while sampling wall-clock budget fresh rather than
inheriting elapsed time from before the process restarted. A task run
without `session` set behaves exactly as before and cannot later be
resumed.

Setting `LoopDeps.budget` to a `kestrel.managers.BudgetManager` makes a
task check its own spend every turn: `spent_day_usd`/`spent_month_usd`
are fixed baselines the caller computes once, up front, before the task
starts, and the loop adds its own growing `meter.session_usd` on top of
them for each check. Crossing the soft threshold switches every
following turn to whichever registry entry is tagged `"cheap"` (the
packaged default registry now ships one), printing a warning either
way -- there is no TUI yet to show it in -- and a task degrades at most
once, to at most one cheaper entry, never cascading through further
tiers and never reverting to a costlier one even if spend later looks
fine. Crossing the hard threshold ends the task immediately with
`TerminationReason.BUDGET_HALT`; because every turn up to and including
the one that tripped it was already journaled (when `session` is also
set), `resume_task` picks it back up once an operator raises the cap.
Leaving `budget` unset (the default) skips every check above and
behaves exactly as it did before this field existed.

## Running a task

`kestrel run "<task>" --repo PATH` drives the agent loop against a real
repository from the command line, non-interactively:

```sh
kestrel run "add a function + unit test, make it pass" --repo /path/to/repo
```

It resolves config, registry, and starting model exactly like the plain
REPL does (the same `--config`/`--model` flags, in either position
around the subcommand), builds a fresh `ApprovalManager`, `UndoManager`,
and `CostMeter` for this run alone, and calls `run_task` to completion.
`--max-turns`, `--max-total-tokens`, and `--max-wall-clock-s` override
`LoopLimits`'s own defaults. A destructive tool call still prompts on
the real terminal exactly as described under [Approval](#approval); a
piped, non-interactive answer works identically to a typed one.

When it finishes, `kestrel run` prints a terse summary -- the task id
(needed to undo this run later), the termination reason, the turn
count, the total priced cost, and every distinct path the run's own
undo journal recorded a mutation for -- then exits `0` if the task
reached `TASK_COMPLETE` and `1` for every other termination reason:

```text
task_id: 3b1e7b7a-...
reason: TASK_COMPLETE
turns: 3
total_usd: $0.0071
files changed:
  src/greet.py
```

`kestrel undo --task-id ID --repo PATH` reverts every file mutation a
prior run recorded under that id, restoring each touched path to its
exact pre-task content and printing what it reverted. Reverting twice
in a row is safe (see [State](#state)): the second call targets the
first's own compensating journal entry and simply toggles the file back.

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
