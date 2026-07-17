# kestrel

<p align="center">
  <img src="assets/kestrel.svg" alt="Pixel-art kestrel perched on a branch: blinking, shaking, and flapping">
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
last message this function annotates. Whether an entry needs that marker
is `ModelEntry.requires_explicit_cache_breakpoint`, a plain per-entry
flag rather than a hardcoded backend name -- onboarding a backend that
needs one is a registry change, not a code change.

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
task outright -- see "Compaction" below for how context-window
pressure is recovered from before that ever happens, and why it can
still happen despite that.

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

Setting `LoopDeps.observer` to a `kestrel.agent.observer.LoopObserver`
makes a running task's own progress externally observable, live,
rather than only inspectable once it ends: seven hooks fire as the
task runs -- a turn starting, each streamed text chunk, a tool call
starting and finishing, a fresh `VerificationReport` landing, a turn's
own priced cost settling, and the task's own termination. Every call
happens synchronously, inline, on the same coroutine driving the task,
so an observer must stay fast and exception-free -- nothing it returns
is ever read, and nothing it does can change what the loop decides
next. Leaving `observer` unset (the default) wires in an all-no-op
`NullLoopObserver` and behaves exactly as it did before this field
existed.

### Compaction

Context-window pressure is recovered from, not just detected. Once a
turn's own billed prompt size (`deps.meter.turns[-1].input_tokens`)
reaches 70% of the active model's `context_window`, the next
iteration's pre-check folds everything in `history` but the most
recent few messages into one model-generated summary via
`kestrel.agent.compaction.compact_history`, before spending another
real model call. That call is explicitly instructed to preserve the
task's own goal, any plan or TODO language the conversation already
contains, and the most recent verification outcome, rather than
inventing new steps of its own -- the folded-away messages are
replaced by a single `"system"`-role message holding that summary.

The summarization call itself is priced, journaled (when
`LoopDeps.session` is set), and checked against `LoopDeps.budget` (when
set) exactly like any other turn -- sharing its own journal record's
turn id with the real turn that immediately follows it, and never
incrementing `LoopResult.turns_used`, which counts only real model
calls -- and it can itself end the task outright (a hard budget halt or
the cumulative token cap) before that following turn's own model call
is ever made. This reduces how often a task ends `CONTEXT_OVERFLOW`,
not the possibility of it: a single message that alone exceeds the
window still ends the task that way, compaction or not. Neither the
70% threshold nor how many trailing messages are kept verbatim is
configurable via `kestrel.toml` yet -- both are fixed constants.

## Running a task

`kestrel run "<task>" --repo PATH` drives the agent loop against a real
repository from the command line, non-interactively:

```sh
kestrel run "add a function + unit test, make it pass" --repo /path/to/repo
```

It resolves config, registry, starting model, and the target repo's own
`KESTREL.md` exactly like the plain REPL does (the same
`--config`/`--model` flags, in either position around the subcommand),
builds a fresh `ApprovalManager`, `UndoManager`, `SessionManager`, and
`CostMeter` for this run alone, and calls `run_task` to completion.
`--max-turns`, `--max-total-tokens`, and `--max-wall-clock-s` override
`LoopLimits`'s own defaults. A destructive tool call still prompts on
the real terminal exactly as described under [Approval](#approval); a
piped, non-interactive answer works identically to a typed one.

`--require-verification`/`--no-require-verification`
(default: **enabled**) sets `LoopDeps.require_verification` -- with it
on, a task only completes once the most recent `verify` call passed
(see [Agent loop](#agent-loop)); pass `--no-require-verification` for a
task whose repo has no configured `verify` commands, or that should
still complete on the model's own say-so as it did before this flag
existed.

`--session-budget-usd`, `--day-budget-usd`, `--month-budget-usd`, and
`--budget-soft-threshold` build the run's own `BudgetManager` (see
[Budget](#budget)), each falling back to `kestrel.toml`'s own
`[managers.budget]` table when left unset; day and month spend baselines
are computed once, up front, from every other task's own journaled
history under this repo. Because every run now also builds a
`SessionManager`, a task that halts `BUDGET_HALT` can always be picked
back up with `kestrel run --resume TASK_ID --repo PATH` instead of the
usual `kestrel run "<task>" --repo PATH` invocation -- same flags
otherwise, continuing the halted task's own history, cost, and turn
count rather than starting over.

When it finishes, `kestrel run` prints a terse summary -- the task id
(needed to undo this run later), the termination reason, the turn
count, the total priced cost, a cache-hit line once at least one turn
has recorded real usage, and every distinct path the run's own undo
journal recorded a mutation for -- then exits `0` if the task reached
`TASK_COMPLETE` and `1` for every other termination reason:

```text
task_id: 3b1e7b7a-...
reason: TASK_COMPLETE
turns: 3
total_usd: $0.0071
cache_hit: 62%
files changed:
  src/greet.py
```

A `BUDGET_HALT` termination prints an abbreviated summary instead --
still the task id and reason line, but the turn count, cost, cache-hit,
and files-changed lines are replaced by a dedicated message naming which
cap tripped and the exact command to resume:

```text
task_id: 3b1e7b7a-...
reason: BUDGET_HALT
budget halt: session cap reached; resume with: kestrel run --resume 3b1e7b7a-... --repo /path/to/repo
```

`kestrel undo --task-id ID --repo PATH` reverts every file mutation a
prior run recorded under that id, restoring each touched path to its
exact pre-task content and printing what it reverted. Reverting twice
in a row is safe (see [State](#state)): the second call targets the
first's own compensating journal entry and simply toggles the file back.

## TUI

Running `kestrel` with no subcommand mounts a Textual-based cockpit
instead of the plain REPL: a conversation pane streaming the active
task's assistant text, an artifact viewer for the task's most recently
produced artifact, a collapsible tool log of tool calls and their
outcomes, a diff view for the most recent file mutation, and a one-line
status bar docked to the top of the screen. A 2fr-wide left column
holds the conversation pane above a task-input box; a 1fr-wide right
column stacks the artifact, tool-log, and diff panes.

`F1`-`F4` jump focus to the task-input box, the tool log, the diff
pane, and the artifact pane in turn; `ctrl+q` quits. The artifact pane
shows the task's most recent `VerificationReport`, rendered as Markdown
with `kestrel.tools.verify.render_verification_markdown` and sanitized
before display;
it shows static placeholder content until a task's own `verify` tool
call produces its first report.

Submitting text in the task-input box always runs a full task through
`run_task` -- the same tool-calling agent loop `kestrel run` drives --
never a plain chat turn: the conversation pane renders the assistant's
own text incrementally at newline boundaries as it arrives, the status
bar refreshes after every turn, and the run ends with a terse one-line
summary naming the termination reason, turn count, and total cost.
`kestrel.repl`'s own `run_turn`/`ReplSession`/`run_repl` remain in the
codebase, unchanged and still tested, for the plain non-interactive
REPL path -- they are simply not what the cockpit itself drives.

The status bar renders one line from a `StatusSnapshot` value --
active model, mode and effort, context-window usage, and session/day
spend against their caps:

```text
{model_id} · {mode}/{effort} · ctx {pct}% ({used}/{window}) · session ${session_usd:.4f}{cap} · day ${day_usd:.4f}{cap}
```

`ctx` renders as `--% (--/{window})` before any turn has billed. Each
`{cap}` segment renders ` / cap $X.XXXX` when that scope's budget cap
is configured, and is omitted entirely (a bare `$X.XXXX`) when it
isn't -- the same "`None` means no cap" convention the cost meter and
budget manager use throughout. `StatusBar.show(snapshot)` is the
widget's own hook onto this rendering; a fresh `KestrelApp` shows an
idle snapshot on mount, and `TuiLoopObserver` (see below) keeps it
current for the rest of a submitted task's own run.

Every submitted task builds its own collaborators -- provider client,
approval gate, undo journal, cost meter, session journal, and budget
manager -- through `kestrel.task_setup.build_task_deps`, the same
function `kestrel run` itself now delegates to. Extracting that
construction out from under the CLI's own `argparse.Namespace` means
the cockpit and the CLI build an identical bundle from one shared
place, rather than two copies that could quietly drift apart from each
other over time.

`kestrel.tui.observer_bridge.TuiLoopObserver` is the bridge between a
running task and the cockpit's own widgets: it is handed to
`build_task_deps` as that task's observer, and its hooks fire
synchronously, inline, on the same coroutine driving the task, so
calling widget methods directly from inside them is safe. A turn
starting or finishing refreshes the status bar; each streamed chunk of
assistant text is sanitized and appended to the conversation pane's own
currently streaming line; a tool call starting or finishing writes a
line to the tool log and toggles the loading indicator; an `edit_file`
call that actually mutated a file renders that mutation in the diff
pane; a `verify` tool call's own `VerificationReport` renders in the
artifact pane; and the task's own termination writes a terse summary
line.

Every tool call a running task makes writes two lines to the
collapsible tool log: `-> {name}({summary})` when it starts, where
`summary` is the call's own (sanitized) JSON arguments capped at 120
characters with a trailing `...` when longer, and `<- {name}
({elapsed_s:.1f}s)` once it finishes. A small loading indicator, docked
directly under the status bar, is visible for as long as at least one
tool call is in flight and hidden the rest of the time. Whenever an
`edit_file` call actually changes a file, the diff pane re-renders to
show that mutation as a syntax-highlighted unified diff, built from the
undo journal's own before/after content for the change -- there is no
history to browse back through; only the single most recent mutation
is ever shown, and a fresh one simply replaces whatever the pane showed
before.

The default theme ("kestrel") is a restrained rust-and-slate palette
defined entirely as ordinary Textual CSS variables in
`src/kestrel/tui/kestrel.tcss`. Customize it by editing that file's
rules directly, or by pointing a `KestrelApp` subclass's own
`CSS_PATH` at a different stylesheet -- either way, no changes to the
app's own Python code are required.

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

## Mutation testing

`uv run mutmut run` checks that the automated suite would actually catch
a broken change to the two modules where a subtly wrong formula or
predicate is easiest to miss and costliest to ship silently:
`kestrel/cost/meter.py` (the pricing arithmetic) and
`kestrel/agent/loop.py` (the loop's own termination, verification-gate,
and budget predicates), configured via `[tool.mutmut]`'s `only_mutate`.
It systematically mutates small pieces of each -- flipping a comparison,
swapping an operator -- and reruns the targeted subset of the suite
(`uv run pytest -m 'p007 or p022 or p026 or p028 or p031 or p032 or
(p033 and not live)'`, configured via `pytest_add_cli_args_test_selection`)
against every mutant; a mutant that still passes is a "survivor" worth
triaging, since it means some change to that logic would ship
unnoticed. `p033` is in that set alongside the unit-level markers
because its own acceptance suite is what exercises the verification
gate and both budget thresholds through a real `kestrel run` invocation
rather than a scripted `LoopDeps` call, so a mutant that only a full
CLI run would catch does not slip through -- `and not live` excludes
its own opt-in live smoke test, which needs real credentials and would
otherwise get a network-backed run of its own for every single mutant.

This is a manually invoked quality check, not a CI gate -- mutation
testing's own runtime cost (many reruns of the targeted suite, one per
mutant) makes it unsuited to running on every commit the way the sanity
and coverage gates do. Run it locally before a change to either module
lands, and triage any survivor it reports: either the suite is missing a
case that would have caught it, or the surviving mutant is behaviorally
equivalent to the original and can be marked accordingly. `mutmut` itself
only runs under WSL on Windows -- see its own upstream note if `mutmut
run` reports no native Windows support.

## Docstring coverage

`uv run interrogate` checks that `src/kestrel` and `tests` stay
documented as the codebase grows, failing if coverage drops below the
80% floor set in `[tool.interrogate]`. Like `mutmut`, this is a manually
invoked quality check rather than a CI gate -- unlike `mutmut`, its
runtime cost is negligible, so there's no reason not to run it alongside
`ruff`/`mypy` before a change lands. Pass `-v` for a per-file breakdown
of exactly which functions, classes, or modules are missing a docstring.

## Jetson quickstart

Deploying to an NVIDIA Jetson Orin Nano? See
[`docs/provisioning-jetson.md`](docs/provisioning-jetson.md) for the full
flash-to-REPL walkthrough, and run `scripts/jetson-flightcheck.sh` (add
`--ci-mode` off-device) to confirm the board's environment preconditions
before installing Kestrel itself.
