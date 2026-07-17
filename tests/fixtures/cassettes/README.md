# Cassettes

Each `.sse` file here is a literal recording of an OpenAI-compatible chat
completions streaming response: one `data: {json}` line per chunk, blank
lines between them, terminated by `data: [DONE]`. `tests/fixtures/mock_openai.py`
serves a cassette's raw bytes verbatim over HTTP with a
`text/event-stream` content type, regardless of the request that hit it --
it is a fixed recording, not a request-aware simulator.

The final chunk in a cassette carries the turn's `usage` and an empty
`choices` list, matching how OpenAI-compatible backends emit token counts
when a request sets `stream_options.include_usage`. Every other chunk
carries one `choices[0].delta` fragment and (on the last content chunk) a
`finish_reason`.

A `cassette_sequence` passed to `MockOpenAIServer` may mix cassette paths
with bare `int` status codes: an `int` entry fails that one request with
a fixed error body at that status code instead of replaying a cassette,
letting a script express "fail with 429, then succeed" as
`[429, some_cassette]` without a second server or a code change to this
module.

## Current cassettes

- `openrouter_glm52_hello.sse` -- a three-chunk "Hello from GLM-5.2" reply
  with `usage` (`prompt_tokens=42`, `completion_tokens=7`, `cached_tokens=0`)
  and `finish_reason="stop"`.
- `openrouter_glm52_ansi.sse` -- the same shape, but one chunk's `content`
  carries a screen-clear escape sequence and a terminal-title OSC sequence
  embedded between two chunks of ordinary text, for exercising terminal
  output sanitization against a realistic adversarial completion.
- `zai_glm52_hello.sse` -- the zai-backend counterpart to the OpenRouter
  "hello" cassette: a four-chunk "Hello from Z.ai GLM" reply with `usage`
  (`prompt_tokens=40`, `completion_tokens=6`, `cached_tokens=0`) and
  `finish_reason="stop"`.
- `toolcall_read_file.sse` -- a single `read_file` tool call: the id and
  name arrive in the first chunk, `arguments` (`{"path": "src/greet.py"}`)
  is split across the next two chunks, then `finish_reason="tool_calls"`
  and the usage chunk.
- `toolcall_execute_pytest.sse` -- the same shape as
  `toolcall_read_file.sse`, naming an `execute` tool call with
  `arguments` `{"cmd": ["pytest", "-q"], "timeout_s": 30}` split across
  two chunks.
- `done_no_more_tools.sse` -- an ordinary text-only "Task complete."
  reply with `finish_reason="stop"`, standing in for a model declaring a
  multi-turn task finished with no further tool calls.
- `toolcall_edit_greet.sse` -- a single `edit_file` tool call, whole
  JSON arguments in one chunk rather than split, replacing a
  `"# TODO: implement greet"` anchor with a real `greet` function body.
- `toolcall_execute_rm.sse` -- a single `execute` tool call, whole JSON
  arguments in one chunk, naming `{"cmd": ["rm", "somefile"]}` -- a
  destructive command for exercising the approval gate.
- `toolcall_read_file_payload.sse` -- a single `read_file` tool call,
  whole JSON arguments in one chunk, naming `{"path": "payload.txt"}` --
  reusable across every corpus case in the injection acceptance suite,
  since only the fixture file's own on-disk content changes per case,
  never this tool call's arguments.
- `toolcall_verify.sse` -- a single `verify` tool call, whole (empty)
  JSON arguments `{}` in one chunk, standing in for a model asking to
  run every command KESTREL.md configures rather than narrowing via
  `only`.
- `compaction_summary.sse` -- an ordinary text-only "Summary: ..." reply
  with `usage` (`prompt_tokens=42`, `completion_tokens=18`,
  `cached_tokens=0`) and `finish_reason="stop"`, standing in for a
  compaction call's own model response -- token counts distinct from
  every other cassette's, so a system test can tell which turn priced
  which call.
- `toolcall_edit_farewell.sse` -- a single `edit_file` tool call, whole
  JSON arguments in one chunk, replacing a `"# TODO: implement
  farewell"` anchor with a real `farewell` function body -- the second
  of two files a multi-file acceptance scenario edits, alongside
  `toolcall_edit_greet.sse`'s own `greet.py`.
- `toolcall_edit_ansi_payload.sse` -- a single `edit_file` tool call,
  whole JSON arguments in one chunk, naming `payload.txt` and replacing
  a `"before"` anchor with `"BEFORE-EDITED"` -- for a fixture file whose
  own content is the `ansi_escape_laden_payload` injection-corpus
  case's payload, so the diff pane's rendering of the resulting
  mutation can be checked for leftover raw escape bytes.
- `cache_hit_turn1_cold.sse` -- a `read_file` tool call with `usage`
  (`prompt_tokens=100`, `completion_tokens=15`, `cached_tokens=0`),
  standing in for the first turn of a session, before any prior turn's
  prefix exists to hit against.
- `cache_hit_turn2_warm.sse` -- a `read_file` tool call with `usage`
  (`prompt_tokens=150`, `completion_tokens=15`, `cached_tokens=120`, an
  80% hit rate), standing in for a later turn against a cache-capable
  backend once the stable prefix has already been primed.
- `cache_hit_turn3_done.sse` -- an ordinary text-only closing reply with
  `usage` (`prompt_tokens=150`, `completion_tokens=12`,
  `cached_tokens=120`), the same 80% hit rate as
  `cache_hit_turn2_warm.sse`, so a three-turn session's aggregate ratio
  clears the 50% alert threshold comfortably.
- `budget_toolcall_big.sse` -- a `read_file` tool call with a
  deliberately huge `usage.prompt_tokens=500000` (`completion_tokens=0`,
  `cached_tokens=0`), sized so two turns at a registry entry's round
  per-token rate cross a small USD budget cap on clean dollar amounts,
  mirroring `tests/unit/test_p031_budget_wiring.py`'s own token-count
  convention.
- `budget_done_small.sse` -- an ordinary text-only "Task complete."
  reply with a deliberately small `usage` (`prompt_tokens=100`,
  `completion_tokens=10`, `cached_tokens=0`), standing in for the turn
  that closes out a task after a budget degrade or a resume, priced low
  enough not to itself threaten whatever cap the scenario configured.

## Re-recording the zai cassette

The zai backend has no environment-variable seam the way OpenRouter does
(see `KESTREL_OPENROUTER_BASE_URL`): a test simply builds a registry entry
whose `endpoint` field is the mock server's `base_url` directly, since
that field is exactly what a real deployment would set to a genuine Z.ai
endpoint. Re-recording `zai_glm52_hello.sse` (or adding a new zai
cassette) needs no special handling beyond the general recipe below --
only the registry entry construction differs, not the cassette format.

## Recording a new cassette

There is no live-capture tool in this suite (a recorded cassette is meant
to be a small, hand-authored, deterministic fixture, not a byte-for-byte
capture of a real API response). To add one:

1. Decide the exact text, token counts, and stop reason you want the
   fixture to represent.
2. Write one `chat.completion.chunk` JSON object per line, each prefixed
   with `data:` and a space, and followed by a blank line, splitting the
   reply text across as many `delta.content` fragments as you like.
3. End the content chunks with one carrying an empty `delta` and the
   `finish_reason` you want (`"stop"`, `"tool_calls"`, or `"length"`).
4. Add one final chunk with an empty `choices` list and a populated
   `usage` object (`prompt_tokens`, `completion_tokens`, and optionally
   `prompt_tokens_details.cached_tokens`).
5. Terminate the file with a `data: [DONE]` line.

Any control characters (escape sequences, etc.) belong in the JSON as
`\uXXXX` escapes rather than raw bytes, so the file stays a plain,
diffable text file. A quick way to generate a cassette correctly is to
build the chunk dicts in a short Python script and write
`"data: " + json.dumps(chunk)` per line -- `json.dumps` escapes control
characters for you.

### Tool-call chunks

A chunk that carries a tool call instead of (or alongside) text uses
`delta.tool_calls` in place of, or next to, `delta.content`:

```json
{"index": 0, "id": "call_...", "type": "function", "function": {"name": "...", "arguments": "..."}}
```

`id`, `type`, and `function.name` only need to appear once, typically on
the first chunk that introduces the call; every chunk contributes its
`function.arguments` string, concatenated in arrival order, so the full
JSON arguments payload can be split across as many chunks as you like --
including a single whole-JSON chunk, since the normalizer handles both.
Follow the tool-call chunks with the usual `finish_reason: "tool_calls"`
chunk and the closing usage chunk, exactly as for a text completion.
