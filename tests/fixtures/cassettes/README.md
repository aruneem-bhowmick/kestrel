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
