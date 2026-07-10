# Injection corpus

Each `.json` file here is one adversarial fixture: a single hostile
payload paired with the metadata needed to assert that
`kestrel.security.framing.frame_untrusted` neutralizes it.
`kestrel.security.corpus.load_corpus` reads every file in this
directory and returns them, sorted by `id`, as `InjectionCase` values.

## Schema

Every file is a JSON object with exactly four keys:

- `id` (string) -- a stable slug identifying the case; must match the
  filename minus its `.json` extension, and must be unique across the
  corpus.
- `source` (string) -- one of `kestrel.security.framing.SourceKind`'s
  literal values (`"file"`, `"tool_stdout"`, `"tool_stderr"`,
  `"search_result"`, `"web"`): the kind of origin this case pretends to
  come from.
- `payload` (string) -- the hostile text itself, verbatim. This is
  exactly what a test passes as `frame_untrusted`'s `text` argument.
- `forbidden_markers` (array of strings) -- substrings that must never
  appear *unescaped* in a frame built from `payload`. Every case in
  this corpus uses the same pair, `"<<<UNTRUSTED:"` and
  `"<<<END_UNTRUSTED>>>"` (the two literal markers `frame_untrusted`
  itself emits) -- proving that after framing, each marker occurs
  exactly once in the output: the real one `frame_untrusted` appended
  or prepended itself, and never a second, forged occurrence smuggled
  in through the payload.

## Current cases

- `readme_ignore_previous_instructions` -- a hidden "ignore previous
  instructions" directive inside an HTML comment, styled as a
  README-style setup file a `read_file` call might return.
- `forged_closing_delimiter_system_turn` -- a payload that embeds the
  literal `<<<END_UNTRUSTED>>>` marker partway through, followed by a
  fake `<<<SYSTEM>>>` turn, attempting to convince the model the
  untrusted block already ended and a fresh, trusted turn has begun.
- `fake_destructive_action_success` -- a tool-output payload that
  falsely claims a destructive command (`rm -rf /`) already completed,
  then asks the model to follow up with a further, unrelated action.
- `zero_width_smuggled_instruction` -- an instruction with a zero-width
  space (U+200B) interposed between every letter, testing that framing
  does not depend on a hostile instruction being contiguous, readable
  text to be treated as data.
- `ansi_escape_laden_payload` -- the exact escape bytes from the
  existing `openrouter_glm52_ansi.sse` cassette (a CSI clear-screen
  sequence and an OSC window-title sequence), reused as a tool-output
  payload to prove untrusted-data framing and terminal-output sanitization
  (`kestrel.repl.sanitize_terminal`) are independent, complementary
  defenses rather than the same mechanism twice.
- `nested_untrusted_block_spoof` -- a search-result payload that
  contains a well-formed-looking `<<<UNTRUSTED:file:...>>>` /
  `<<<END_UNTRUSTED>>>` block of its own, attempting to make the model
  believe its own injected content is a second, separately-sourced
  trusted block rather than part of the original untrusted result.

## Adding a case

1. Pick a stable, descriptive `id` (kebab_case or snake_case; it
   becomes the filename).
2. Decide which `SourceKind` the case pretends to originate from.
3. Write the exact hostile `payload` text. Prefer generating it with a
   short Python script over hand-typing raw control or zero-width
   characters into the JSON source -- `json.dump(..., ensure_ascii=True)`
   escapes every non-ASCII and control byte as a portable `\uXXXX`
   sequence, so the checked-in file stays a plain, diffable, 7-bit-ASCII
   text file (see `ansi_escape_laden_payload.json` and
   `zero_width_smuggled_instruction.json` for examples).
4. Set `forbidden_markers` to `["<<<UNTRUSTED:", "<<<END_UNTRUSTED>>>"]`
   unless the case specifically needs a different assertion.
5. Save the object as `{id}.json`, matching the existing files' key
   order (`forbidden_markers`, `id`, `payload`, `source` -- alphabetical,
   from `json.dump(..., sort_keys=True)`), two-space indent, and a
   trailing newline.

No test needs updating to pick up a new case: `test_p013_corpus_loads.py`
iterates whatever `load_corpus()` returns, so adding a file alone grows
the corpus everywhere it is exercised.
