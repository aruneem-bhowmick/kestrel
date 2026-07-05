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
USD-per-million-token rates. On startup Kestrel checks `./models.toml`
and a per-user config directory before falling back to the registry
bundled with the package (`src/kestrel/data/models.default.toml`), which
ships two GLM-5.2 routes (OpenRouter and Z.ai direct). Every entry is
validated at load time; misconfigured entries fail with a message naming
the file, the entry, and the field at fault.
