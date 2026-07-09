#!/usr/bin/env bash
set -euo pipefail

# Verify the environment preconditions the Jetson provisioning guide
# (docs/provisioning-jetson.md) sets up, without changing anything.
#
# Every check below is read-only, so this script is safe to re-run at
# any time. Pass --ci-mode to downgrade the checks that only make sense
# on a physical Jetson board (the JetPack release file, MAXN power mode,
# and the NVMe mount) to SKIP, so the same script also runs unmodified
# on a non-Jetson CI runner while still verifying architecture, uv,
# Python, and credential presence.

CI_MODE=0
for arg in "$@"; do
  case "$arg" in
    --ci-mode)
      CI_MODE=1
      ;;
    *)
      echo "jetson-flightcheck.sh: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

FAILED=0

_pass() {
  printf 'PASS  %-11s  %s\n' "$1" "$2"
}

_fail() {
  printf 'FAIL  %-11s  %s\n' "$1" "$2"
  FAILED=1
}

_skip() {
  printf 'SKIP  %-11s  %s\n' "$1" "$2"
}

# 1. Architecture must be aarch64 -- the Orin Nano's native target. This
#    check always runs, even under --ci-mode, since a CI job asserting
#    the Jetson target still needs a real arm64 runner to be meaningful.
arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
  _pass arch "$arch"
else
  _fail arch "expected aarch64, found $arch"
fi

# 2. JetPack release file -- proof the board was flashed with L4T/JetPack.
if [[ "$CI_MODE" -eq 1 ]]; then
  _skip jetpack "--ci-mode"
elif [[ -f /etc/nv_tegra_release ]]; then
  _pass jetpack "$(tr -d '\n' < /etc/nv_tegra_release)"
else
  _fail jetpack "/etc/nv_tegra_release not found"
fi

# 3. Power mode must be MAXN (mode 0) for full CPU/GPU/memory clocks.
if [[ "$CI_MODE" -eq 1 ]]; then
  _skip power-mode "--ci-mode"
elif command -v nvpmodel >/dev/null 2>&1; then
  mode_line="$(nvpmodel -q 2>/dev/null | grep -i 'NV Power Mode' || true)"
  if [[ "$mode_line" == *MAXN* ]]; then
    _pass power-mode "MAXN"
  else
    _fail power-mode "expected MAXN, found: ${mode_line:-unknown}"
  fi
else
  _fail power-mode "nvpmodel not found"
fi

# 4. NVMe mounted -- Kestrel's repos, virtual environments, and session
#    logs all depend on it (see docs/provisioning-jetson.md).
if [[ "$CI_MODE" -eq 1 ]]; then
  _skip nvme "--ci-mode"
elif mountpoint -q /mnt/nvme 2>/dev/null; then
  _pass nvme "/mnt/nvme"
else
  _fail nvme "/mnt/nvme is not a mount point"
fi

# 5. uv must be on PATH -- Kestrel's only supported packaging/env tool.
if command -v uv >/dev/null 2>&1; then
  _pass uv "$(command -v uv)"
else
  _fail uv "not found on PATH"
fi

# 6. python3.12 must resolve through uv's own managed toolchain.
if ! command -v uv >/dev/null 2>&1; then
  _fail python "uv not found; cannot resolve python 3.12"
elif python_path="$(uv python find 3.12 2>/dev/null)"; then
  _pass python "$python_path"
else
  _fail python "python 3.12 not resolvable via 'uv python find 3.12'"
fi

# 7. At least one provider credential must be present. Names only -- a
#    flight check never prints a secret value.
present=()
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  present+=("OPENROUTER_API_KEY")
fi
if [[ -n "${ZAI_API_KEY:-}" ]]; then
  present+=("ZAI_API_KEY")
fi
if [[ "${#present[@]}" -gt 0 ]]; then
  _pass api-key "${present[*]}"
else
  _fail api-key "neither OPENROUTER_API_KEY nor ZAI_API_KEY is set"
fi

exit "$FAILED"
