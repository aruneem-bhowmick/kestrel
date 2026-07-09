"""Regression guard: no vendor name leaks outside the LiteLLM adapter.

Every model is reached through :class:`kestrel.provider.base.ProviderClient`;
the backend a given model id actually routes to is an implementation
detail of the adapter, not something any other call site should ever
branch on by name. This is enforced by grepping the source tree rather
than by a runtime check, since the property being guarded is about what
code was *written*, not what it does when run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.p005, pytest.mark.p006, pytest.mark.regression]

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "kestrel"

# The adapter itself is the one place backend dispatch legitimately happens
# (KES-PRV-002's "no call site names a vendor" applies to everything else).
# The registry's own schema module legitimately enumerates the set of valid
# backend identifiers -- declaring that set is not the same as a call site
# choosing among them. The doctor module reports the diagnostic status of
# each registry-declared backend identifier by name (e.g. an "ollama" check
# reporting that integration is not implemented yet) -- a status label is
# not routing logic either. Packaged TOML data ships concrete backend
# values by design (and is out of scope for this guard regardless, since
# only *.py files are scanned below).
_EXCLUDED_PATHS = {
    _SRC_ROOT / "doctor.py",
    _SRC_ROOT / "provider" / "litellm_client.py",
    _SRC_ROOT / "registry" / "model.py",
}

# "zai" (bare, no dot) is the registry's own backend identifier, distinct
# from "z.ai" (the vendor's own name for itself); both are guarded here so
# neither form of the name can leak into a call site outside the adapter.
_VENDOR_NAME_PATTERN = re.compile(
    r"openrouter|z\.ai|zai|anthropic|ollama", re.IGNORECASE
)


def test_no_vendor_names_outside_adapter() -> None:
    """Given every Python source file in the package outside the excluded
    set, when scanned, then none of them mentions a vendor name -- proving
    backend selection stays confined to the adapter and the registry
    schema, exactly as it was written."""
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if path in _EXCLUDED_PATHS:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _VENDOR_NAME_PATTERN.search(line):
                offenders.append(
                    f"{path.relative_to(_SRC_ROOT.parent.parent)}:{lineno}: {line.strip()}"
                )

    assert offenders == []
