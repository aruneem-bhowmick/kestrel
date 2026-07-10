"""Defenses against hostile content flowing into the model.

Houses the untrusted-data framing primitive
(:func:`~kestrel.security.framing.frame_untrusted`).
"""

from kestrel.security.framing import SourceKind, frame_untrusted

__all__ = [
    "SourceKind",
    "frame_untrusted",
]
