"""Defenses against hostile content flowing into the model.

Houses the untrusted-data framing primitive
(:func:`~kestrel.security.framing.frame_untrusted`) and the loader for
the adversarial injection corpus used to test it
(:func:`~kestrel.security.corpus.load_corpus`).
"""

from kestrel.security.corpus import InjectionCase, load_corpus
from kestrel.security.framing import SourceKind, frame_untrusted

__all__ = [
    "InjectionCase",
    "SourceKind",
    "frame_untrusted",
    "load_corpus",
]
