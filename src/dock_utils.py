#!/usr/bin/env python3
"""Small shared naming/parsing helpers for the GPCR docking pipeline.

Used by the analytics dashboard (``app.py``) and the post-screen triage
(``prescreen.py``), so the compound-name derivation lives in a single place.
"""
import re

# The chemical-series prefix is the leading run of letters in a base compound name,
# e.g. "CPD3" -> "CPD". Compiled once and reused.
_SERIES_RE = re.compile(r"^([A-Za-z]+)")


def base_compound(name: str) -> str:
    """Strip the docking variant suffix from a compound name.

    ``run_screen.py`` expands each input compound into ``{name}_var{N}`` (the top
    stereo/tautomer variants) or ``{name}_fallback`` (when 3D embedding fails). This
    recovers the base compound by dropping the first ``_var``/``_fallback`` and
    everything after it::

        "CPD3_var1"     -> "CPD3"
        "ABC4_fallback" -> "ABC4"
        "CPD3"          -> "CPD3"
    """
    return str(name).split("_var")[0].split("_fallback")[0]


def series_of(base: str) -> str:
    """Return the chemical-series prefix (leading alphabetic run) of a base name.

    ``"CPD3" -> "CPD"``, ``"ABC4" -> "ABC"``. If the name has no leading letters, the whole
    (stringified) name is returned unchanged.
    """
    m = _SERIES_RE.match(str(base))
    return m.group(1) if m else str(base)
