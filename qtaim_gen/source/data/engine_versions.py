"""
One version number for the whole descriptor engine (charge, surface and QTAIM engines), and
per version the routines it added or whose stored values it changed. Every engine-produced
routine records the ENGINE_VERSION it ran at in the job's provenance.json
(utils/provenance.py); a stored routine is current when that version is at least the last
version that added or changed it.

Rules: a new routine goes in "added" under a new version; a change that alters a routine's
stored values goes in "changed". Versions 1-3 are the earlier charge/surface-engine stamps
(timings.json "_engine_versions", which moved together); the QTAIM engine predates the first
stamp and enters under version 1. The native wavefunction converter is not versioned here:
its provenance entry records the mode, and a converter change that alters wavefunctions is a
new mode (omol.NATIVE_CONVERT_MODES), which validation and the converter check compare.
Stamps only exist since 2026-11-03, so changes before that (converged MBIS as the default,
2026-10-15; planarity in the surface engine, 2026-11-02) carry no version of their own.
"""

from qtaim_gen.source.data.multiwfn import (
    ENGINE_LEVEL2_ROUTINES,
    ENGINE_ROUTINES,
    SURFACE_ENGINE_LEVEL2_ROUTINES,
    SURFACE_ENGINE_ROUTINES,
)

ENGINE_VERSION = 4

# the sub-engine that produces each engine routine: the "by" of its provenance entry
ROUTINE_ENGINE = {
    **{r: "charge_engine" for r in ENGINE_ROUTINES - SURFACE_ENGINE_ROUTINES},
    **{r: "surface_engine" for r in SURFACE_ENGINE_ROUTINES},
    "qtaim": "qtaim_engine",
}
ENGINES = frozenset(ROUTINE_ENGINE.values())

_V2 = ENGINE_LEVEL2_ROUTINES | {"elf_fuzzy"}
ENGINE_CHANGES = {
    1: {"added": frozenset(ROUTINE_ENGINE) - _V2 - SURFACE_ENGINE_LEVEL2_ROUTINES, "changed": frozenset(),
        "note": "charge and surface engines, QTAIM engine (before version stamps)"},
    2: {"added": _V2, "changed": frozenset(),
        "note": "elf_fuzzy (full_set 1), gradient-norm and Laplacian fuzzy integrals (full_set 2)"},
    3: {"added": SURFACE_ENGINE_LEVEL2_ROUTINES, "changed": frozenset(),
        "note": "ESP surface (other_esp, full_set 2) in the surface engine"},
    4: {"added": frozenset(), "changed": frozenset(),
        "note": "one ENGINE_VERSION for all engines; provenance.json replaces the timings.json stamps"},
}


def last_change(routine: str) -> int:
    """The last version that added or changed this routine (0 if no version did: not an
    engine routine)."""
    return max((v for v, c in ENGINE_CHANGES.items() if routine in c["added"] | c["changed"]), default=0)


def is_current(routine: str, version) -> bool:
    """True when an engine routine recorded at this version needs no recompute."""
    last = last_change(routine)
    return bool(last) and isinstance(version, int) and version >= last


def routines_at(version: int) -> frozenset:
    """The engine routines a run of this version computed (added at or before it)."""
    return frozenset().union(*(c["added"] for v, c in ENGINE_CHANGES.items() if v <= version))
