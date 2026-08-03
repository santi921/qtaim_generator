"""Compatibility shim for unpickling Compositions saved with older pymatgen.

pymatgen renamed ``Composition._natoms`` -> ``Composition._n_atoms`` between the
2023.x and 2024.10 releases. Any pickle containing a Composition (directly or
transitively via Site/Molecule/MoleculeGraph) produced under pymatgen ~2023
fails on newer pymatgen because ``num_atoms`` reads ``_n_atoms``. The error
surfaces as ``AttributeError: attr='is_ordered' not found on Site`` because
the property getter for ``Site.is_ordered`` raises and Python treats that as
attribute-missing.

The shim installs a ``__setstate__`` on Composition that copies the legacy
``_natoms`` value to ``_n_atoms`` when restoring stale pickles. Fresh pickles
already have ``_n_atoms`` and are unaffected. Duplicated here so qtaim_gen
can load legacy data without depending on qtaim_embed.
"""
from pymatgen.core.composition import Composition

_ORIGINAL_SETSTATE = getattr(Composition, "__setstate__", None)


def _composition_setstate(self, state):
    if _ORIGINAL_SETSTATE is not None:
        _ORIGINAL_SETSTATE(self, state)
    else:
        self.__dict__.update(state)
    if "_n_atoms" not in self.__dict__ and "_natoms" in self.__dict__:
        self.__dict__["_n_atoms"] = self.__dict__["_natoms"]


Composition.__setstate__ = _composition_setstate
