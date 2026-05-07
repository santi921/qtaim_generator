"""Per-source structure loaders for SOAP featurization.

Every loader yields ``(structure_id: str, atoms: ase.Atoms)`` tuples.

If ``max_n`` is set, the loader returns at most that many structures, sampled
deterministically with ``seed`` (uniform across the source). Loaders with a
known total count pre-pick indices; loaders that have to stream use reservoir
sampling.

OMol uses ``data/OMol4M_lmdbs/<vertical>/structure.lmdb`` (pymatgen Molecules).
The four comparator loaders consume the on-disk subsamples documented in
``data/comparators/README.md``.
"""

from __future__ import annotations

import gzip
import io
import json
import lzma
import pickle
import random
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import h5py
import lmdb
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as ASE_Z
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    iterate: Callable[..., Iterator[tuple[str, Atoms]]]
    default_root: Path
    description: str


# ---------------- helpers ---------------- #


def _sample_indices(total: int, max_n: Optional[int], seed: int) -> Optional[set[int]]:
    if max_n is None or max_n >= total:
        return None
    rng = random.Random(seed)
    return set(rng.sample(range(total), max_n))


def _reservoir(
    iterator: Iterator[tuple[str, Atoms]], k: int, seed: int
) -> list[tuple[str, Atoms]]:
    rng = random.Random(seed)
    reservoir: list[tuple[str, Atoms]] = []
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def _atoms_from_zs_pos(zs, pos) -> Atoms:
    return Atoms(numbers=list(zs), positions=np.asarray(pos, dtype=float))


# ---------------- omol ---------------- #


def iter_omol(
    root: Path,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> Iterator[tuple[str, Atoms]]:
    """Yield (key, ase.Atoms) over an OMol vertical's structure.lmdb.

    OMol structure records are pymatgen Molecules under the "molecule" key.
    Sampling uses reservoir sampling because LMDB length is the only cheap
    count; we still enumerate keys, but never read pickle blobs we will throw
    away.
    """
    path = root / "structure.lmdb"
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin(write=False) as txn:
            cur = txn.cursor()
            keys = [
                raw.decode("ascii")
                for raw, _ in cur
                if raw.decode("ascii", errors="replace") != "length"
            ]
        keep_idx = _sample_indices(len(keys), max_n, seed)
        with env.begin(write=False) as txn:
            for i, key in enumerate(keys):
                if keep_idx is not None and i not in keep_idx:
                    continue
                raw = txn.get(key.encode("ascii"))
                if raw is None:
                    continue
                rec = pickle.loads(raw)
                mol = rec["molecule"]
                zs = [site.species.elements[0].Z for site in mol]
                yield key, _atoms_from_zs_pos(zs, mol.cart_coords)
    finally:
        env.close()


# ---------------- pcqm4mv2 ---------------- #


def iter_pcqm4mv2(
    root: Path,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> Iterator[tuple[str, Atoms]]:
    """Yield (idx, ase.Atoms) over the gzipped subsampled SDF.

    Each V2000 mol block is delimited by ``$$$$``. Records are 1:1 with
    `pcqm4m-v2-train-subsample10.sdf.gz` ordering (which is itself the seed=42
    10% subsample of the upstream 3.38M training SDFs).
    """
    sdf_path = root / "pcqm4m-v2-train-subsample10.sdf.gz"
    # known total from the subsample log; verified by counting at scan time.
    total = 337_861
    keep_idx = _sample_indices(total, max_n, seed)

    with gzip.open(sdf_path, "rb") as f:
        suppl = Chem.ForwardSDMolSupplier(f, removeHs=False, sanitize=False)
        for i, mol in enumerate(suppl):
            if keep_idx is not None and i not in keep_idx:
                continue
            if mol is None:
                continue
            conf = mol.GetConformer(0) if mol.GetNumConformers() else None
            if conf is None:
                continue
            zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
            pos = [list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]
            yield f"pcqm4mv2_{i}", _atoms_from_zs_pos(zs, pos)


# ---------------- qmugs ---------------- #


def iter_qmugs(
    root: Path,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> Iterator[tuple[str, Atoms]]:
    """Yield (CHEMBL_id/conf_NN, ase.Atoms) over the subsampled tarball."""
    tar_path = root / "structures_subsample10.tar.gz"
    total = 199_298
    keep_idx = _sample_indices(total, max_n, seed)

    with tarfile.open(tar_path, "r:gz") as tf:
        i = 0
        for member in tf:
            if not member.isfile() or not member.name.endswith(".sdf"):
                continue
            if keep_idx is not None and i not in keep_idx:
                i += 1
                continue
            f = tf.extractfile(member)
            if f is None:
                i += 1
                continue
            mol = Chem.MolFromMolBlock(
                f.read().decode("utf-8", errors="replace"),
                sanitize=False,
                removeHs=False,
            )
            if mol is not None and mol.GetNumConformers():
                conf = mol.GetConformer(0)
                zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
                pos = [list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]
                # member.name = "structures/CHEMBL123/conf_05.sdf"
                stem = Path(member.name).with_suffix("").as_posix()
                yield stem.replace("structures/", "qmugs_"), _atoms_from_zs_pos(zs, pos)
            i += 1


# ---------------- qm7x ---------------- #


def iter_qm7x(
    root: Path,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> Iterator[tuple[str, Atoms]]:
    """Yield (mol/conformer, ase.Atoms) across all eight subsampled HDF5 shards."""
    shards = sorted(root.glob("*_subsample10.xz"))
    if not shards:
        raise FileNotFoundError(f"no QM7-X subsample shards found under {root}")

    # Two-pass: first count, then sample. Counting is cheap because we just
    # decompress and read group keys (no array data).
    with tempfile.TemporaryDirectory() as td:
        manifest: list[tuple[Path, str, str]] = []  # (shard, mol_id, conf_id)
        for s in shards:
            tmp_h5 = Path(td) / f"{s.stem}.h5"
            with lzma.open(s, "rb") as fxz, open(tmp_h5, "wb") as out:
                while True:
                    b = fxz.read(1024 * 1024)
                    if not b:
                        break
                    out.write(b)
            with h5py.File(tmp_h5, "r") as h:
                for mol_id in h:
                    for conf_id in h[mol_id]:
                        manifest.append((tmp_h5, mol_id, conf_id))

        total = len(manifest)
        keep_idx = _sample_indices(total, max_n, seed)

        # Group by shard for sequential file open.
        kept = (
            manifest
            if keep_idx is None
            else [m for i, m in enumerate(manifest) if i in keep_idx]
        )
        kept.sort(key=lambda t: (str(t[0]), t[1], t[2]))

        current_h5: Optional[h5py.File] = None
        current_path: Optional[Path] = None
        try:
            for shard_h5, mol_id, conf_id in kept:
                if current_path != shard_h5:
                    if current_h5 is not None:
                        current_h5.close()
                    current_h5 = h5py.File(shard_h5, "r")
                    current_path = shard_h5
                grp = current_h5[mol_id][conf_id]
                zs = list(grp["atNUM"][:])
                pos = grp["atXYZ"][:]
                key = f"qm7x_{mol_id}/{conf_id}"
                yield key, _atoms_from_zs_pos(zs, pos)
        finally:
            if current_h5 is not None:
                current_h5.close()


# ---------------- schnet4aim ---------------- #


def _atoms_from_xyz_path(p: Path) -> Iterator[tuple[str, Atoms]]:
    """Iterate frames of an extended-XYZ trajectory."""
    with open(p) as f:
        frame = 0
        while True:
            header = f.readline()
            if not header:
                return
            n_atoms = int(header.strip())
            comment = f.readline()  # noqa: F841
            zs = []
            pos = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                sym = parts[0]
                xyz = [float(x) for x in parts[1:4]]
                zs.append(ASE_Z[sym])
                pos.append(xyz)
            yield f"{p.stem}_frame{frame}", _atoms_from_zs_pos(zs, pos)
            frame += 1


def iter_schnet4aim(
    root: Path,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> Iterator[tuple[str, Atoms]]:
    """Yield (name, ase.Atoms) across both JSON databases plus the 3 XYZ trajs.

    Total ~4,884 records: 3,865 electronic + 1,016 energetic + 13P-CO2/300K
    trajectory (95-atom frames) + 13P-CO2/900K + chemical_reaction/IRC.
    """
    items: list[tuple[str, Atoms]] = []
    pt = Chem.GetPeriodicTable()

    for fname, prefix in [("electronic.json", "elec"), ("energetic.json", "energ")]:
        p = root / "examples" / "databases" / fname
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        for i in range(len(d["natoms"])):
            zs = [pt.GetAtomicNumber(s) for s in d["ele"][i]]
            pos = d["pos"][i]
            name = d["name"][i]
            items.append((f"schnet4aim_{prefix}_{name}", _atoms_from_zs_pos(zs, pos)))

    for rel in [
        "examples/extrapolation/13P-CO2/300K_trj.xyz",
        "examples/extrapolation/13P-CO2/900K_trj.xyz",
        "examples/extrapolation/chemical_reaction/IRC.xyz",
    ]:
        p = root / rel
        if p.exists():
            for key, atoms in _atoms_from_xyz_path(p):
                items.append((f"schnet4aim_{key}", atoms))

    if max_n is None or max_n >= len(items):
        for x in items:
            yield x
        return

    rng = random.Random(seed)
    for x in rng.sample(items, max_n):
        yield x


# ---------------- registry ---------------- #


LOADERS: dict[str, SourceSpec] = {
    "omol": SourceSpec(
        name="omol",
        iterate=iter_omol,
        default_root=Path("data/OMol4M_lmdbs/droplet"),
        description="OMol4M structure.lmdb (pymatgen Molecule) per vertical.",
    ),
    "pcqm4mv2": SourceSpec(
        name="pcqm4mv2",
        iterate=iter_pcqm4mv2,
        default_root=Path("data/comparators/pcqm4mv2/raw"),
        description="OGB PCQM4Mv2 10% subsample SDF (training set 3D).",
    ),
    "qmugs": SourceSpec(
        name="qmugs",
        iterate=iter_qmugs,
        default_root=Path("data/comparators/qmugs/raw"),
        description="QMugs 10% subsample of conformer SDFs.",
    ),
    "qm7x": SourceSpec(
        name="qm7x",
        iterate=iter_qm7x,
        default_root=Path("data/comparators/qm7x/raw"),
        description="QM7-X 10%/molecule subsample HDF5 shards.",
    ),
    "schnet4aim": SourceSpec(
        name="schnet4aim",
        iterate=iter_schnet4aim,
        default_root=Path("data/comparators/schnet4aim/raw"),
        description="SchNet4AIM electronic + energetic JSONs + 3 extrapolation XYZ.",
    ),
}
