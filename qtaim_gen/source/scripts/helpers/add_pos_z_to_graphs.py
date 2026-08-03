"""
Patch existing graph LMDBs with atom positions and atomic numbers.

Joins each graph's ``mol_name`` (stored by the grapher, byte-identical to its
structure.lmdb key) back to one or more structure LMDBs and attaches:

    graph["atom"].pos : float32 (n_atoms, 3), cartesian coords
    graph["atom"].z   : int64   (n_atoms,),   atomic numbers

This produces the same graphs as rebuilding with the pos/z-aware grapher
(qtaim_embed.data.grapher.build_graph), without re-running the converter.

Modeled on qtaim_gen.source.utils.scaling.apply_scalers_to_lmdb_inplace:
streams the source read-only, writes to a temp LMDB in batches, then
os.replace()s atomically. A crash leaves the original untouched. Graphs that
already carry ``pos`` are copied through verbatim, so re-runs are no-ops.

Any key miss or atom-count mismatch aborts the run before the atomic replace:
both indicate a real join problem, never something to skip.

Usage:
    python -m qtaim_gen.source.scripts.helpers.add_pos_z_to_graphs \
        --graph_lmdbs graphs/train/shard_0.lmdb graphs/train/shard_1.lmdb \
        --structure_lmdbs data/OMol4M_lmdbs/tm_react/structure.lmdb \
        [--batch_size 2000] [--dry_run]
"""

import argparse
import os
import pickle

import lmdb
import torch
from tqdm import tqdm

from qtaim_embed.data.lmdb import load_graph_from_serialized, serialize_graph
from qtaim_gen.source.core.converter import clean_id
from qtaim_gen.source.utils.scaling import LMDB_MAP_SIZE, _is_metadata


def _open_readonly(path: str) -> lmdb.Environment:
    return lmdb.open(
        path, subdir=False, readonly=True, lock=False,
        readahead=True, meminit=False,
    )


class StructureLookup:
    """mol_name -> (coords, atomic_numbers) across one or more structure LMDBs.

    Tries the direct ascii key first. On a miss, lazily builds a
    clean_id(key) -> key map per env to cover converter runs that stored
    quote-stripped ids (converter.clean_id), then retries.
    """

    def __init__(self, structure_paths: list[str]):
        if not structure_paths:
            raise ValueError("at least one structure LMDB is required")
        self.envs = [(_open_readonly(p), p) for p in structure_paths]
        self._cleaned_maps: list[dict[str, bytes]] | None = None

    def _get_raw(self, mol_name: str):
        key = mol_name.encode("ascii")
        for env, _ in self.envs:
            with env.begin(write=False) as txn:
                raw = txn.get(key)
            if raw is not None:
                return raw
        if self._cleaned_maps is None:
            self._cleaned_maps = []
            for env, _ in self.envs:
                cmap = {}
                with env.begin(write=False) as txn:
                    for k, _v in txn.cursor():
                        if k == b"length":
                            continue
                        cmap[clean_id(k)] = k
                self._cleaned_maps.append(cmap)
        for (env, _), cmap in zip(self.envs, self._cleaned_maps):
            orig_key = cmap.get(mol_name)
            if orig_key is not None:
                with env.begin(write=False) as txn:
                    return txn.get(orig_key)
        return None

    def get(self, mol_name: str, n_atoms_expected: int):
        raw = self._get_raw(mol_name)
        if raw is None:
            raise RuntimeError(
                f"mol_name {mol_name!r} not found in any structure LMDB: "
                f"{[p for _, p in self.envs]}"
            )
        molecule = pickle.loads(raw)["molecule_graph"].molecule
        if len(molecule) != n_atoms_expected:
            raise RuntimeError(
                f"atom count mismatch for {mol_name!r}: structure has "
                f"{len(molecule)} atoms, graph has {n_atoms_expected}"
            )
        pos = torch.tensor(molecule.cart_coords, dtype=torch.float32)
        z = torch.tensor(molecule.atomic_numbers, dtype=torch.long)
        return pos, z

    def close(self):
        for env, _ in self.envs:
            env.close()


def patch_lmdb(
    graph_lmdb_path: str,
    lookup: StructureLookup,
    batch_size: int = 2000,
    dry_run: bool = False,
) -> dict:
    """Attach pos/z to every graph in one LMDB. Returns counts."""
    tmp_path = graph_lmdb_path + ".posz.tmp"
    for p in (tmp_path, tmp_path + "-lock"):
        if os.path.exists(p):
            os.remove(p)

    src = _open_readonly(graph_lmdb_path)
    dst = None
    if not dry_run:
        dst = lmdb.open(
            tmp_path, map_size=LMDB_MAP_SIZE, subdir=False,
            meminit=False, map_async=True,
        )

    patched = skipped = 0
    buf: list[tuple[bytes, bytes]] = []

    def _flush():
        if not buf or dry_run:
            buf.clear()
            return
        with dst.begin(write=True) as wtxn:
            for k, v in buf:
                wtxn.put(k, v)
        buf.clear()

    try:
        with src.begin(write=False) as rtxn:
            total = rtxn.stat()["entries"]
            progress = tqdm(
                rtxn.cursor(), total=total,
                desc=os.path.basename(graph_lmdb_path), unit="graph",
            )
            for key_bytes, value_bytes in progress:
                if _is_metadata(key_bytes, set()):
                    buf.append((key_bytes, value_bytes))
                    continue

                obj = pickle.loads(value_bytes)
                is_dict = isinstance(obj, dict)
                graph_bytes = obj["molecule_graph"] if is_dict else obj
                graph = load_graph_from_serialized(graph_bytes)

                if "pos" in graph["atom"]:
                    skipped += 1
                    buf.append((key_bytes, value_bytes))
                else:
                    pos, z = lookup.get(graph.mol_name, graph["atom"].num_nodes)
                    graph["atom"].pos = pos
                    graph["atom"].z = z
                    new_bytes = serialize_graph(graph, ret=True)
                    payload = {"molecule_graph": new_bytes} if is_dict else new_bytes
                    buf.append((key_bytes, pickle.dumps(payload, protocol=-1)))
                    patched += 1

                if len(buf) >= batch_size:
                    _flush()
            _flush()
    except Exception:
        src.close()
        if dst is not None:
            dst.close()
            for p in (tmp_path, tmp_path + "-lock"):
                if os.path.exists(p):
                    os.remove(p)
        raise

    src.close()
    if not dry_run:
        dst.sync()
        dst.close()
        os.replace(tmp_path, graph_lmdb_path)
        stale_lock = tmp_path + "-lock"
        if os.path.exists(stale_lock):
            os.remove(stale_lock)

    return {"patched": patched, "skipped_has_pos": skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Attach atom.pos/atom.z to existing graph LMDBs by "
        "joining mol_name against structure LMDBs."
    )
    parser.add_argument(
        "--graph_lmdbs", nargs="+", required=True,
        help="Graph LMDB file(s) to patch in place (merged, shard, or split).",
    )
    parser.add_argument(
        "--structure_lmdbs", nargs="+", required=True,
        help="structure.lmdb file(s) covering every mol_name in the graphs.",
    )
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Scan and validate the join without writing anything.",
    )
    args = parser.parse_args()

    for path in args.graph_lmdbs + args.structure_lmdbs:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    lookup = StructureLookup(args.structure_lmdbs)
    try:
        for graph_path in args.graph_lmdbs:
            counts = patch_lmdb(
                graph_path, lookup,
                batch_size=args.batch_size, dry_run=args.dry_run,
            )
            tag = "[dry run] " if args.dry_run else ""
            print(
                f"{tag}{graph_path}: patched {counts['patched']}, "
                f"already had pos {counts['skipped_has_pos']}"
            )
    finally:
        lookup.close()


if __name__ == "__main__":
    main()
