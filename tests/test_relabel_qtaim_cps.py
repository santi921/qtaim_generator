"""relabel-qtaim-cps: nuclear CPs filed under a close same-element neighbour are relabeled, bond keys too."""

import json
import os
import time

import pytest

from qtaim_gen.source.scripts.helpers import relabel_qtaim_cps as rq

INP = "! UKS wB97M-V\n*xyz 0 1\nC 0.0 0.0 0.0\nH 1.09 0.0 0.0\nH 1.5 0.62 0.0\nO -1.2 0.0 0.0\n*\n"
ATOMS = {0: ("C", [0.0, 0.0, 0.0]), 1: ("H", [1.09, 0.0, 0.0]), 2: ("H", [1.5, 0.62, 0.0]), 3: ("O", [-1.2, 0.0, 0.0])}


def _ncp(atom, rho, dx=0.01):
    el, pos = ATOMS[atom]
    return {"cp_num": atom + 1, "element": el, "number": str(atom + 1), "pos_ang": [pos[0] + dx, pos[1], pos[2]],
            "density_all": rho}


def _bcp(a, b, rho):
    pa, pb = ATOMS[a][1], ATOMS[b][1]
    return {"pos_ang": [(x + y) / 2 for x, y in zip(pa, pb)], "density_all": rho}


# what a correct parse gives
RIGHT = {"0": _ncp(0, 120.0), "1": _ncp(1, 0.42), "2": _ncp(2, 0.31), "3": _ncp(3, 300.0),
         "0_1": _bcp(0, 1, 0.28), "0_2": _bcp(0, 2, 0.20), "1_2": _bcp(1, 2, 0.15), "0_3": _bcp(0, 3, 0.40)}
# the old mapper: atoms 1 and 2 swapped CPs, and every bond key through the same map
SWAPPED = {"0": RIGHT["0"], "1": RIGHT["2"], "2": RIGHT["1"], "3": RIGHT["3"],
           "0_2": RIGHT["0_1"], "0_1": RIGHT["0_2"], "1_2": RIGHT["1_2"], "0_3": RIGHT["0_3"]}


def _job(tmp_path, record=SWAPPED, root_copy=None, inp=True):
    job = tmp_path / "job"
    (job / "generator").mkdir(parents=True)
    (job / "generator" / "qtaim.json").write_text(json.dumps(record))
    if root_copy is not None:
        (job / "qtaim.json").write_text(json.dumps(root_copy))
    if inp:
        (job / "orca.inp").write_text(INP)
    return job


def _stored(job, rel="generator/qtaim.json"):
    return json.loads((job / rel).read_text())


def _run(job, dry_run=False):
    return rq.process_folder(str(job), None, None, dry_run=dry_run)


class TestPermutation:

    def test_clean_swap(self):
        assert rq.permutation(SWAPPED, ATOMS) == {1: 2, 2: 1}

    def test_correct_record_has_no_moves(self):
        assert rq.permutation(RIGHT, ATOMS) == {}

    def test_relabel_restores_nuclear_and_bond_keys(self):
        assert rq.relabel(SWAPPED, rq.permutation(SWAPPED, ATOMS)) == RIGHT

    def test_cp_on_an_atom_whose_own_cp_stayed_is_not_clean(self):
        dup = dict(RIGHT, **{"1": RIGHT["2"]})
        assert rq.permutation(dup, ATOMS) is None

    def test_different_element_is_not_clean(self):
        cross = dict(RIGHT, **{"0": RIGHT["3"], "3": RIGHT["0"]})
        assert rq.permutation(cross, ATOMS) is None

    def test_multiwfn_label_must_agree(self):
        wrong_label = dict(SWAPPED, **{"1": dict(RIGHT["2"], number="2")})
        assert rq.permutation(wrong_label, ATOMS) is None

    def test_cp_off_every_atom_is_not_clean(self):
        off = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[1.3, 0.3, 0.0])})
        assert rq.permutation(off, ATOMS) is None

    def test_self_pairs(self):
        assert rq.self_pairs(dict(RIGHT, **{"2_2": _bcp(2, 2, 0.2)})) == 1
        assert rq.self_pairs(RIGHT) == 0


class TestCycles:
    """Three close H atoms: the old map could rotate CPs among them, and pi is then not its own inverse."""

    A3 = {0: ("C", [0.0, 0.0, 0.0]), 1: ("H", [1.09, 0.0, 0.0]), 2: ("H", [1.5, 0.62, 0.0]),
          3: ("H", [1.6, -0.5, 0.0])}

    def _cp(self, atom, rho):
        el, pos = self.A3[atom]
        return {"element": el, "number": str(atom + 1), "pos_ang": [pos[0] + 0.01, pos[1], pos[2]], "density_all": rho}

    def _bond(self, a, b, rho):
        return {"connected_bond_paths": [a + 1, b + 1], "density_all": rho}

    def test_three_cycle(self):
        right = {"0": self._cp(0, 120.0), "1": self._cp(1, 0.42), "2": self._cp(2, 0.31), "3": self._cp(3, 0.36),
                 "0_1": self._bond(0, 1, 0.28), "0_2": self._bond(0, 2, 0.20), "0_3": self._bond(0, 3, 0.22),
                 "1_2": self._bond(1, 2, 0.15)}
        pi = {1: 2, 2: 3, 3: 1}
        # the old mapper: stored nuclear key k holds atom pi(k)'s CP; real bond r-s stored under pi(r)_pi(s)
        stored = {str(k): right[str(pi.get(k, k))] for k in range(4)}
        for key in ("0_1", "0_2", "0_3", "1_2"):
            r, s_ = (int(x) for x in key.split("_"))
            i, j = sorted((pi.get(r, r), pi.get(s_, s_)))
            stored[f"{i}_{j}"] = right[key]
        assert rq.permutation(stored, self.A3) == pi
        assert rq.relabel(stored, pi) == right

    def test_nan_position_is_not_clean(self):
        nan = dict(RIGHT, **{"1": dict(RIGHT["1"], pos_ang=[float("nan"), 0.0, 0.0])})
        assert rq.permutation(nan, ATOMS) is None


class TestFolder:

    def test_relabels_and_a_second_pass_is_clean(self, tmp_path):
        job = _job(tmp_path)
        r = _run(job)
        assert r["status"] == rq.STATUS_RELABELED
        assert r["permutation"] == {os.path.join("generator", "qtaim.json"): {"1": 2, "2": 1}}
        assert _stored(job) == RIGHT
        assert _run(job)["status"] == rq.STATUS_CLEAN

    def test_both_copies(self, tmp_path):
        job = _job(tmp_path, root_copy=SWAPPED)
        assert _run(job)["status"] == rq.STATUS_RELABELED
        assert _stored(job) == RIGHT and _stored(job, "qtaim.json") == RIGHT

    def test_one_unclean_copy_blocks_the_folder(self, tmp_path):
        job = _job(tmp_path, root_copy=dict(RIGHT, **{"1": RIGHT["2"]}))
        assert _run(job)["status"] == rq.STATUS_NOT_CLEAN
        assert _stored(job) == SWAPPED

    def test_dry_run_writes_nothing_and_takes_no_lock(self, tmp_path):
        job = _job(tmp_path)
        (job / ".processing.lock").write_text("other job")
        assert _run(job, dry_run=True)["status"] == rq.STATUS_WOULD_RELABEL
        assert _stored(job) == SWAPPED and (job / ".processing.lock").read_text() == "other job"

    def test_an_old_lock_is_respected(self, tmp_path):
        job = _job(tmp_path)
        lock = job / ".processing.lock"
        lock.write_text("stalled heavy job")
        old = time.time() - 3 * 86400
        os.utime(lock, (old, old))
        assert _run(job)["status"] == rq.STATUS_LOCKED
        assert lock.exists() and _stored(job) == SWAPPED

    def test_self_pairs_are_reported_after_the_relabel(self, tmp_path):
        job = _job(tmp_path, record=dict(SWAPPED, **{"1_1": _bcp(1, 1, 0.2)}))
        r = _run(job)
        assert r["status"] == rq.STATUS_RELABELED and r["self_pairs"] == 1
        assert "2_2" in _stored(job)

    @pytest.mark.parametrize("kwargs,status", [({"inp": False}, rq.STATUS_NO_INP)])
    def test_no_geometry(self, tmp_path, kwargs, status):
        job = _job(tmp_path, **kwargs)
        assert _run(job)["status"] == status and _stored(job) == SWAPPED

    def test_missing_and_no_record(self, tmp_path):
        assert _run(tmp_path / "nope")["status"] == rq.STATUS_MISSING
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _run(empty)["status"] == rq.STATUS_NO_QTAIM_JSON

    def test_an_error_is_failed_and_releases_the_lock(self, tmp_path, monkeypatch):
        job = _job(tmp_path)
        monkeypatch.setattr(rq, "_plan", lambda *a: (_ for _ in ()).throw(OSError("ESTALE")))
        r = _run(job)
        assert r["status"] == rq.STATUS_FAILED and "ESTALE" in r["error"]
        assert not (job / ".processing.lock").exists()


class TestMain:

    def _main(self, tmp_path, monkeypatch, *extra):
        jobs = []
        for name, record in (("a", SWAPPED), ("b", RIGHT), ("c", dict(SWAPPED, **{"1_1": _bcp(1, 1, 0.2)})),
                             ("d", dict(RIGHT, **{"1": RIGHT["2"]}))):
            job = _job(tmp_path / name, record=record)
            jobs.append(str(job))
        lst = tmp_path / "list.txt"
        lst.write_text("\n".join(jobs) + "\n")
        report, remaining = tmp_path / "report.json", tmp_path / "remaining.txt"
        monkeypatch.setattr("sys.argv", ["relabel-qtaim-cps", "--folder_list", str(lst), "--report", str(report),
                                         "--list_remaining", str(remaining), *extra])
        assert rq.main() == 0
        return jobs, json.loads(report.read_text())["aggregate"], remaining.read_text().split()

    @pytest.mark.parametrize("workers", ["1", "2"])
    def test_counts_and_remaining(self, tmp_path, monkeypatch, workers):
        jobs, agg, remaining = self._main(tmp_path, monkeypatch, "--workers", workers)
        assert (agg["relabeled"], agg["clean"], agg["not_clean"], agg["with_self_pairs"]) == (2, 1, 1, 1)
        assert sorted(remaining) == sorted([jobs[2], jobs[3]])

    def test_dry_run(self, tmp_path, monkeypatch):
        jobs, agg, _ = self._main(tmp_path, monkeypatch, "--workers", "1", "--dry_run")
        assert agg["would_relabel"] == 2 and agg["relabeled"] == 0
        assert json.loads(open(os.path.join(jobs[0], "generator", "qtaim.json")).read()) == SWAPPED
