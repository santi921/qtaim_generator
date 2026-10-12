"""Exact electronic ESP at Multiwfn CP positions with gbasis (horton env).

usage: ~/miniconda3/envs/horton/bin/python -I esp_ref.py <wfx> <CPprop.txt> <out.npz>
Electronic part only (no nuclear term), MO density without the EDF, which
neither eleesp1 nor libreta includes. See docs/LIBRETA_ESP.md.
"""
import re
import sys

import numpy as np
from gbasis.evals.electrostatic_potential import electrostatic_potential
from gbasis.wrappers import from_iodata
from iodata import load_one

wfx, cpprop, out = sys.argv[1:4]
text = open(wfx).read()
tag, end = "<Additional Electron Density Function (EDF)>", "</Additional Electron Density Function (EDF)>"
if tag in text:
    text = text[: text.index(tag)] + text[text.index(end) + len(end):].lstrip("\n")
    wfx = out + ".noedf.wfx"
    open(wfx, "w").write(text)
mol = load_one(wfx)
basis = from_iodata(mol)
mo = mol.mo
dm = (mo.coeffsa * mo.occsa) @ mo.coeffsa.T + (mo.coeffsb * mo.occsb) @ mo.coeffsb.T

t = open(cpprop, encoding="latin-1").read()
pos = np.array([[float(x) for x in m.split()] for m in re.findall(r"Position \(Bohr\):\s*(.*)", t)])
nb = dm.shape[0]
step = max(1, int(1.2e8 // (nb * nb)))
# a single far-away zero-charge center: zero charges at the real nuclei give 0/0 = NaN at CPs on a nucleus
far, zero = np.array([[1e6, 1e6, 1e6]]), np.zeros(1)
esp_e = np.concatenate([
    electrostatic_potential(basis, dm, pos[i:i + step], far, zero)
    for i in range(0, len(pos), step)])
np.savez(out, pos=pos, esp_e=esp_e, atcorenums=mol.atcorenums, atcoords=mol.atcoords, nbasis=nb)
print(out, len(pos), "points", nb, "basis")
