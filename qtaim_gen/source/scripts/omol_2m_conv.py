

from fairchem.core.datasets import AseDBDataset
from rdkit.Chem import rdDetermineBonds
from rdkit import Chem
import lmdb
import pickle 
from tqdm import tqdm

from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph


num_to_element = {
    1: "H",  2: "He",  3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",   8: "O",   9: "F",  10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",  16: "S",  17: "Cl", 18: "Ar", 19: "K",  20: "Ca",
    21: "Sc", 22: "Ti", 23: "V",  24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y",  40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I",  54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W",  75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac"
}

def get_molecule_from_xyz(xyz_str, spin, charge):
    rdkit_molecule = Chem.rdmolfiles.MolFromXYZBlock(xyz_str)
    rdDetermineBonds.DetermineConnectivity(rdkit_molecule)
    molecule = Molecule.from_str(xyz_str, fmt="xyz")
    molecule.set_charge_and_spin(
        charge=int(charge), spin_multiplicity=int(spin)
    )
    molecule_graph = MoleculeGraph.with_empty_graph(molecule)
    return molecule, molecule_graph, rdkit_molecule

def get_string_from_pos_atom(pos, numbers, spin, charge):
    n_atoms = len(numbers)
    xyz_str = "{}\n".format(n_atoms)
    spin_charge_line = "{} {}\n".format(charge, spin)
    xyz_str += spin_charge_line
    # write the atom positions
    for ind, atom in enumerate(pos):

        atom_line = "{} {:.3f}\t {:.3f}\t {:.3f}\n".format(
            num_to_element[int(numbers[ind])], atom[0], atom[1], atom[2]
        )
        xyz_str += atom_line
    return xyz_str


def main():

    dry = False

    dataset = AseDBDataset(
        {
            "src": "/home/santiagovargas/dev/EScAIP/dev/data/omol_subset/train/"
        }
    )
    length = len(dataset)

    if not dry: 
        db_charge = lmdb.open(
            "./omol_lmdbs/charge_omol_2m_1.lmdb",
            map_size=int(1099511627776 * 2),
            subdir=False,
            meminit=False,
            map_async=True,
        )


        db_geom = lmdb.open(
            "./omol_lmdbs/geom_omol_2m_1.lmdb",
            map_size=int(1099511627776 * 2),
            subdir=False,
            meminit=False,
            map_async=True,
        )

    
    # get first 1/4 of data 


    for i in tqdm(range(500000)):
        atoms = dataset.get_atoms(i)
        # used to generate geometries 
        pos = dataset[i].pos
        numbers = dataset[i].atomic_numbers
        spin = atoms.info['spin']
        charge = atoms.info['charge']
        
        lowdin_charges, mulliken_charges = [], []

        if "lowdin_charges" in atoms.info.keys(): 
            lowdin_charges = atoms.info['lowdin_charges']
        if "mulliken_charges" in atoms.info.keys(): 
            mulliken_charges = atoms.info['mulliken_charges']

        # convert to xyz string
        xyz_str = get_string_from_pos_atom(pos, numbers, spin, charge)
        
        # convert to rdkit molecule
        molecule, molecule_graph, rdkit_molecule = get_molecule_from_xyz(xyz_str, spin, charge)

        
        bonds = rdkit_molecule.GetBonds()
        bond_list = []
        # Iterate over the bonds and print their information
        for bond in bonds:
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            molecule_graph.add_edge(atom1_idx, atom2_idx) 
            bond_list.append([atom1_idx, atom2_idx])
        
        charge_dict = {
            "lowdin_charges": lowdin_charges,
            "mulliken_charges": mulliken_charges
        }

        data_dict = {
                "molecule": molecule,
                "molecule_graph": molecule_graph,
                "ids": i,
                "bonds": bond_list,
                "spin": int(spin),
                "charge": int(charge),
        }
        

        if not dry: 
            sample_index = i
            txn = db_charge.begin(write=True)
            txn.put(
                f"{sample_index}".encode("ascii"),
                pickle.dumps(charge_dict, protocol=-1),
            )
            txn.commit()

            txn = db_geom.begin(write=True)
            txn.put(
                f"{sample_index}".encode("ascii"),
                pickle.dumps(data_dict, protocol=-1),
            )
            txn.commit()        

    
    if not dry: 
        txn = db_charge.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(length, protocol=-1))
        txn.commit()

        txn = db_geom.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(length, protocol=-1))
        txn.commit()


main()