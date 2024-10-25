import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds



def get_bonds_from_rdkit(xyz):
    # Create an RDKit molecule from the XYZ string
    rdkit_molecule = Chem.rdmolfiles.MolFromXYZFile(xyz)
    # Retrieve the bonds in the RDKit molecule
    # conn_mol = Chem.Mol(rdkit_molecule)
    rdDetermineBonds.DetermineConnectivity(rdkit_molecule)
    bonds = rdkit_molecule.GetBonds()
    bond_list = []
    # Iterate over the bonds and print their information
    for bond in bonds:
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1 = rdkit_molecule.GetAtomWithIdx(atom1_idx)
        atom2 = rdkit_molecule.GetAtomWithIdx(atom2_idx)
        bond_type = bond.GetBondType()
        # print(
        #    f"Bond between atom {atom1.GetSymbol()}({atom1.GetIdx()}) and atom {atom2.GetSymbol()}({atom2.GetIdx()}): {bond_type}"
        # )
        bond_list.append([atom1_idx, atom2_idx])
    return bond_list


def connectedMatrix(struct_pmg, bond_length_dict):
    cM = {}  # connected Matrix
    for ind1, site1 in enumerate(struct_pmg):
        # print(ind1, site1.specie)
        nameStr = str(site1.specie) + str(ind1)
        cM[nameStr] = []
        for ind2, site2 in enumerate(struct_pmg):
            dict_key = str(site1.specie) + "," + str(site2.specie)
            dict_key_rev = str(site2.specie) + "," + str(site1.specie)

            if dict_key in bond_length_dict.keys():
                # print(dict_key)
                pass
            elif dict_key_rev in bond_length_dict.keys():
                # print(dict_key_rev)
                dict_key = dict_key_rev

            if (
                ind1 != ind2
                and np.linalg.norm(site1.coords - site2.coords)
                <= bond_length_dict[dict_key]
            ):
                cM[nameStr].append(ind2)

    # convert to list of lists of format [[origin, connected1], [origin, connected2]...]
    list_of_lists = []
    # print(cM)
    for key in cM.keys():
        for connected in cM[key]:
            # connected_ind = "".join([i for i in connected if i.isdigit()])
            connected_ind = int("".join([i for i in key if i.isdigit()]))
            list_of_lists.append([connected_ind, connected])

    # print(list_of_lists)
    return list_of_lists
