import numpy as np
import json
#from pymatgen.command_line.critic2_caller import Critic2Caller

def parse_critic2(cro, features={"atom": [], "bond": []}):
    """
    Parses a critic2 output file and returns a dictionary of the critical points
    Takes: 
        cro: critic2 output file
        features: dictionary of features to parse
    """


    cp_loaded = json.load(open(cro, "r"))    
    bohr_to_ang = 0.529177249
    translation = {
        'field' : 'field',
         'gradient_norm' : 'grad_norm',
         'laplacian': "lap_norm",
         'hessian_eigenvalues': 'eig_hess',
         'ellipticity': "ellip_e_dens"
    }
    atoms = []
    bonds = []
    species = {}
    processed_dict = {}
    atom_dict = {} # separately parses atoms
    bond_dict = {} # separately parses bonds
    
    if (
        cp_loaded["critical_points"]["number_of_nonequivalent_cps"]
        != cp_loaded["critical_points"]["number_of_cell_cps"]
    ):
        raise ValueError(
            "ERROR: number_of_nonequivalent_cps should always equal number_of_cell_cps!"
        )

    
    for specie in cp_loaded["structure"]["species"]:
        if specie["name"][-1] == "_":
            species[specie["id"]] = specie["name"][:-1]
        else:
            species[specie["id"]] = specie["name"]

    centering_vector = cp_loaded["structure"]["molecule_centering_vector"]

    for ii, atom in enumerate(cp_loaded["structure"]["nonequivalent_atoms"]):
        specie = species[atom["species"]]
        atoms.append(specie)
        tmp = atom["cartesian_coordinates"]
        coords = []
        for jj, val in enumerate(tmp):
            coords.append((val + centering_vector[jj]) * bohr_to_ang)

    for cp in cp_loaded["critical_points"]["nonequivalent_cps"]:
        if cp["rank"] == 3 and cp["signature"] == -1:
            bond_dict[str(cp["id"]) + "_bond"] = {"field": cp["field"]}

        if cp["rank"] == 3 and cp["signature"] == -3:
            atom_dict[str(cp["id"]) + "_" + cp["name"][:-1]] = {"field": cp["field"]}

    if features["bond"] != [] or features["atom"] != []:
        cp_features = cp_loaded["critical_points"]["nonequivalent_cps"]
    
    atom_dict_raw = [i.split("_")[0] for i in list(atom_dict.keys())]
    
    for cp_ind, cp in enumerate(cp_loaded["critical_points"]["cell_cps"]): # finds attractors to BCPs
        if str(cp["id"]) in atom_dict_raw:
            # get the index of the id in atom_dict_raw
            atom_dict_index = atom_dict_raw.index(str(cp["id"]))
            if features["atom"] != []:
                for feature in features["atom"]:
                    if feature in cp_features[cp_ind]:
                        # get the atom_dict_index-th key in atom_dict
                        name_atom = list(atom_dict.keys())[atom_dict_index]

                        if feature == "ellipticity":
                            atom_dict[name_atom]["extra_feat_atom_" + translation[feature]] = np.abs(cp_features[cp_ind]["hessian_eigenvalues"][0]) / np.abs(cp_features[cp_ind]["hessian_eigenvalues"][1]) - 1
                        elif feature == "hessian_eigenvalues":
                            atom_dict[name_atom]["extra_feat_atom_" + translation[feature]] = cp_features[cp_ind]["hessian_eigenvalues"][0]           
                        else:
                            atom_dict[name_atom][
                                "extra_feat_atom_" + translation[feature]
                            ] = cp_features[cp_ind][feature]

        if str(cp["id"]) + "_bond"  in bond_dict:
            # Check if any bonds include fictitious atoms
            bad_bond = False
            for entry in cp["attractors"]:
                if int(entry["cell_id"]) - 1 >= len(atoms):
                    bad_bond = True
            # If so, remove them from the bond_dict
            if bad_bond:
                bond_dict.pop(str(cp["id"]) + "_bond")

            else:
                bond_dict[str(cp["id"]) + "_bond"]["atom_ids"] = [
                    entry["cell_id"] for entry in cp["attractors"]
                ]
                atoms_raw = [
                    atoms[int(entry["cell_id"]) - 1] for entry in cp["attractors"]
                ]
                # get argsort of atoms_raw
                argsort = np.argsort(bond_dict[str(cp["id"]) + "_bond"]["atom_ids"])
                #print(bond_dict[cp["id"]]["atom_ids"], argsort)
                # sort atoms_raw
                atoms_raw = [atoms_raw[i] for i in argsort]
                # sort atom_ids
                bond_dict[str(cp["id"]) + "_bond"]["atom_ids"] = [
                    int(bond_dict[str(cp["id"]) + "_bond"]["atom_ids"][i]) - 1 for i in argsort
                ]
                
                bond_dict[str(cp["id"]) + "_bond"]["atoms"] = atoms_raw
                bond_dict[str(cp["id"]) + "_bond"]["distance"] = (
                    cp["attractors"][0]["distance"] * bohr_to_ang
                    + cp["attractors"][1]["distance"] * bohr_to_ang
                )

                if features["bond"] != []:
                    for feature in features["bond"]:
                        if feature in cp_features[cp_ind]:
                            if feature == "ellipticity":
                                bond_dict[str(cp["id"]) + "_bond"]["extra_feat_bond_" + translation[feature]] = np.abs(cp_features[cp_ind]["hessian_eigenvalues"][0]) / np.abs(cp_features[cp_ind]["hessian_eigenvalues"][1]) - 1
                            elif feature == "hessian_eigenvalues":
                                bond_dict[str(cp["id"]) + "_bond"]["extra_feat_bond_" + translation[feature]] = cp_features[cp_ind]["hessian_eigenvalues"][0]           
                            else:
                                bond_dict[str(cp["id"]) + "_bond"]["extra_feat_bond_" + translation[feature]] = cp_features[cp_ind][feature]
                        
    for cpid in bond_dict:
        bonds.append([int(entry) for entry in bond_dict[cpid]["atom_ids"]])

    processed_dict["bonds"] = bonds # bonds
    processed_dict["bond_cps"] = bond_dict # bond info
    processed_dict["atom_cps"] = atom_dict # atom info
    return processed_dict