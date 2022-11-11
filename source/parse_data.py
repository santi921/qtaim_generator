import pandas as pd
import numpy as np 
#def parse_cp():

def parse_cp(lines, verbose = True):
    lines_split = [line.split() for line in lines]
    cp_bond, cp_atom = False, False
    cp_name = "null"
    cp_dict = {}
    #print(lines_split[0])
    if(lines_split[0][4] == '(3,-3)'): 
        cp_atom = True
        if verbose: print("atom cp")
    elif(lines_split[0][4] == '(3,-1)'): 
        cp_bond = True
        if verbose: print("bond cp")
    else: 
        if verbose: print("ring critical bond not implemented")
        return "ring", cp_dict

    cp_atom_conditionals = {
        "cp_num": ["----------------"],
        "ele_info": ["Corresponding", "nucleus:"],
        "pos_ang": ["Position", "(Angstrom):"],
        "Lagrangian_K": ["Lagrangian", "kinetic", "energy"],
        "Hamiltonian_K": ["Hamiltonian", "kinetic", "energy"],
        "e_density":["Energy", "density", "E(r)"],
        "lap_e_density":["Laplacian", "electron", "density:"] ,
        "e_loc_func":["Electron", "localization", "function"] ,
        "ave_loc_ion_E":["Average", "local", "ionization", "energy"],
        "delta_g_promolecular":["Delta-g", "promolecular"],
        "delta_g_hirsh":["Delta-g", "Hirshfeld"],
        "esp_nuc":["ESP" , "nuclear", "charges:"],
        "esp_e":["ESP", "electrons:"],
        "esp_total":["Total", "ESP:"],
        "grad_norm":["Components", "gradient", "x/y/z"],
        "lap_norm":["Components", "Laplacian", "x/y/z"],
        "eig_hess":["Eigenvalues", "Hessian:"],
        "det_hessian":["Determinant", "Hessian:"],
        "ellip_e_dens":["Ellipticity", "electron", "density:"],
        "eta":["eta", "index:"]
    }

    cp_bond_conditionals = {
        "cp_num": ["----------------"],
        "pos_ang": ["Position", "(Angstrom):"],
        "Lagrangian_K": ["Lagrangian", "kinetic", "energy"],
        "Hamiltonian_K": ["Hamiltonian", "kinetic", "energy"],
        "e_density":["Energy", "density", "E(r)"],
        "lap_e_density":["Laplacian", "electron", "density:"] ,
        "e_loc_func":["Electron", "localization", "function"] ,
        "ave_loc_ion_E":["Average", "local", "ionization", "energy"],
        "delta_g_promolecular":["Delta-g", "promolecular"],
        "delta_g_hirsh":["Delta-g", "Hirshfeld"],
        "esp_nuc":["ESP" , "nuclear", "charges:"],
        "esp_e":["ESP", "electrons:"],
        "esp_total":["Total", "ESP:"],
        "grad_norm":["Components", "gradient", "x/y/z"],
        "lap_norm":["Components", "Laplacian", "x/y/z"],
        "eig_hess":["Eigenvalues", "Hessian:"],
        "det_hessian":["Determinant", "Hessian:"],
        "ellip_e_dens":["Ellipticity", "electron", "density:"],
        "eta":["eta", "index:"]
    }

    if cp_atom:
        unknown_id = 0
        for ind, i in enumerate(lines_split):
            for k, v in cp_atom_conditionals.items():
                if all(x in i for x in v):
                    if(k == "cp_num"):                        
                        cp_dict[k] = int(i[2][:-1])

                    elif(k == 'pos_ang'): 
                        
                        cp_dict[k] = [float(x) for x in i[2:]]

                    elif(k == 'ele_info'):
                        if(i[2] == "Unknown"):
                            cp_name = str(unknown_id) + "_Unknown"
                            cp_dict["number"] = "Unknown"
                            cp_dict["ele"] = "Unknown"
                            unknown_id += 1
                        else: 
                            cp_dict["element"] = i[2].split("(")[1]
                            cp_dict["number"] = i[2].split("(")[0]
                            cp_name = cp_dict["number"] + "_" + cp_dict["element"]

                    elif(k == 'esp_total'):
                        cp_dict[k] = float(i[2])

                    elif(k == 'eig_hess'):
                        cp_dict[k] = np.sum(np.array([float(x) for x in i[-3:]]))

                    elif(k == 'grad_norm' or k == 'lap_norm'):
                        cp_dict[k] = float(lines_split[ind+2][-1])

                    else: 
                        #print(i)
                        cp_dict[k] = float(i[-1])
                    
                    cp_atom_conditionals.pop(k)
                    break

    elif cp_bond: 
        for ind, i in enumerate(lines_split):
            for k, v in cp_bond_conditionals.items():
                if all(x in i for x in v):
                    if(k == "cp_num"):                        
                        cp_dict[k] = int(i[2][:-1])
                        cp_name = str(cp_dict[k]) + "_bond"
                    elif(k == 'pos_ang'): 
                        cp_dict[k] = [float(x) for x in i[2:]]
                    elif(k == 'esp_total'):
                        cp_dict[k] = float(i[2])
                    elif(k == 'eig_hess'):
                        cp_dict[k] = np.sum(np.array([float(x) for x in i[-3:]]))
                    elif(k == 'grad_norm' or k == 'lap_norm'):
                        cp_dict[k] = float(lines_split[ind+2][-1])
                        
                    else: 
                        cp_dict[k] = float(i[-1])
                    #print(v)
                    cp_bond_conditionals.pop(k)

                    break
    
    else: print("error")

    return cp_name, cp_dict

def get_qtaim_descs(file = "./CPprop_1157_1118_1158.txt", verbose = False):
    """
    file: str
        path to file
    """
    # read file
    cp_dict, ret_dict = {}, {}

    with open(file) as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
    
    # section lines into segments on ----------------
    track = 0 
    for ind, line in enumerate(lines): 
        if "----------------" in line:
            lines_segment = []
        lines_segment.append(line)
        if(ind < len(lines)-1):
            if "----------------" in lines[ind+1]:
                cp_dict[track] = lines_segment
                track += 1
        else: 
            cp_dict[track] = lines_segment

    for k, v in cp_dict.items():
        ind_atom, cp_dict = parse_cp(v, verbose = verbose)
        ret_dict[ind_atom] = cp_dict

    # remove keys-value pairs that are "ring"
    ret_dict = {k: v for k, v in ret_dict.items() if "ring" not in k}
    return ret_dict

def add_ind_to_dict(dict, lines_dft_file): 
    """
    dict: dict
        dictionary of cp descriptors
    lines_dft_file: list
        list of lines from dft input file
    """
    # TODO
    return dict

def map_feats_to_bonds(df, qtaim_descs):
    # takes xyz positions of bond CP and finds nearest neighbors in dict
    
    list_of_bond_inds = []
    list_of_list_of_features = []
    for k, v in qtaim_descs.items():
        if "bond" in k:
            pass
            #list_of_bond_inds.append(k)
            #list_of_list_of_features.append(v)

    return list_of_bond_inds, list_of_list_of_features

def check_num_atoms(qtaim_descs, n_atoms):
    nuclear_cp_count = 0 
    for k, v in qtaim_descs.items():
        #print(k)
        if(k.split("_")[1] != 'bond' and k.split("_")[1] != 'Unknown'):
            nuclear_cp_count += 1
            
    print(nuclear_cp_count, n_atoms)
    if(n_atoms != nuclear_cp_count):
        return False
    return True

def count_atoms(dict_comp):
    count = 0 
    for k, v in dict_comp.items():
        count+=v
    return int(count)

def check_qtaim_mapping(map_dict, atom_dict, qtaim_descs):
    for k, v in map_dict.items():
        qtaim_descs_key = list(qtaim_descs.keys())[k]
        print(list(atom_dict.keys()))
        atom_dict_key = list(atom_dict.keys())[v]
        element_qtaim = qtaim_descs[qtaim_descs_key]["element"]
        element_atom = atom_dict[atom_dict_key]["element"]
        xyz_qtaim = qtaim_descs[qtaim_descs_key]["pos_ang"]
        xyz_atom = atom_dict[atom_dict_key]["pos"]
        
        if(np.linalg.norm(np.array(xyz_qtaim) - np.array(xyz_atom)) > 0.1):
            print("error in mapping")
            return False
        if (element_atom != element_qtaim):
            print("error in mapping")
            return False
    
    return True

def merge_qtaim_inds(qtaim_descs, dft_inp_file):
    
    ret_dict = {}   
    atom_dict = {}
    # open dft input file 
    with open(dft_inp_file) as f:
        lines = f.readlines()
        # strip tabs
        lines = [line[:-1] for line in lines]
        
    # find line starting with "* xyz"
    for ind, line in enumerate(lines):
        if "* xyz" in line:
            xyz_ind = ind
            break

    # filter lines before and including xyz_ind
    lines = lines[xyz_ind+1:-1]  
    
    
    for ind, line in enumerate(lines):
        line_split = line.split()
        atom_dict[ind] = {"element": line_split[0], "pos": [float(x) for x in line_split[1:]]} 
    
    for k, v in qtaim_descs.items():
        if "bond" not in k and "Unknown" not in k:
            ret_dict[int(v["cp_num"])] = int(v["number"])
            ind += 1

    valid_atoms = check_qtaim_mapping(ret_dict, atom_dict, qtaim_descs)
    
    if(valid_atoms): 
        print("valid mapping")
        features_atom_remapped = {}
        keys_qtaim = list(qtaim_descs.keys())
        for k, v in ret_dict.items():
            key_kth = keys_qtaim[k]
            features_atom_remapped[v] = qtaim_descs[key_kth]

        return ret_dict, features_atom_remapped
    else:
        print("invalid mapping")
        return ret_dict, {}

def main():
    
    json_loc = "../data/hydro/"
    #json_file = json_loc + "20220613_reaction_data.json"
    #json_loc_mg = "/home/santiagovargas/dev/bondnet/bondnet/dataset/mg_dataset/"
    #json_file = json_loc + "merged_mg.json"
    json_file = json_loc + "rev_corrected_bonds_qm_9_hydro_training.json"
    
    pandas_file = pd.read_json(json_file)
    QTAIM_loc = json_loc + "QTAIM/"
    
    for ind, row in pandas_file.head(10).iterrows():

        reaction_id = row["reaction_id"]
        QTAIM_loc_reactant = json_loc + "QTAIM/" + str(reaction_id) + "/reactants/"
        QTAIM_loc_product = json_loc + "QTAIM/" + str(reaction_id) + "/products/"
        print(reaction_id)
        qtaim_descs_reactant = get_qtaim_descs(QTAIM_loc_reactant + "CPprop.txt", verbose = False)
        qtaim_descs_product = get_qtaim_descs(QTAIM_loc_product + "CPprop.txt", verbose = False)
        # remove CPs 
        natoms = count_atoms(row["composition"])
        react_cp_found = check_num_atoms(qtaim_descs_reactant, natoms)
        prod_cp_found = check_num_atoms(qtaim_descs_product, natoms)

        features_in_order_reactants = merge_qtaim_inds(qtaim_descs_reactant, dft_inp_file = QTAIM_loc_product + "input.in")
        features_in_order_products = merge_qtaim_inds(qtaim_descs_product, dft_inp_file = QTAIM_loc_product + "input.in")

        #qtaim_descs_reactant = bond_feat_add_bond_inds(qtaim_descs_reactant, atom_maps_reactant)
        #qtaim_descs_product = bond_feat_add_bond_inds(qtaim_descs_product, atom_maps_product)
        print(react_cp_found, prod_cp_found)
        # bond cp xyz to atom indicies

        #reactant_feats_list, product_feats_list = pull_feats_from_qtaim(qtaim_descs_reactant, qtaim_descs_product)


        # reactants
        #try: reactants = row["combined_reactants_graph"]
        #except: reactants = row["reactant_molecule_graph"]
        # products
        #try: products = row["combined_products_graph"]
        #except: products = row["product_molecule_graph"]
        #product_bonds = row["combined_product_bonds_global"]
        #reactant_bonds = row["reactant_bonds"]


main()