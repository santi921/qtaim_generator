import json
import os 
import pandas as pd
import numpy as np 


def parse_cp(lines, verbose = True):
    lines_split = [line.split() for line in lines]
    cp_bond, cp_atom = False, False
    cp_name = "null"
    cp_dict = {}

    if('(3,-3)' in lines_split[0]): 
        cp_atom = True
        if verbose: print("atom cp")

    elif('(3,-1)'in lines_split[0]): 
        cp_bond = True
        if verbose: print("bond cp")
    
    else: 
        if verbose: print("ring critical bond not implemented")
        return "ring", cp_dict

    cp_atom_conditionals = {
        "cp_num": ["----------------"],
        "ele_info": ["Corresponding", "nucleus:"],
        "pos_ang": ["Position", "(Angstrom):"],
        "esp_total":["Total", "ESP:"],
    }

    cp_bond_conditionals = {
        "cp_num": ["----------------"],
        "pos_ang": ["Position", "(Angstrom):"],
        "e_loc_func":["Electron", "localization", "function"] ,
        "esp_total":["Total", "ESP:"],
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

def dft_inp_to_dict(dft_inp_file):
    atom_dict = {}

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
    
    return atom_dict

def only_atom_cps(qtaim_descs): 
    ret_dict = {}
    ret_dict_bonds = {}
    for k, v in qtaim_descs.items():
        if "bond" not in k and "Unknown" not in k:
            ret_dict[k] = v
        if "bond" in k:
            ret_dict_bonds[k] = v
    return ret_dict, ret_dict_bonds

def find_cp(atom_dict, atom_cp_dict):
    """
    From a dictionary of atom ind, position, and element, find the corresponding cp in the atom_cp_dict    
    Takes: 
        atom_dict: dict
            dictionary of atom ind, position, and element
        atom_cp_dict: dict
            dictionary of cp ind, position, and element
    Returns:
        cp_key: str
            key of cp_dict
        cp_dict: dict
            dictionary of cp values matching atom 
    """

    for k, v in atom_cp_dict.items():
        distance = np.linalg.norm(np.array(v["pos_ang"]) - np.array(atom_dict["pos"]))
        if(v["cp_num"] == atom_dict["ind"]):
            element_cond_initial = v["element"] == atom_dict["element"]
            distance_cond_initial = distance < 0.5  
            return k, v 
        
        else: 
            dist_cond = distance < 0.5
            element_cond = v["element"] == atom_dict["element"]
            if(dist_cond and element_cond):
                return k, v

    return False, {}
        
def find_cp_map(dft_dict, atom_cp_dict):
    """
    Iterate through dft dict and find nearest cp in atom_cp_dict
    Takes: 
        dft_dict: dict
            dictionary of dft atoms
        atom_cp_dict: dict  
            dictionary of qtaim atoms
    Returns:
        ret_dict: dict
    """
    ret_dict, qtaim_to_dft = {}, {}
    missing_atoms = []
    for k, v in dft_dict.items():
        v_send = {"element": v["element"], "pos": v["pos"], "ind": k}
        ret_key, dict_ret = find_cp(v_send, atom_cp_dict)

        if(ret_key != False):
            ret_dict[k] = dict_ret
            qtaim_to_dft[k] = {"key": ret_key, "pos": dict_ret["pos_ang"]}
        else:
            #print("CP no match found in dft")
            ret_dict[k] = {}
            qtaim_to_dft[k] = {"key": -1, "pos": []}
            missing_atoms.append(k)

    return ret_dict, qtaim_to_dft, missing_atoms
    
def find_bond_cp(i, bonds_cps):
    """
    Takes:
        i: list
            list of two atom indices
        bonds_cps: dict
            dictionary of bond cps
    Returns:
        dict_cp_bond: dict
            dictionary of cp values for bond
    """
    dict_cp_bond = {}
    for k, v in bonds_cps.items():
        if(i == v["atom_inds"] or i == [v["atom_inds"][1], v["atom_inds"][0]]):
            return v
        
    return False

def add_closest_atoms_to_bond(bond_cps, dft_dict):
    """
    Takes in bonds cps and adds the index of the closest atoms to the bond
    Takes:
        bond_cps: dict
            dictionary of bond cps
        dft_to_qtaim: dict
            dictionary of dft to qtaim atom indices
    Returns:
        bond_cps: dict
            dictionary of bond cps with closest atoms added
    """
    for k, v in bond_cps.items():
        for i in k:
            dists = []
            for j in dft_dict.keys():
                dists.append(np.linalg.norm(np.array(v["pos_ang"]) - np.array(dft_dict[j]["pos"])))
            
            bond_cps[k]["atom_inds"] = np.argsort(dists)[:2].tolist()
    return bond_cps

def bond_cp(bond_cps, bond_list, dft_dict):
    ret_dict = {}
    
    bond_cps = add_closest_atoms_to_bond(bond_cps, dft_dict)

    for i in bond_list: 
        dict_cp_bond = find_bond_cp(i, bond_cps)
        
        if dict_cp_bond != False:
            ret_dict[tuple(i)] = dict_cp_bond
        else: 
            #print("No bond found for ", i)
            ret_dict[tuple(i)] = {}

    return ret_dict

def merge_qtaim_inds(qtaim_descs, bond_list, dft_inp_file):
    """
        Gets mapping of qtaim indices to atom indices and remaps atom CP descriptors
        qtaim_descs: dict of qtaim descriptors 
        dft_inp_file: str input file for dft

        returns: dict of qtaim descriptors ordered by atoms in dft_inp_file
    """ 
    # open dft input file 
    dft_dict = dft_inp_to_dict(dft_inp_file)
    # find only atom cps to map
    atom_only_cps, bonds_only_cps = only_atom_cps(qtaim_descs)
    # remap qtaim indices to atom indices
    atom_cps_remapped, qtaim_to_dft, missing_atoms = find_cp_map(dft_dict, atom_only_cps)
    # remapping bonds
    bond_cps = bond_cp(bonds_only_cps, bond_list, dft_dict)
    # merge dictionaries
    ret_dict = {**atom_cps_remapped, **bond_cps}
    return ret_dict

def gather_imputation(df, features_atom, features_bond, root_dir  = "../data/hydro/", json_file_imputed = "./imputed_vals.json"):           
        
    impute_dict = {"atom": {}, "bond": {}}
    for i in features_atom: 
        impute_dict["atom"][i] = []
    for i in features_bond:
        impute_dict["bond"][i] = []

    if os.path.exists(json_file_imputed):
        print("attempting to use previously stored imputation values")
        with open(json_file_imputed, "r") as f:
            impute_dict = json.load(f)
        return impute_dict

    else: 
        for ind, row in df.iterrows():
            try: 
                reaction_id = row["reaction_id"]
                bonds_reactants = row["reactant_bonds"]
                QTAIM_loc_reactant = root_dir + "QTAIM/" + str(reaction_id) + "/reactants/"
                cp_file_reactants = QTAIM_loc_reactant + "CPprop.txt"
                dft_inp_file_reactant = QTAIM_loc_reactant + "input.in"

                qtaim_descs_reactants = get_qtaim_descs(cp_file_reactants, verbose = False)
                
                bonds_products = row["product_bonds"]
                QTAIM_loc_product = root_dir + "QTAIM/" + str(reaction_id) + "/products/"
                cp_file_products = QTAIM_loc_product + "CPprop.txt"
                dft_inp_file_product = QTAIM_loc_product + "input.in"
                qtaim_descs_products = get_qtaim_descs(cp_file_products,  verbose = False)
                
                
                mapped_descs_reactants = merge_qtaim_inds(
                    qtaim_descs_reactants, 
                    bonds_reactants, 
                    dft_inp_file_reactant)
                
                mapped_descs_products = merge_qtaim_inds(
                    qtaim_descs_products, 
                    bonds_products,  
                    dft_inp_file_product)
                
                for k, v in mapped_descs_reactants.items():
                    if(v=={}):
                        pass 
                    elif (type(k) == tuple):
                        for i in features_bond:
                            impute_dict["bond"][i].append(v[i])
                    elif(type(k) == int):
                        for i in features_atom:
                            impute_dict["atom"][i].append(v[i])
                    else: pass 

                for k, v in mapped_descs_products.items():
                    if(v=={}):
                        pass 
                    elif (type(k) == tuple):
                        for i in features_bond:
                            impute_dict["bond"][i].append(v[i])
                    elif(type(k) == int):
                        for i in features_atom:
                            impute_dict["atom"][i].append(v[i])
                    else: pass 
            
            except: print(reaction_id)

    # get the mean and median of each feature
    for k, v in impute_dict.items():
        for k1, v1 in v.items():
            impute_dict[k][k1] = {"mean":np.mean(np.array(v1)), "median": np.median(np.array(v1))}
    # save dictionary as json 
    with open(json_file_imputed, "w") as f:
        json.dump(impute_dict, f)

    return impute_dict
    
def main():
    impute = True

    drop_list = []
    #json_loc = "../data/hydro/"
    #json_file = json_loc + "rev_corrected_bonds_qm_9_hydro_training.json"
    
    json_loc = "../data/hydro3/"
    json_file = json_loc + "qm9_merge_3.json"
    
    json_loc = '../data/holdout/'
    json_file = json_loc + "holdout_test_set_complete_refined.json"

    json_loc = '../data/protonated1/'
    json_file = json_loc + "protonated_reactions_1.json"
    pandas_file = pd.read_json(json_file)
    pandas_out = json_loc + "qtaim_nonimputed.json"
  
    #json_loc = "../data/madeira/"
    #json_file = json_loc + "merged_mg.json"
    #pandas_file = pd.read_json(json_file)
    #pandas_out = json_loc + "mg_qtaim_complete_temp.json"
    json_loc = "../data/rapter/"
    json_file = json_loc + "20230512_mpreact_assoc.bson"

    print("reading file from: {}".format(json_file))
    if json_file.endswith(".json"):
        path_json = json_file
        pandas_file = pd.read_json(path_json)
    else:
        path_bson = json_file
        with open(path_bson,'rb') as f:
            data = bson.decode_all(f.read())
        pandas_file=pd.DataFrame(data)


    
    print(pandas_file.shape)
    
    if impute: 
        imputed_file = json_loc + "impute_vals.json"
        #imputed_file = json_loc + "mg_impute.json"
    #imputed_file = json_loc + "hydro_temp_impute.json"

    bond_list_reactants = []
    bond_list_products = []


    features_atom = ['Lagrangian_K', 'Hamiltonian_K', 'e_density', 'lap_e_density', 
        'e_loc_func', 'ave_loc_ion_E', 'delta_g_promolecular', 'delta_g_hirsh', 'esp_nuc', 
        'esp_e', 'esp_total', 'grad_norm', 'lap_norm', 'eig_hess', 
        'det_hessian', 'ellip_e_dens', 'eta']

    features_bond = ['Lagrangian_K', 'Hamiltonian_K', 'e_density', 'lap_e_density', 
        'e_loc_func', 'ave_loc_ion_E', 'delta_g_promolecular', 'delta_g_hirsh', 'esp_nuc',
        'esp_e', 'esp_total', 'grad_norm', 'lap_norm', 'eig_hess',
        'det_hessian', 'ellip_e_dens', 'eta']

    features_atom = ['esp_total', "ellip_e_dens", "Lagrangian_K", "Hamiltonian_K", "eta"]
    features_bond = ['esp_total', "ellip_e_dens", "Lagrangian_K", "Hamiltonian_K", "eta"]

    if impute:
        impute_dict = gather_imputation(
            pandas_file, 
            features_atom, 
            features_bond, 
            root_dir  = json_loc, 
            json_file_imputed = imputed_file)

    for i in features_atom:
        str_reactant = "extra_feat_atom_reactant_" + i
        str_product = "extra_feat_atom_product_" + i
        pandas_file[str_reactant] = "" 
        pandas_file[str_product] = ""

    for i in features_bond:
        str_reactant = "extra_feat_bond_reactant_" + i
        str_product = "extra_feat_bond_product_" + i
        pandas_file[str_reactant] = "" 
        pandas_file[str_product] = ""
    
    print("Done gathering imputation data...")
    fail_count = 0
    for ind, row in pandas_file.iterrows():
        
        try: 
            reaction_id = row["reaction_id"]
            bonds_reactants = row["reactant_bonds"]
            QTAIM_loc_reactant = json_loc + "QTAIM/" + str(reaction_id) + "/reactants/"
            cp_file_reactants = QTAIM_loc_reactant + "CPprop.txt"
            dft_inp_file_reactant = QTAIM_loc_reactant + "input.in"
            
            bonds_products = row["product_bonds"]
            QTAIM_loc_product = json_loc + "QTAIM/" + str(reaction_id) + "/products/"
            cp_file_products = QTAIM_loc_product + "CPprop.txt"
            dft_inp_file_product = QTAIM_loc_product + "input.in"
            
            qtaim_descs_reactants = get_qtaim_descs(cp_file_reactants, verbose = False)
            qtaim_descs_products = get_qtaim_descs(cp_file_products,  verbose = False)
            
            mapped_descs_reactants = merge_qtaim_inds(
                qtaim_descs_reactants, 
                bonds_reactants, 
                dft_inp_file_reactant)
            
            mapped_descs_products = merge_qtaim_inds(
                qtaim_descs_products, 
                bonds_products,  
                dft_inp_file_product)

            
            bonds_products, bonds_reactants = [], []
            
            #print(mapped_descs_products)
            # fill in imputation values
            for k, v in mapped_descs_reactants.items():
                if(v=={}):
                    #print(mapped_descs_reactants)
                    if (type(k) == tuple):
                        bonds_reactants.append(list(k))
                        for i in features_bond:
                            if impute:
                                mapped_descs_reactants[k][i] = impute_dict["bond"][i]["median"]
                            else: 
                                #print("feature missing, {}, {}".format(ind,i))
                                mapped_descs_reactants[k][i] =  -1
                                if(ind not in drop_list):
                                    drop_list.append(ind)
                                                            
                    else: 
                        for i in features_atom:
                            if impute:
                                mapped_descs_reactants[k][i] = impute_dict["atom"][i]["median"]   
                            else:   
                                #print("feature missing, {}, {}".format(ind,i))
                                mapped_descs_reactants[k][i] = -1
                                if(ind not in drop_list):
                                    drop_list.append(ind)
            
            
            for k, v in mapped_descs_products.items():
                if(v=={}):
                    if (type(k) == tuple): # bond cp 
                        for i in features_bond:
                            bonds_products.append(list(k))
                            if impute:
                                mapped_descs_products[k][i] = impute_dict["bond"][i]["median"]

                            else: 
                                
                                mapped_descs_products[k][i] =  -1
                                if(ind not in drop_list):
                                    drop_list.append(ind)
                    else: # atom cp
                        for i in features_atom:
                            if impute:
                                mapped_descs_products[k][i] = impute_dict["atom"][i]["median"]  

                            else: 
                                
                                mapped_descs_products[k][i] = -1
                                if(ind not in drop_list):
                                    drop_list.append(ind)
        

            # get all the values of a certain key for every dictionary in the dicitonary
            cps_reactants = mapped_descs_reactants.keys()
            cps_products = mapped_descs_products.keys()
            flat_reactants, flat_products = {}, {}

            
            for cps_reactant in cps_reactants:
                for i in features_atom: 
                    if (type(cps_reactant) != tuple):
                        # check if the key exists
                        name = "extra_feat_atom_reactant_" + i
                        if name not in flat_reactants.keys():
                            flat_reactants[name] = []    
                        flat_reactants[name].append(mapped_descs_reactants[cps_reactant][i])

                for i in features_bond:
                    if (type(cps_reactant) == tuple): 
                        # check if the key exists
                        name = "extra_feat_bond_reactant_" + i
                        if name not in flat_reactants.keys():
                            flat_reactants[name] = []    
                        flat_reactants[name].append(mapped_descs_reactants[cps_reactant][i])

            for cps_product in cps_products:
                for i in features_atom: 
                    if (type(cps_product) != tuple):
                        # check if the key exists
                        name = "extra_feat_atom_product_" + i
                        if name not in flat_products.keys():
                            flat_products[name] = []    
                        flat_products[name].append(mapped_descs_products[cps_product][i])

                for i in features_bond:
                    if (type(cps_product) == tuple): 
                        # check if the key exists
                        name = "extra_feat_bond_product_" + i
                        if name not in flat_products.keys():
                            flat_products[name] = []    
                        flat_products[name].append(mapped_descs_products[cps_product][i])


            # update the pandas file with the new values
            for k, v in flat_reactants.items():
                if ("bond" in k): 
                    pandas_file.at[ind, k] = [v]
                else: 
                    pandas_file.at[ind, k] = np.array(v)

            for k, v in flat_products.items():
                if ("bond" in k): 
                    pandas_file.at[ind, k] = [v]
                else: 
                    pandas_file.at[ind, k] = np.array(v)


            keys_products = mapped_descs_products.keys()
            keys_reactants = mapped_descs_reactants.keys()
            # filter keys that aren't tuples 
            keys_products = [x for x in keys_products if type(x) == tuple]
            keys_reactants = [x for x in keys_reactants if type(x) == tuple]
            bond_list_reactants.append([keys_reactants])
            bond_list_products.append([keys_products])


        except: 
            drop_list.append(ind)
            bond_list_reactants.append([])
            bond_list_products.append([])
            # iterate over all the features and set them to -1
            for i in features_atom:
                name = "extra_feat_atom_reactant_" + i
                pandas_file.at[ind, name] = -1
                name = "extra_feat_atom_product_" + i
                pandas_file.at[ind, name] = -1
            # iterate over all the features and set them to -1
            for i in features_bond:
                name = "extra_feat_bond_reactant_" + i
                pandas_file.at[ind, name] = -1
                name = "extra_feat_bond_product_" + i
                pandas_file.at[ind, name] = -1

            print(reaction_id)            
            fail_count += 1

    print(fail_count/ind)            
    pandas_file["extra_feat_bond_reactant_indices_qtaim"] = bond_list_reactants
    pandas_file["extra_feat_bond_product_indices_qtaim"] = bond_list_products

    # if impute false then drop the rows that have -1 values
    if not impute:
        pandas_file.drop(drop_list, inplace=True)
        pandas_file.reset_index(drop=True, inplace=True)
    print("length of drop list: {}".format(len(drop_list)))
    print("done gathering and imputing features...")
    # save the pandas file
    print(pandas_file.shape)
    pandas_file.to_json(pandas_out)

main()
