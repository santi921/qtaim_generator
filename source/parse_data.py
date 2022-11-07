import pandas as pd

#def parse_cp():
    

def get_qtaim_descs(file = "CPprop_1157_1118_1158.txt"):
    """
    file: str
        path to file
    """
    # read file
    ret_dict = {}

    with open(file) as f:
        lines = f.readlines()
    
    
    # get the lines with the descriptors
    #lines = [line for line in lines if "QTAIM" in line]
    # get the descriptors
    #lines = [line.split() for line in lines]
    # get the values
    #lines = [line[1] for line in lines]
    # convert to floats
    #lines = [float(line) for line in lines]
    return ret_dict



def find_atom_critical_points(pmg_molecule, ):
    pass

def find_bond_critical_points(bonds):
    pass

def main():
    
    json_loc = "../data/mg2/"
    json_file = json_loc + "20220613_reaction_data.json"
    pandas_file = pd.read_json(json_file)

    for ind, row in pandas_file.iterrows():
        
        reaction_id = row["reaction_id"]
        # reactants
        try: reactants = row["combined_reactants_graph"]
        except: reactants = row["reactant_molecule_graph"]
        # products
        try: products = row["combined_products_graph"]
        except: products = row["product_molecule_graph"]


        product_bonds = row["combined_product_bonds_global"]
        reactant_bonds = row["reactant_bonds"]


