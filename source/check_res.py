import subprocess
import os
import random
from glob import glob
import pandas as pd

files = glob("*/")
print(len(files))


def main():

    json_tf = True
    if json_tf:

        json_loc = "../data/hydro/"
        json_file = json_loc + "qm_9_hydro_complete.json"
        json_loc = "../data/mg2/"
        json_file = json_loc + "merged_mg.json"
        json_loc = "../data/mg1/"
        json_file = json_loc + "20220613_reaction_data.json"
        
        pandas_file = pd.read_json(json_file)
        product_wfn_count = 0
        reactant_wfn_count = 0
        product_out_count = 0
        reactant_out_count = 0
        
        for ind, _ in pandas_file.iterrows():

            row = pandas_file.iloc[ind]
            reaction_id = row["reaction_id"]

            # create folder in json directory for each reaction
            QTAIM_loc = json_loc + "QTAIM/"
            QTAIM_loc_reactant = json_loc + "QTAIM/" + str(reaction_id) + "/reactants/"
            QTAIM_loc_product = json_loc + "QTAIM/" + str(reaction_id) + "/products/"

            if os.path.exists(QTAIM_loc_reactant + "input.wfn"):
                if os.path.getsize(QTAIM_loc_reactant + "input.wfn") > 0:
                    reactant_wfn_count += 1                
            if os.path.exists(QTAIM_loc_reactant + "CPprop.txt") :
                if os.path.getsize(QTAIM_loc_reactant + "CPprop.txt") > 0:
                    reactant_out_count += 1


            if os.path.exists(QTAIM_loc_product + "input.wfn"):
                if os.path.getsize(QTAIM_loc_product + "input.wfn") > 0:
                    product_wfn_count += 1
            if os.path.exists(QTAIM_loc_product + "CPprop.txt"):
                if os.path.getsize(QTAIM_loc_product + "CPprop.txt") > 0:
                    product_out_count += 1

        print("Product wfn count: {}".format(product_wfn_count))
        print("Reactant wfn count: {}".format(reactant_wfn_count))
        print("Product out count: {}".format(product_out_count))
        print("Reactant out count: {}".format(reactant_out_count))
        print("total rows: {}".format(len(pandas_file)))

    else:
        count = 0
        for ind in range(len(files)):
            if os.path.exists(ind + "/input.wfn"):
                if os.path.exists("./" + str(ind)):
                    if os.path.getsize("./" + str(ind)) <= 0:
                        print(str(ind))
                else:
                    print(str(ind))

                count += 1

        print("{} / {}".format(count, len(files)))


main()
