import pandas as pd 
import numpy as np 
from glob import glob 
import subprocess
import os 
import random

def main():
    json_tf = True
    if(json_tf): 
        json_loc = "../data/hydro/"
        json_file = json_loc + "rev_corrected_bonds_qm_9_hydro_training.json"
        pandas_file = pd.read_json(json_file)

        for ind, _ in pandas_file.iterrows():
            
            ind_random = random.choice(range(len(pandas_file)))
            
            row = pandas_file.iloc[ind_random]
            reaction_id = row["reaction_id"]
            # create folder in json directory for each reaction 
            QTAIM_loc = json_loc + "QTAIM/"
            QTAIM_loc_reactant = json_loc + "QTAIM/" + str(reaction_id) + "/reactants/"
            QTAIM_loc_product = json_loc + "QTAIM/" + str(reaction_id) + "/products/"
            # create folder for each reaction + reactants + products
            
            if not os.path.exists(QTAIM_loc):
                os.mkdir(QTAIM_loc)
            if not os.path.exists(json_loc + "QTAIM/" + str(reaction_id)):
                os.mkdir(json_loc + "QTAIM/" + str(reaction_id))
            if not os.path.exists(QTAIM_loc_reactant):
                os.mkdir(QTAIM_loc_reactant)
            if not os.path.exists(QTAIM_loc_product):
                os.mkdir(QTAIM_loc_product)


            # reactants
            reactants = row["combined_reactants_graph"]
            atoms = int(len(reactants["molecule"]["sites"]))
            with open(QTAIM_loc_reactant + "/input.in", 'w') as f:
                f.write("!B3LYP def2-SVP AIM\n")
                f.write("%SCF\n")
                f.write("    MaxIter 1000\n")
                f.write("END\n")
                f.write("* xyz {} {}\n".format(reactants["molecule"]["charge"], reactants["molecule"]["spin_multiplicity"]))
                for ind in range(atoms): 
                    xyz = reactants["molecule"]["sites"][ind]["xyz"]
                    atom = reactants["molecule"]["sites"][ind]["species"][0]["element"]
                    f.write("{}\t{: .4f}\t{: .4f}\t{: .4f}\n".format(atom, xyz[0], xyz[1], xyz[2]))
                f.write("*\n") 


            # products
            products = row["combined_products_graph"]
            atoms = int(len(products["molecule"]["sites"]))
            with open(QTAIM_loc_product + "/input.in", 'w') as f:
                f.write("!B3LYP def2-SVP AIM\n")
                f.write("%SCF\n")
                f.write("    MaxIter 1000\n")
                f.write("END\n")
                f.write("* xyz {} {}\n".format(products["molecule"]["charge"], products["molecule"]["spin_multiplicity"]))
                for ind in range(atoms): 
                    xyz = products["molecule"]["sites"][ind]["xyz"]
                    atom = products["molecule"]["sites"][ind]["species"][0]["element"]
                    f.write("{}\t{: .4f}\t{: .4f}\t{: .4f}\n".format(atom, xyz[0], xyz[1], xyz[2]))
                f.write("*\n") 


            # run QTAIM on reactants
            with open(QTAIM_loc_reactant + '/props.sh', 'w') as f: 
                f.write("#!/bin/bash\n")
                f.write("./Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn " + QTAIM_loc_reactant + "/input.wfn < ./Multiwfn_3.8_dev_bin_Linux_noGUI/data.txt | tee ./" + QTAIM_loc_reactant + "out \n")

            # run QTAIM on products
            with open(QTAIM_loc_product + '/props.sh', 'w') as f: 
                f.write("#!/bin/bash\n")
                f.write("./Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn " + QTAIM_loc_product + "/input.wfn < ./Multiwfn_3.8_dev_bin_Linux_noGUI/data.txt | tee ./" + QTAIM_loc_product + "out \n")

    else:   
        dir_source = "./"
        files = glob(dir_source+"*xyz") # xyz file names, would need to change for pandas
        for i in files:
            folder = i.split('_')[1].split('.')[0]
            try: 
                os.mkdir(folder)
            except: 
                pass

            with open(i) as xyz:
                lines = xyz.readlines()

            atoms = int(lines[0])

            with open(folder + "/input.in", 'w') as f:
                f.write("!B3LYP def2-SVP AIM\n\n")
                f.write("*xyz 0 1\n")
                for ind in range(atoms): 
                    f.write(str(lines[ind+2].split()[0]) + "\t" + str(lines[ind+2].split()[1]) + "\t" + str(lines[ind+2].split()[1]) + "\t" + str(lines[ind+2].split()[2]) + "\n")        
                f.write("*\n")

            with open(folder + '/props.sh', 'w') as f: 
                #print("#!/bin/bash\n")
                #print("../Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn input.wfn < ../Multiwfn_3.8_dev_bin_Linux_noGUI/data.txt\n")
                f.write("#!/bin/bash\n")
                f.write("./Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn " + folder + "/input.wfn < ./Multiwfn_3.8_dev_bin_Linux_noGUI/data.txt | tee ./" +folder+"/out \n")

            #print(i.split("_")[1].split('.')[0])

main()