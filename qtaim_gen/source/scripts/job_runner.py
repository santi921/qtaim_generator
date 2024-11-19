from qtaim_gen.source.core.omol import gbw_analysis
import os 
from random import shuffle

def main():

    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000"
    # set mem
    os.system("ulimit -s unlimited")

    root_folder = ""    
    orca2mkl = "/home/santiagovargas/dev/orca5/orca_2mkl"
    multiwfn_cmd = (
        
    )

    parse_only=False
    separate=True
    overwrite=True
    clean=True
    orca_6=True
    restart=False
    runs = 1000
    # iterate through folders in root_folder randomly
    for i in range(runs):
        folders = os.listdir(root_folder)
        shuffle(folders)
        for folder in folders:
            folder_full = os.path.join(root_folder, folder)

            try:
                gbw_analysis(
                    folder=folder_full,
                    orca_2mkl_cmd=orca2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite, 
                    orca_6=orca_6,
                    clean=clean, 
                    restart=restart
                )  # works!
            except:
                print(f"Error in gbw_analysis - case {i}")
                continue
   
main()
