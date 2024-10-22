from pathlib import Path
import os, stat, json

from qtaim_gen.source.data.multiwfn import (
    charge_data, 
    bond_order_data, 
    fuzzy_data, 
    other_data, 
    qtaim_data
)

from qtaim_gen.source.core.parse_multiwfn import ( 
    parse_charge_doc, 
    parse_bond_order_doc, 
    parse_fuzzy_doc, 
    parse_other_doc
)

def write_multiwfn_conversion(
        out_folder, 
        read_file,
        overwrite=False, 
        name="convert.in",
        orca_2mkl_cmd="orca_2mkl"
    ): 
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    completed_tf = (
        os.path.exists(out_file)
        and os.path.getsize(out_file) > 0
    )  

    if completed_tf and not overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("{} ".format(orca_2mkl_cmd)+ str(Path.home().joinpath(out_folder, read_file)) + "\n")


def write_multiwfn_exe(
        out_folder, 
        read_file, 
        multi_wfn_cmd, 
        multiwfn_input_file, 
        convert_gbw=False, 
        overwrite=False, 
        name="props.mfwn"
        ): 
    """
    Function to write a bash script that runs multiwfn on a given input file.
    Args:
        out_folder(str): folder to write the bash script to
        read_file(str): file to read from
        multi_wfn_cmd(str): command to run multiwfn
        multiwfn_input_file(str): input file for multiwfn
        convert_gbw(bool): whether to convert the input file to a gbw file
        overwrite(bool): whether to overwrite the file if it already exists
        name(str): name of the bash script
    """

    out_file = str(Path.home().joinpath(out_folder, name))
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    completed_tf = (
        os.path.exists(out_file)
        and os.path.getsize(out_file) > 0
    )  

    if (not completed_tf) or overwrite:
        with open(out_file, "w") as f:
            f.write("#!/bin/bash\n")
            if convert_gbw: 
                f.write("orca_2mkl "+ str(Path.home().joinpath(out_folder)) + "\n")
            
            multiwfn_input_file_root = multiwfn_input_file.split("/")[-1].split(".")[0]

            
            f.write(
                "{} ".format(multi_wfn_cmd) # multiwfn command
                + str(Path.home().joinpath(out_folder, read_file)) # wfn/gbw file
                + " < {} | tee ".format(multiwfn_input_file) # multiwfn input file
                + str(Path.home().joinpath(out_folder, "{}.out".format(multiwfn_input_file_root))) # output file
                + "\n"
            )

        st = os.stat(out_file)
        os.chmod(out_file, st.st_mode | stat.S_IEXEC)


def create_jobs(folder, multiwfn_cmd, orca_2mkl_cmd): 
    """
    Create job files for multiwfn analysis
    """
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-root_folder", type=str, default="./")
    #parser.add_argument("-full_multiwfn", type=bool, default=True)
    #parser.add_argument("-routine_list", type=list, default=["qtaim"])
    #parser.add_argument("-multiwfn_cmd", type=str, default="Multiwfn"
    #args = parser.parse_args()
    #root_folder = args.root_folder
    #full_multiwfn = bool(args.full_multiwfn)
    #multiwfn_cmd = args.multiwfn_cmd
    
    routine_list = ["qtaim", "fuzzy", "bond", "charge", "other"]
    
    #print("root_folder: {}".format(folder))
    #print("multiwfn_cmd: {}".format(multiwfn_cmd))
    #print("routine_list: {}".format(routine_list))

    job_dict = {}
    for routine in routine_list:
        if routine == "qtaim":
            job_dict["qtaim"] = os.path.join(folder, "qtaim.txt")
            # write qtaim data file
            with open(os.path.join(folder, "qtaim.txt"), "w") as f:
                data = qtaim_data()
                f.write(data)
                
        elif routine == "fuzzy":
            job_dict["fuzzy"] = os.path.join(folder, "fuzzy.txt")
            with open(os.path.join(folder, "fuzzy.txt"), "w") as f:
                data = fuzzy_data()
                f.write(data)

        elif routine == "bond":
            job_dict["bond"] = os.path.join(folder, "bond.txt")
            with open(os.path.join(folder, "bond.txt"), "w") as f:
                data = bond_order_data()
                f.write(data)

        elif routine == "charge":
            job_dict["charge"] = os.path.join(folder, "charge.txt")
            with open(os.path.join(folder, "charge.txt"), "w") as f:
                data = charge_data()
                f.write(data)

        elif routine == "other":
            job_dict["other"] = os.path.join(folder, "other.txt")
            with open(os.path.join(folder, "other.txt"), "w") as f:
                data = other_data()
                f.write(data)
        else:
            print("routine not recognized")



    wfn_present = False
    for file in os.listdir(folder):
        if file.endswith(".wfn"): 
            wfn_present = True
            file_read = os.path.join(folder, file)     
        if file.endswith(".gbw"):
            bool_gbw = True
            file_gbw = os.path.join(folder, file)   
            
    if not wfn_present and bool_gbw: 
        # write conversion script from gbw to wfn
        write_multiwfn_conversion(
            out_folder=folder, 
            #out_file=file_gbw,
            overwrite=True, 
            name="convert.in",
            orca_2mkl_cmd=orca_2mkl_cmd
        )
    else:
        pass 
        #print("wfn present")

        for key, value in job_dict.items():
            #print("key: {}".format(key))
            write_multiwfn_exe(
                out_folder=folder,
                read_file=file_read, 
                multi_wfn_cmd=multiwfn_cmd, 
                multiwfn_input_file=value, 
                convert_gbw=False, 
                overwrite=True, 
                name="props_{}.mfwn".format(key), 
                orca_2mkl_cmd=orca_2mkl_cmd
            )


def run_jobs(folder):
    """
    Run conversion and multiwfn jobs
    """
    order_of_operations = ["qtaim", "fuzzy", "bond", "charge", "other"]
    wfn_present = False
    for file in os.listdir(folder):
        if file.endswith(".wfn"): 
            wfn_present = True
        if file.endswith("convert.in"):
            conv_file = os.path.join(folder, file)
            
    # run conversion script if wfn file is not present
    if not wfn_present:
        # run conversion script
        os.system("{}".format(conv_file))
    
    # run multiwfn scripts
    for order in order_of_operations:  
        mfwn_file = os.path.join(folder, "props_{}.mfwn".format(order))
        os.system("{}".format(mfwn_file))
    

def parse_multiwfn(folder):
    
    routine_list = ["fuzzy", "bond", "charge", "other", "qtaim"]

    for file in os.listdir(folder): 
        if file.endswith(".out"):
            file_full_path = os.path.join(folder, file)
            for routine in routine_list:
                if routine in file:
    
                    json_file = file_full_path.replace(".out", ".json")
                    
                    if routine == "fuzzy":
                        data=parse_fuzzy_doc(file_full_path)
                        with open(json_file, 'w') as f:
                            json.dump(data, f)
                    
                    elif routine == "bond":
                        data=parse_bond_order_doc(file_full_path)
                        with open(json_file, 'w') as f:
                            json.dump(data, f)

                    elif routine == "other":
                        data=parse_other_doc(file_full_path)
                        with open(json_file, 'w') as f:
                            json.dump(data, f)

                    elif routine == "charge":
                        charge_dict_overall, atomic_dipole_dict_overall, dipole_info=parse_charge_doc(file_full_path)    
                        charge_dict_overall = {
                            "charge": charge_dict_overall,
                            "dipole": dipole_info,
                            "atomic_dipole": atomic_dipole_dict_overall
                        }
                        with open(json_file, 'w') as f:
                            json.dump(charge_dict_overall, f)

                    elif routine == "qtaim":
                        pass
                        # TODO unify with .inp file
