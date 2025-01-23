import os 
import pickle as pkl
import lmdb
import json
import pickle
from typing import List, Dict, Optional
import glob as glob



def running_average(old_avg: float, new_value: float, n: int, n_new: Optional[int] = 1) -> float:
    """simple running average
    Args:
        old_avg (float): old average
        new_value (float): new value
        n (int): number of samples
        n_new (Optional[int]): number of new samples
    """
    return old_avg + (new_value - old_avg) * n_new / n


def write_lmdb(data: dict[dict], lmdb_dir: str, lmdb_name: str, global_values: Optional[Dict[str, float]] = {}): 
    """General method to write data to an LMDB file.
    Args:
        data (dict[dict]): Data to write to the LMDB file.
        lmdb_dir (str): Directory to write the LMDB file.
        lmdb_name (str): Name of the LMDB file.
        global_values (Optional[Dict[str, float]], optional): Global values to write to the LMDB file. Defaults to {}.
    """

    if not os.path.exists(lmdb_dir):
        os.makedirs(lmdb_dir, exist_ok=True)

    db = lmdb.open(
        lmdb_dir + lmdb_name,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # write samples
    for ind, sample in data.items():

        sample_index = ind
        txn = db.begin(write=True)
        txn.put(
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )

        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(data), protocol=-1))
    txn.commit()

    if global_values != {}:
        for key, value in global_values.items():
            txn = db.begin(write=True)
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
            txn.commit()

    db.sync()
    db.close()


def merge_lmdbs(db_paths: str, out_path: str, output_file: str):
    env_out = lmdb.open(
        os.path.join(out_path, output_file),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    
    idx = 0
    for db_path in db_paths:
        #print("merge in {}".format(db_path))
        env_in = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        
        #should set indexes so that properties do not writtent down as well.
        with env_out.begin(write=True) as txn_out, env_in.begin(write=False) as txn_in:
            cursor = txn_in.cursor()
            for key, value in cursor:
                if key.decode("ascii") != "length":
                    try:
                        #int(key.decode("ascii"))
                        txn_out.put(
                        f"{key}".encode("ascii"),
                        value,
                        )
                        idx+=1
                        #print(idx)
                    #write properties
                    except ValueError:
                        txn_out.put(
                            key,
                            value
                        )
        env_in.close()
    
    #update length
    txn_out=env_out.begin(write=True)
    txn_out.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn_out.commit()
        
    env_out.sync()
    env_out.close()


def cleanup_lmdb_files(directory: str, pattern: str, dry_run: Optional[bool] = False):
    """
    Clean LMDB files in a directory that match a pattern.
    Args:
        directory (str): Directory to search for files.
        pattern (str): Pattern to match files.
        dry_run (Optional[bool], optional): If True, do not delete files. Defaults to False.
    """
    file_list = glob.glob(os.path.join(directory, pattern))

    for file_path in file_list:
        try:
            if not dry_run:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else: 
                print(f"Dry run, would delete file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {file_path}. {str(e)}")


def split_list(lst: list, chunk_size: int):
    if chunk_size == -1: 
        yield lst
    
    elif chunk_size < len(lst):
        for i in range(0, len(lst), chunk_size):  
            yield lst[i:i + chunk_size]
    else: 
        yield lst
    

def json_2_lmdbs(root_dir: str, out_dir:str, data_type: str, out_lmdb: str, chunk_size: int, clean: Optional[bool]=False):
    
    chunk_ind = 1
    files_target = glob.glob(root_dir + "*/{}.json".format(data_type))
    data_dict = {}
    
    for chunk in split_list(files_target, chunk_size):
        for file in chunk: 
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict[name] = data
        
        write_lmdb(data_dict, out_dir, f"{data_type}_{chunk_ind}.lmdb")
        chunk_ind += 1

    files_out = glob.glob("{}/{}_*.lmdb".format(root_dir, data_type))
    merge_lmdbs(files_out, out_dir, out_lmdb)
    cleanup_lmdb_files(directory=out_dir, pattern="{}_*.lmdb".format(data_type), dry_run=not clean)
    cleanup_lmdb_files(directory=out_dir, pattern="{}_*.lmdb-lock".format(data_type), dry_run=not clean)

'''
def main():
     
    
    dir_active = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/"
    chunk_size = 3
    json_2_lmdbs(dir_active, "./lmdb_test/", "charge", "merged_charge.lmdb", chunk_size, clean=True)
    json_2_lmdbs(dir_active, "./lmdb_test/", "bond", "merged_bond.lmdb", chunk_size, clean=True)
    json_2_lmdbs(dir_active, "./lmdb_test/", "other", "merged_other.lmdb", chunk_size, clean=True)
    json_2_lmdbs(dir_active, "./lmdb_test/", "qtaim", "merged_qtaim.lmdb", chunk_size, clean=True)
    """
    files_charge = glob.glob(dir_active + "*/charge.json")
    files_other = glob.glob(dir_active + "*/other.json")
    files_bond = glob.glob(dir_active + "*/bond.json")
    files_qtaim = glob.glob(dir_active + "*/qtaim.json")

    data_dict_charge = {}
    data_dict_other = {}
    data_dict_bond = {}
    data_dict_qtaim = {}

    chunk_size = 2
    chunk_ind = 1    


    for chunk in split_list(files_charge, chunk_size):
        data_dict_charge = {}
        for file in chunk: 
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict_charge[name] = data
        
        write_lmdb(data_dict_charge, "./lmdb_test/", f"charge_{chunk_ind}.lmdb")
        chunk_ind += 1
        
    chunk_ind = 1
    for chunk in split_list(files_bond, chunk_size):
        data_dict_bond = {}
        for file in chunk:
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict_bond[name] = data

        write_lmdb(data_dict_bond, "./lmdb_test/", f"bond_{chunk_ind}.lmdb")
        chunk_ind += 1

    chunk_ind = 1
    for chunk in split_list(files_other, chunk_size):
        data_dict_other = {}
        for file in chunk:
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict_other[name] = data

        write_lmdb(data_dict_other, "./lmdb_test/", f"other_{chunk_ind}.lmdb")
        chunk_ind += 1

    chunk_ind = 1   
    for chunk in split_list(files_qtaim, chunk_size):
        data_dict_qtaim = {}
        for file in chunk:
            with open(file, "r") as f:
                data = json.load(f)
                name = file.split("/")[-2]
                data_dict_qtaim[name] = data
                
        write_lmdb(data_dict_qtaim, "./lmdb_test/", f"qtaim_{chunk_ind}.lmdb")
        chunk_ind += 1



    print("done!")
    files_charge_out = glob.glob("./lmdb_test/charge_*.lmdb")
    merge_lmdbs(files_charge_out, "./lmdb_test/", "merged_charge.lmdb")

    # read merged file 
    env = lmdb.open(
        "./lmdb_test/merged_charge.lmdb",
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )
    #for key, value in env.begin().cursor():
    #print(key.decode("ascii"))
    #print(pkl.loads(value))
    
    
    print("clean files")
    cleanup_lmdb_files(directory="./lmdb_test/", pattern="bond_*.lmdb", dry_run=True)
    cleanup_lmdb_files(directory="./lmdb_test/", pattern="charge_*.lmdb", dry_run=True)
    cleanup_lmdb_files(directory="./lmdb_test/", pattern="other_*.lmdb", dry_run=True)
    cleanup_lmdb_files(directory="./lmdb_test/", pattern="qtaim_*.lmdb", dry_run=True)
    """


main()

# final global dict for qtaim-embed should have the following: 
# - feature size 
# - feature names 
# - element set 
# - ring_size_set??
# - allow / included charges 
# - allowed/included spins

'''