### generates monitoring db for tracking qtaim runs on OMol - 4M 
import sqlite3
import json
import pandas as pd
import os 
from datetime import datetime
import concurrent.futures

from qtaim_gen.source.utils.validation import (
    validation_checks, 
    validate_timing_dict, 
    validate_qtaim_dict,
    validate_bond_dict,
    validate_charge_dict,
    validate_fuzzy_dict, 
    get_charge_spin_n_atoms_from_folder
)

import numpy as np 
from qtaim_gen.source.core.parse_qtaim import dft_inp_to_dict

def process_folder(args):
    folder_path, full_set = args
    info = get_information_from_job_folder(folder_path, full_set)
    # Add job_id, subset, folder as needed
    return info

def get_charge_spin_n_atoms_from_folder(folder: str, logger=None, verbose=False) -> tuple:
    # check for a file ending with .inp
    inp_files = [f for f in os.listdir(folder) if f.endswith(".inp")]
    # add *.in files to list
    inp_files += [f for f in os.listdir(folder) if f.endswith(".in")]
    # remove convert.in
    inp_files = [f for f in inp_files if f != "convert.in"]

    if not inp_files:
        if logger:
            logger.error(f"No .inp file found in folder: {folder}.")
        if verbose:
            print(f"No .inp file found in folder: {folder}.")
        return False
    inp_file = inp_files[0]  # take the first .inp file found

    if logger:
        logger.info(f'Using input file "{inp_file}" for validation.')
        
    if verbose:
        print(f'Using input file "{inp_file}" for validation.')

    # gather n_atoms, spin, charge from the orca.inp file
    orca_inp_path = os.path.join(folder, inp_file)  # might need to change this name
    if not os.path.exists(orca_inp_path):
        if verbose:
            print(f"Missing orca.inp file at {orca_inp_path}.")
        if logger:
            logger.error(f"Missing orca.inp file at {orca_inp_path}.")
        return False
    return dft_inp_to_dict(orca_inp_path, parse_charge_spin=True)

def get_val_breakdown_from_folder(
        folder: str, 
        full_set: int, 
        spin_tf: bool, 
        n_atoms: int
    ) -> dict:
        
        info = {
            "total_time": None,
            "t_qtaim": None,
            "t_charge": None,
            "t_bond": None,
            "t_fuzzy": None,
            "t_other": None,
            "val_time": None,
            "val_qtaim": None,
            "val_charge": None,
            "val_bond": None,
            "val_fuzzy": None,
            "val_other": None,
        }   

        # check timings
        timings_file = os.path.join(folder, 'timings.json')
        if os.path.exists(timings_file):
            with open(timings_file, 'r') as f:
                timings = json.load(f)
            total_time = np.array(list(timings.values())).sum()
            info['total_time'] = total_time
            
            for col in timings.keys():
                info[f't_{col}'] = timings[col]
            val_time = validate_timing_dict(timings_file, logger=None, full_set=full_set, spin_tf=spin_tf)
            info['val_time'] = val_time

        # check fuzzy    
        fuzzy_file = os.path.join(folder, 'fuzzy_full.json')
        if os.path.exists(fuzzy_file):
            tf_fuzzy = validate_fuzzy_dict(fuzzy_file, logger=None, n_atoms=n_atoms, spin_tf=spin_tf, full_set=full_set,)
            info['val_fuzzy'] = tf_fuzzy
        
        # check charge
        charge_file = os.path.join(folder, 'charge.json')
        if os.path.exists(charge_file):
            tf_charge = validate_charge_dict(charge_file, logger=None)
            info['val_charge'] = tf_charge
        
        # check bond
        bond_file = os.path.join(folder, 'bond.json')
        if os.path.exists(bond_file):
            tf_bond = validate_bond_dict(bond_file, logger=None)
            info['val_bond'] = tf_bond
        
        # check qtaim
        qtaim_file = os.path.join(folder, 'qtaim.json')
        if os.path.exists(qtaim_file):
            tf_qtaim = validate_qtaim_dict(qtaim_file, n_atoms=n_atoms, logger=None)
            info['val_qtaim'] = tf_qtaim

        # echeck other 
        other_file = os.path.join(folder, 'other.json')
        if os.path.exists(other_file):
            tf_other = validate_timing_dict(other_file, logger=None, full_set=full_set, spin_tf=spin_tf)
            info['val_other'] = tf_other

        return info

def get_information_from_job_folder(folder: str, full_set: int) -> dict:
    """Extracts relevant information from the job folder name."""
    
    # check if folder has /generator/ subdirectory, if so get timings.json in that folder
    
    info = {
        "validation_level_0": None,
        "validation_level_1": None,
        "validation_level_2": None,
        "total_time": None,
        "t_qtaim": None,
        "t_other": None,
        "last_edit_time": None,
        "val_time": None,
        "val_qtaim": None,
        "val_charge": None,
        "val_bond": None,
        "val_fuzzy": None,
        "val_other": None,
        "n_atoms": None,
        "spin": None,
        "charge": None,
    }
    
    # get .inp file in the folder for spin, charge, n_atoms
    dft_dict = get_charge_spin_n_atoms_from_folder(folder, logger=None, verbose=False)
    n_atoms = len(dft_dict["mol"])
    spin = dft_dict.get("spin", None)
    charge = dft_dict.get("charge", None)
    
    info['n_atoms'] = n_atoms
    info['spin'] = spin
    info['charge'] = charge

    if spin != 1:
        spin_tf = True
    else:
        spin_tf = False
    
    
    # check is there is a generator subfolder
    print("Folder to analyze: ", folder)
    
    if 'generator' in os.listdir(folder):
        print("Found generator subfolder.")
        gen_folder = folder.split('/generator/')[0] + '/generator/'
        timings_file = os.path.join(gen_folder, 'timings.json')
        
        if os.path.exists(timings_file):
            with open(timings_file, 'r') as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info['total_time'] = total_time
            
            for col in timings.keys():
                info[f't_{col}'] = timings[col]

        tf_validation_level_0 = validation_checks(
            folder, 
            full_set=0, 
            verbose=False,
            move_results=True,
            logger=None
        )

        tf_validation_level_1 = validation_checks(
            folder, 
            full_set=1, 
            verbose=False,
            move_results=True,
            logger=None
        )

        tf_validation_level_2 = validation_checks(
            folder, 
            full_set=2, 
            verbose=False,
            move_results=True,
            logger=None
        )

        # set val_qtaim, val_charge, val_bond, val_fuzzy, val_other to True for corresponding level
        if full_set == 0:
            status_val = tf_validation_level_0
        elif full_set == 1:
            status_val = tf_validation_level_1
        elif full_set == 2:
            status_val = tf_validation_level_2

        if status_val:
            info['val_qtaim'] = True
            info['val_charge'] = True
            info['val_bond'] = True
            info['val_fuzzy'] = True
            info['val_other'] = True
            info["val_time"] = True
        else:
            dict_val = get_val_breakdown_from_folder(gen_folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf)
            info.update(dict_val)

        # check edit date of timings.json
        mtime_timestamp = os.path.getmtime(timings_file)
        # Convert the timestamp to a datetime object
        mtime_datetime = datetime.fromtimestamp(mtime_timestamp)
        # Format the datetime object into a human-readable string
        # Example format: YYYY-MM-DD HH:MM:SS
        human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
        info['last_edit_time'] = human_readable_mtime

        info.update({
            'validation_level_0': tf_validation_level_0,
            'validation_level_1': tf_validation_level_1,
            'validation_level_2': tf_validation_level_2
        })

    else: 
        timings_file = os.path.join(folder, 'timings.json')
        
        if os.path.exists(timings_file):
            with open(timings_file, 'r') as f:
                timings = json.load(f)
            total_time = float(np.array(list(timings.values())).sum())
            info['total_time'] = total_time
            
            for col in timings.keys():
                info[f't_{col}'] = timings[col]
        edit_time = os.path.getmtime(timings_file)
        # Convert the timestamp to a datetime object
        mtime_datetime = datetime.fromtimestamp(edit_time)
        # Format the datetime object into a human-readable string
        human_readable_mtime = mtime_datetime.strftime("%Y-%m-%d %H:%M:%S")
        info['last_edit_time'] = human_readable_mtime

        dict_val = get_val_breakdown_from_folder(folder, n_atoms=n_atoms, full_set=full_set, spin_tf=spin_tf)
        info.update(dict_val)

    return info
    
def validate_folder(folder_path):
    # Replace with your actual validation logic
    return os.path.exists(folder_path)  # Example: just checks existence

def scan_and_store_parallel(root_dir, db_path, full_set=0, max_workers=8):
    # Gather all folder paths to process
    jobs = []
    for job_id in os.listdir(root_dir):
        job_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(job_path):
            continue
        for subset in os.listdir(job_path):
            subset_path = os.path.join(job_path, subset)
            if not os.path.isdir(subset_path):
                continue
            for folder in os.listdir(subset_path):
                folder_path = os.path.join(subset_path, folder)
                if os.path.isdir(folder_path):
                    jobs.append((folder_path, full_set, job_id, subset, folder))

    # Parallel processing
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(get_information_from_job_folder, job[0], job[1]): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            info = future.result()
            info.update({"job_id": job[2], "subset": job[3], "folder": job[4]})
            results.append(info)

    # Write to DB in a single thread
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    columns = list(results[0].keys())
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    c.execute(f"CREATE TABLE IF NOT EXISTS validation ({col_defs})")
    for info in results:
        values = [str(info.get(col, "")) for col in columns]
        placeholders = ", ".join(["?"] * len(columns))
        c.execute(f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()



def print_summary(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Jobs with all validation columns True
    validation_cols = [
        'val_qtaim', 'val_charge', 'val_bond', 'val_fuzzy', 'val_other', 'val_time'
    ]
    where_clause = " AND ".join([f"{col}='True'" for col in validation_cols])
    c.execute(f"SELECT COUNT(DISTINCT job_id) FROM validation WHERE {where_clause}")
    all_valid_jobs = c.fetchone()[0]
    print(f"Jobs with all validation columns True: {all_valid_jobs}")

    # 2. Number of jobs done per category (by subset)
    c.execute("SELECT subset, COUNT(DISTINCT job_id) FROM validation GROUP BY subset")
    print("Number of jobs per category (subset):")
    for subset, count in c.fetchall():
        print(f"  {subset}: {count}")

    # 3. Average total_time per domain (subset)
    c.execute("SELECT subset, AVG(CAST(total_time AS FLOAT)) FROM validation GROUP BY subset")
    print("Average total_time per category (subset):")
    for subset, avg_time in c.fetchall():
        print(f"  {subset}: {avg_time:.2f}")

    # 4. Average number of atoms per domain (subset)
    c.execute("SELECT subset, AVG(CAST(n_atoms AS FLOAT)) FROM validation GROUP BY subset")
    print("Average number of atoms per category (subset):")
    for subset, avg_atoms in c.fetchall():
        print(f"  {subset}: {avg_atoms:.2f}")

    conn.close()

def scan_and_store(root_dir, db_path, full_set=0):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get all possible columns from a sample info dict
    sample_folder = None
    for job_id in os.listdir(root_dir):
        job_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(job_path):
            continue
        for subset in os.listdir(job_path):
            subset_path = os.path.join(job_path, subset)
            if not os.path.isdir(subset_path):
                continue
            for folder in os.listdir(subset_path):
                folder_path = os.path.join(subset_path, folder)
                if os.path.isdir(folder_path):
                    sample_folder = folder_path
                    break
            if sample_folder:
                break
        if sample_folder:
            break

    if sample_folder is None:
        print("No valid folder found for schema inference.")
        return

    info = get_information_from_job_folder(sample_folder, full_set)
    columns = list(info.keys()) + ["job_id", "subset", "folder"]

    # Create table with dynamic columns
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    c.execute(f"CREATE TABLE IF NOT EXISTS validation ({col_defs}, PRIMARY KEY(job_id, subset, folder))")

    for job_id in os.listdir(root_dir):
        job_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(job_path):
            continue
        for subset in os.listdir(job_path):
            subset_path = os.path.join(job_path, subset)
            if not os.path.isdir(subset_path):
                continue
            for folder in os.listdir(subset_path):
                folder_path = os.path.join(subset_path, folder)
                if not os.path.isdir(folder_path):
                    continue
                info = get_information_from_job_folder(folder_path, full_set)
                info.update({"job_id": job_id, "subset": subset, "folder": folder})
                values = [str(info.get(col, "")) for col in columns]
                # Check if row exists
                c.execute("SELECT * FROM validation WHERE job_id=? AND subset=? AND folder=?", (job_id, subset, folder))
                existing = c.fetchone()
                if existing:
                    # Compare values, update only if different
                    existing_dict = dict(zip(columns, existing))
                    if any(existing_dict.get(col, "") != str(info.get(col, "")) for col in columns):
                        set_clause = ", ".join([f"{col}=?" for col in columns])
                        c.execute(f"UPDATE validation SET {set_clause} WHERE job_id=? AND subset=? AND folder=?", values + [job_id, subset, folder])
                else:
                    placeholders = ", ".join(["?"] * len(columns))
                    c.execute(f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()



if __name__ == "__main__":
    root_dir = "/lus/eagle/projects/generator/OMol25_postprocessing/"  # Change to your root directory
    db_path = "validation_results.sqlite"
    scan_and_store_parallel(root_dir, db_path)
    print_summary(db_path)