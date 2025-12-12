### generates monitoring db for tracking qtaim runs on OMol - 4M 
import sqlite3
import pandas as pd
import json
import os 
from datetime import datetime
#import concurrent.futures
import concurrent.futures
from tqdm import tqdm

from qtaim_gen.source.utils.validation import (
    validation_checks, 
    get_charge_spin_n_atoms_from_folder,
    get_val_breakdown_from_folder
)

import numpy as np 

def process_folder(args):
    folder_path, full_set = args
    info = get_information_from_job_folder(folder_path, full_set)
    # Add job_id, subset, folder as needed
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
    print("DFT dict: ", dft_dict)
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
    #print("Folder to analyze: ", folder)
    
    if 'generator' in os.listdir(folder):
        #print("Found generator subfolder.")
        gen_folder = folder + '/generator/'
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

def scan_and_store(root_dir, db_path, full_set=0):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get all possible columns from a sample info dict
    sample_folder = None
    for job_id in os.listdir(root_dir):
        cat_path_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(cat_path_path):
            continue
        for subset in os.listdir(cat_path_path):
            subset_path = os.path.join(cat_path_path, subset)
            if os.path.isdir(subset_path):
                sample_folder = subset_path
                break
        if sample_folder:
            break

    if sample_folder is None:
        print("No valid folder found for schema inference.")
        return

    info = get_information_from_job_folder(sample_folder, full_set)
    columns = list(info.keys()) + ["job_id", "subset", "folder", "t_hirsh_fuzzy_spin"]
    columns = list(set(columns))  # ensure uniqueness

    # Create table with dynamic columns
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    c.execute(f"CREATE TABLE IF NOT EXISTS validation ({col_defs}, PRIMARY KEY(job_id, subset, folder))")

    for job_id in os.listdir(root_dir):
        cat_path_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(cat_path_path):
            continue
        for subset in os.listdir(cat_path_path):
            subset_path = os.path.join(cat_path_path, subset)
            if not os.path.isdir(subset_path):
                continue
            folder = subset_path  # treat subset_path as the folder
            if not os.path.isdir(folder):
                continue
            try: 
                info = get_information_from_job_folder(folder, full_set)
                info.update({"job_id": subset, "subset": job_id, "folder": folder})
                values = [str(info.get(col, "")) for col in columns]
                # Check if row exists
                c.execute("SELECT * FROM validation WHERE job_id=? AND subset=? AND folder=?", (subset, job_id, folder))
                existing = c.fetchone()
                if existing:
                    # Compare values, update only if different
                    existing_dict = dict(zip(columns, existing))
                    if any(existing_dict.get(col, "") != str(info.get(col, "")) for col in columns):
                        set_clause = ", ".join([f"{col}=?" for col in columns])
                        c.execute(f"UPDATE validation SET {set_clause} WHERE job_id=? AND subset=? AND folder=?", values + [subset, job_id, folder])
                else:
                    placeholders = ", ".join(["?"] * len(columns))
                    c.execute(f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})", values)
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
    conn.commit()
    conn.close()


def scan_and_store_parallel(root_dir, db_path, full_set=0, max_workers=8):
    import sqlite3

    # Gather all folders to process
    jobs = []
    for job_id in os.listdir(root_dir):
        cat_path_path = os.path.join(root_dir, job_id)
        if not os.path.isdir(cat_path_path):
            continue
        for subset in os.listdir(cat_path_path):
            subset_path = os.path.join(cat_path_path, subset)
            if os.path.isdir(subset_path):
                jobs.append((subset_path, full_set, subset, job_id, subset_path))

    # Use a sample folder to infer columns
    sample_folder = jobs[0][0] if jobs else None
    if sample_folder is None:
        print("No valid folder found for schema inference.")
        return

    info = get_information_from_job_folder(sample_folder, full_set)
    columns = list(info.keys()) + ["job_id", "subset", "folder", "t_hirsh_fuzzy_spin"]
    columns = list(set(columns))  # ensure uniqueness

    # Parallel folder processing with tqdm
    results = []
    def process_job(args):
        folder, full_set, subset, job_id, folder_path = args
        try:
            info = get_information_from_job_folder(folder, full_set)
            info.update({"job_id": subset, "subset": job_id, "folder": folder_path})
            return info
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for info in tqdm(executor.map(process_job, jobs), total=len(jobs), desc="Processing folders"):
            if info is not None:
                results.append(info)

    # Serial DB write with tqdm
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    c.execute(f"CREATE TABLE IF NOT EXISTS validation ({col_defs}, PRIMARY KEY(job_id, subset, folder))")

    for info in tqdm(results, desc="Writing to DB"):
        values = [str(info.get(col, "")) for col in columns]
        job_id = info["job_id"]
        subset = info["subset"]
        folder = info["folder"]
        # Check if row exists
        c.execute("SELECT * FROM validation WHERE job_id=? AND subset=? AND folder=?", (job_id, subset, folder))
        existing = c.fetchone()
        if existing:
            existing_dict = dict(zip(columns, existing))
            if any(existing_dict.get(col, "") != str(info.get(col, "")) for col in columns):
                set_clause = ", ".join([f"{col}=?" for col in columns])
                c.execute(
                    f"UPDATE validation SET {set_clause} WHERE job_id=? AND subset=? AND folder=?",
                    values + [job_id, subset, folder]
                )
        else:
            placeholders = ", ".join(["?"] * len(columns))
            c.execute(
                f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})",
                values
            )
    conn.commit()
    conn.close()

def create_overall_count_db(folder_jobs_OMol="/lus/eagle/projects/generator/jobs_by_topdir", db_path="/lus/eagle/projects/generator/jobs_by_topdir/overall_counts.sqlite"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # iterate through the .txt files in folder_jobs_OMol, count the lines in each file, name the key in the db the file name without .txt
    for file_name in os.listdir(folder_jobs_OMol):
        if file_name.endswith(".txt"):
            subset_name = file_name[:-4]  # remove .txt
            file_path = os.path.join(folder_jobs_OMol, file_name)
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f)
            # create table if not exists
            c.execute(f"CREATE TABLE IF NOT EXISTS overall_counts (subset TEXT PRIMARY KEY, count INTEGER)")
            # insert or replace count
            c.execute(f"INSERT OR REPLACE INTO overall_counts (subset, count) VALUES (?, ?)", (subset_name, line_count))
    
    conn.commit()
    conn.close()

def print_summary(db_path="/lus/eagle/projects/generator/jobs_by_topdir", path_to_overall_counts_db="/lus/eagle/projects/generator/jobs_by_topdir/overall_counts.sqlite"):
    if path_to_overall_counts_db: 
        counts_overall = {}       
        # read overall counts and add them to the summary
        overall_conn = sqlite3.connect(path_to_overall_counts_db)
        overall_c = overall_conn.cursor()
        overall_c.execute("SELECT subset, count FROM overall_counts")
        print("Overall job counts per category (subset):")
        for subset, count in overall_c.fetchall():
            counts_overall[subset] = count
        overall_conn.close()


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

    # valid jobs per category
    c.execute(f"""SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE {where_clause} GROUP BY subset""")
    print(f"Subset valid jobs per category (Total - {all_valid_jobs}):")
    for subset, count in c.fetchall():
        if path_to_overall_counts_db:
            print(f"  {subset}: \t {count} / {counts_overall.get(subset, 'N/A')}")
        else: 
            print(f"  {subset}: \t {count}")

    # 2. Number of jobs done per category (by subset)
    c.execute("SELECT subset, COUNT(DISTINCT job_id) FROM validation GROUP BY subset")
    print("Number of jobs per category (subset):")
    for subset, count in c.fetchall():
        if path_to_overall_counts_db:
            print(f"  {subset}: \t {count} / {counts_overall.get(subset, 'N/A')}")
        else:
            print(f"  {subset}: {count}")

    # 3. Average total_time per category (subset)
    c.execute("SELECT subset, AVG(CAST(total_time AS FLOAT)) FROM validation GROUP BY subset")
    print("Average total_time per category (subset):")
    for subset, avg_time in c.fetchall():
        print(f"  {subset}: \t {avg_time:.2f}")

    # 4. Average number of atoms per category (subset)
    c.execute("SELECT subset, AVG(CAST(n_atoms AS FLOAT)) FROM validation GROUP BY subset")
    print("Average number of atoms per category (subset):")
    for subset, avg_atoms in c.fetchall():
        print(f"  {subset}: \t {avg_atoms:.2f}")


    # 5. Calcs completed over the last 24 hours    since_time = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    since_time = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("SELECT COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ?", (since_time,))
    recent_jobs = c.fetchone()[0]

    # 6. Calcs over last 24 hours per category
    c.execute("SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ? GROUP BY subset", (since_time,))
    print(f"Subset jobs completed in the last 24 hours (Total - {recent_jobs}):")
    for subset, count in c.fetchall():
        print(f"  {subset}: \t {count}")

    # 7. print all counts from overall counts db
    if path_to_overall_counts_db:
        print("Overall job counts per category (subset):")
        for subset, count in counts_overall.items():
            print(f"  {subset}: \t {count}")    

    conn.close()


if __name__ == "__main__":
    root_dir = "/lus/eagle/projects/generator/OMol25_postprocessing/"  # Change to your root directory
    db_path = "validation_results.sqlite"
    #scan_test(root_dir, db_path)
    #scan_and_store_parallel(root_dir, db_path)
    print_summary(db_path)


    root_data_dir = "/usr/workspace/vargas58/orca_test/wave_2_benchmarks_filtered/"
    calc_root_dir = "/p/lustre5/vargas58/maria_benchmarks/wave2_omol_sp_tight/"
    
    root_data_dir = "/usr/workspace/vargas58/orca_test/wave_2_benchmarks_filtered/"
    calc_root_dir = "/p/lustre5/vargas58/maria_benchmarks/wave2_omol_opt_tight/"