### generates monitoring db for tracking qtaim runs on OMol - 4M
import sqlite3
import pandas as pd
import os
from datetime import datetime
# import concurrent.futures
import concurrent.futures
from tqdm import tqdm

from qtaim_gen.source.utils.validation import get_information_from_job_folder

import logging


def scan_and_store_parallel(root_dir, db_path, full_set=0, max_workers=8, sub_dirs_to_sweep=None, debug=False):
    import sqlite3
    # Set up logging - save log to file 
--

    # Gather all folders to process
    jobs = []
    for job_id in os.listdir(root_dir):
        if sub_dirs_to_sweep is not None and job_id not in sub_dirs_to_sweep:
            logger.info(f"Skipping subset: {job_id}")

        if sub_dirs_to_sweep is not None and job_id in sub_dirs_to_sweep:
            logger.info(f"Running subset: {job_id}")
        
            cat_path_path = os.path.join(root_dir, job_id)
            if not os.path.isdir(cat_path_path):
                continue
            
            for subset in os.listdir(cat_path_path):
                subset_path = os.path.join(cat_path_path, subset)
                if os.path.isdir(subset_path):
                    jobs.append((subset_path, full_set, subset, job_id, subset_path))
    if debug: 
        jobs = jobs[:100]  # limit for debugging
    
    logger.info(f"Total folders to process: {len(jobs)}")
    # Use a sample folder to infer columns
    sample_folder = jobs[0][0] if jobs else None
    if sample_folder is None:
        print("No valid folder found for schema inference.")
        return

    info = get_information_from_job_folder(sample_folder, full_set)
    columns = list(info.keys()) + ["job_id", "subset", "folder", "t_hirsh_fuzzy_spin", "t_hirsh_fuzzy_density", "t_becke_fuzzy_density", "t_becke_fuzzy_spin"]
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
            logger.error(f"Error processing folder {folder}: {e}")
            #print(f"Error processing folder {folder}: {e}")
            return None

    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for info in tqdm(
            executor.map(process_job, jobs), total=len(jobs), desc="Processing folders"
        ):
            if info is not None:
                results.append(info)
    
    #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #    for info in tqdm(executor.map(process_job, jobs), total=len(jobs), desc="Processing folders"):
    #        if info is not None:
    #            results.append(info)

    # Serial DB write with tqdm
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    col_defs = ", ".join([f"{col} TEXT" for col in columns])
    c.execute(
        f"CREATE TABLE IF NOT EXISTS validation ({col_defs}, PRIMARY KEY(job_id, subset, folder))"
    )

    upserts = []
    for info in results:
        values = [str(info.get(col, "")) for col in columns]
        job_id = info["job_id"]
        subset = info["subset"]
        folder = info["folder"]
        upserts.append((values, job_id, subset, folder))

    for values, job_id, subset, folder in tqdm(upserts, desc="Writing to DB"):
        c.execute("SELECT * FROM validation WHERE job_id=? AND subset=? AND folder=?", (job_id, subset, folder))
        existing = c.fetchone()
        if existing:
            existing_dict = dict(zip(columns, existing))
            if any(existing_dict.get(col, "") != str(val) for col, val in zip(columns, values)):
                set_clause = ", ".join([f"{col}=?" for col in columns])
                c.execute(f"UPDATE validation SET {set_clause} WHERE job_id=? AND subset=? AND folder=?", values + [job_id, subset, folder])
        else:
            placeholders = ", ".join(["?"] * len(columns))
            c.execute(f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()

    """NON-Batched
    for info in tqdm(results, desc="Writing to DB"):
        values = [str(info.get(col, "")) for col in columns]
        job_id = info["job_id"]
        subset = info["subset"]
        folder = info["folder"]
        # Check if row exists
        c.execute(
            "SELECT * FROM validation WHERE job_id=? AND subset=? AND folder=?",
            (job_id, subset, folder),
        )
        existing = c.fetchone()
        if existing:
            existing_dict = dict(zip(columns, existing))
            if any(
                existing_dict.get(col, "") != str(info.get(col, "")) for col in columns
            ):
                set_clause = ", ".join([f"{col}=?" for col in columns])
                c.execute(
                    f"UPDATE validation SET {set_clause} WHERE job_id=? AND subset=? AND folder=?",
                    values + [job_id, subset, folder],
                )
        else:
            placeholders = ", ".join(["?"] * len(columns))
            c.execute(
                f"INSERT INTO validation ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
    conn.commit()
    conn.close()
    """

def create_overall_count_db(
    folder_jobs_OMol="/lus/eagle/projects/generator/jobs_by_topdir",
    db_path="/lus/eagle/projects/generator/jobs_by_topdir/overall_counts.sqlite",
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # iterate through the .txt files in folder_jobs_OMol, count the lines in each file, name the key in the db the file name without .txt
    for file_name in os.listdir(folder_jobs_OMol):
        if file_name.endswith(".txt"):
            subset_name = file_name[:-4]  # remove .txt
            file_path = os.path.join(folder_jobs_OMol, file_name)
            with open(file_path, "r") as f:
                line_count = sum(1 for line in f)
            # create table if not exists
            c.execute(
                f"CREATE TABLE IF NOT EXISTS overall_counts (subset TEXT PRIMARY KEY, count INTEGER)"
            )
            # insert or replace count
            c.execute(
                f"INSERT OR REPLACE INTO overall_counts (subset, count) VALUES (?, ?)",
                (subset_name, line_count),
            )

    conn.commit()
    conn.close()


def print_summary(
    db_path="/lus/eagle/projects/generator/jobs_by_topdir",
    path_to_overall_counts_db="/lus/eagle/projects/generator/jobs_by_topdir/overall_counts.sqlite",
):

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

    def get_tabs(subset):
        length = len(subset)
        if length <= 8:
            return "\t\t\t"
        elif length <= 16:
            return "\t\t"
        elif length <= 24:
            return "\t"
        else:
            return ""

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Jobs with all validation columns True
    validation_cols = [
        "val_qtaim",
        "val_charge",
        "val_bond",
        "val_fuzzy",
        "val_other",
        "val_time",
    ]
    where_clause = " AND ".join([f"{col}='True'" for col in validation_cols])
    c.execute(f"SELECT COUNT(DISTINCT job_id) FROM validation WHERE {where_clause}")
    all_valid_jobs = c.fetchone()[0]
    print("---" * 30)
    print(f"Jobs with all validation columns True: {all_valid_jobs}")

    # valid jobs per category
    c.execute(
        f"""SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE {where_clause} GROUP BY subset"""
    )
    print(f"Subset valid jobs per category (Total - {all_valid_jobs}):")
    for subset, count in c.fetchall():
        if path_to_overall_counts_db:
            print(
                f"  {subset}: {get_tabs(subset)} {count} / {counts_overall.get(subset, 'N/A')}"
            )
        else:
            print(f"  {subset}: {get_tabs(subset)} {count}")

    # table of number of val True per category
    print("---" * 30)
    print("Number of jobs passing each validation per category (subset):")
    for col in validation_cols:
        c.execute(
            f"""SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE {col}='True' GROUP BY subset"""
        )
        print(f"  Validation: {col}")
        for subset, count in c.fetchall():
            if path_to_overall_counts_db:
                print(
                    f"    {subset}: {get_tabs(subset)} {count} / {counts_overall.get(subset, 'N/A')}"
                )
            else:
                print(f"    {subset}: {get_tabs(subset)} {count}")

    # 2. Number of jobs done per category (by subset)
    c.execute("SELECT subset, COUNT(DISTINCT job_id) FROM validation GROUP BY subset")
    print("---" * 30)
    print("Number of jobs per category (subset):")
    for subset, count in c.fetchall():
        if path_to_overall_counts_db:
            print(
                f"  {subset}: {get_tabs(subset)} {count} / {counts_overall.get(subset, 'N/A')}"
            )
        else:
            print(f"  {subset}: {get_tabs(subset)} {count}")

    # 3. Average total_time per category (subset)
    c.execute(
        "SELECT subset, AVG(CAST(total_time AS FLOAT)) FROM validation GROUP BY subset"
    )
    print("---" * 30)
    print("Average total_time per category (subset):")
    for subset, avg_time in c.fetchall():
        print(f"  {subset}: {get_tabs(subset)} {avg_time:.2f}")

    # 4. Average number of atoms per category (subset)
    c.execute(
        "SELECT subset, AVG(CAST(n_atoms AS FLOAT)) FROM validation GROUP BY subset"
    )
    print("---" * 30)
    print("Average number of atoms per category (subset):")
    for subset, avg_atoms in c.fetchall():
        print(f"  {subset}: {get_tabs(subset)} {avg_atoms:.2f}")

    # 5. Calcs completed over the last 24 hours    since_time = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    # add since last hour as well
    since_time = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    since_time_hr = (datetime.now() - pd.Timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    c.execute(
        "SELECT COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ?",
        (since_time,),
    )
    recent_jobs = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ?",
        (since_time_hr,),
    )
    recent_jobs_hr = c.fetchone()[0]

    # 6. Calcs over last 24 hours per category
    print("---" * 30)
    print(
        f"Subset jobs working in the last 24 hours / 1 hour (Total - {recent_jobs} / {recent_jobs_hr}):"
    )
    dict_one_day = {}
    dict_one_hour = {}
    dict_one_day_full_val = {}
    dict_one_hour_full_val = {}

    c.execute(
        "SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ? GROUP BY subset",
        (since_time,),
    )
    for subset, count in c.fetchall():
        dict_one_day[subset] = count

    c.execute(
        "SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ? GROUP BY subset",
        (since_time_hr,),
    )
    for subset, count in c.fetchall():
        dict_one_hour[subset] = count

    c.execute(
        f"""SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ? AND {where_clause} GROUP BY subset""",
        (since_time,),
    )
    for subset, count in c.fetchall():
        dict_one_day_full_val[subset] = count

    c.execute(
        f"""SELECT subset, COUNT(DISTINCT job_id) FROM validation WHERE last_edit_time > ? AND {where_clause} GROUP BY subset""",
        (since_time_hr,),
    )
    for subset, count in c.fetchall():
        dict_one_hour_full_val[subset] = count

    for subset in set(list(dict_one_day.keys()) + list(dict_one_hour.keys())):
        count_day = dict_one_day.get(subset, 0)
        count_hr = dict_one_hour.get(subset, 0)
        print(
            f"  {subset}: {get_tabs(subset)} {count_day} / {count_hr} / {dict_one_day_full_val.get(subset, 0)} / {dict_one_hour_full_val.get(subset, 0)}"
        )

    # 7. print all counts from overall counts db
    if path_to_overall_counts_db:
        print("---" * 30)
        print("Overall job counts per category (subset):")
        for subset, count in counts_overall.items():
            # add tabs depending on the length of subset

            print(f"  {subset}: {get_tabs(subset)} {count}")

    conn.close()


if __name__ == "__main__":
    root_dir = "/lus/eagle/projects/generator/OMol25_postprocessing/"  # Change to your root directory
    db_path = "validation_results.sqlite"
    # scan_test(root_dir, db_path)
    # scan_and_store_parallel(root_dir, db_path)
    print_summary(db_path)

    root_data_dir = "/usr/workspace/vargas58/orca_test/wave_2_benchmarks_filtered/"
    calc_root_dir = "/p/lustre5/vargas58/maria_benchmarks/wave2_omol_sp_tight/"

    root_data_dir = "/usr/workspace/vargas58/orca_test/wave_2_benchmarks_filtered/"
    calc_root_dir = "/p/lustre5/vargas58/maria_benchmarks/wave2_omol_opt_tight/"
