#!/usr/bin/env python3
import os
import argparse
from typing import Optional, List
from qtaim_gen.source.utils.io import get_folders_from_file

should_stop = False


def handle_signal(signum, frame):
    global should_stop
    print(f"Received signal {signum}, initiating graceful shutdown...")
    should_stop = True


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for launching a batch of Parsl jobs.

    Args:
        argv: Optional list of CLI args (excludes program name). If None,
            arguments are read from sys.argv.

    Returns:
        Exit code (0 on success).
    """

    parser = argparse.ArgumentParser(
        prog="job-filter",
        description=(
            "Filter and refine a list of job folders for QTAIM analysis. "
            "This script processes the list of folders in --job_file and outputs a refined list."
        ),
    )

    parser.add_argument(
        "--num_folders",
        type=int,
        default=-1,
        help="number of folders to check and try to run",
    )

    parser.add_argument(
        "--job_file",
        type=str,
        help="file containing list of folders to run analysis on",
        default="./out.txt",
    )

    parser.add_argument(
        "--refined_job_file",
        type=str,
        help="file to save refined list of folders",
        default="refined_jobs.txt",
    )

    parser.add_argument(
        "--full_set",
        type=int,
        default=0,
        help="level of calculation detail (0-baseline, 1-modest, 2-full)",
    )

    parser.add_argument(
        "--root_omol_results",
        type=str,
        default=None,
        help="root where to store results, should mimic root_omol_inputs",
    )

    parser.add_argument(
        "--root_omol_inputs",
        type=str,
        default=None,
        help="root where input folders are located",
    )

    parser.add_argument(
        "--move_results",
        action="store_true",
        help="whether to move results from temp to final location",
    )

    parser.add_argument(
        "--check_orphaned",
        action="store_true",
        help="whether to check for orphaned jobs",
    )

    parser.add_argument(
        "--check_orca",
        action="store_true",
        help="require orca.json during validation (for retroactive ORCA .out parsing)",
    )

    args = parser.parse_args(argv)
    # print(args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    # Safely extract boolean flags and other values using getattr
    # overrun_running: bool = bool(getattr(args, "overrun_running", False))

    full_set: int = int(getattr(args, "full_set", 0))
    move_results: bool = bool(getattr(args, "move_results", False))
    job_file: str = getattr(args, "job_file")
    orphaned_check: bool = bool(getattr(args, "check_orphaned", False))
    check_orca: bool = bool(getattr(args, "check_orca", False))
    refined_job_file: str = getattr(args, "refined_job_file", "refined_jobs.txt")
    root_omol_results: Optional[str] = getattr(args, "root_omol_results", None)
    root_omol_inputs: Optional[str] = getattr(args, "root_omol_inputs", None)
    # if not set default to the entire length of the job file

    num_folders: int = int(getattr(args, "num_folders", -1))
    if num_folders <= 0:
        with open(job_file, "r") as f:
            num_folders = sum(1 for _ in f)

    # set mem
    # Basic static checks before launching heavy work
    if not os.path.exists(job_file):
        print(f"Error: job_file '{job_file}' does not exist")
        return 2

    folders_run = get_folders_from_file(
        job_file,
        num_folders,
        root_omol_results,
        root_omol_inputs,
        pre_validate=True,
        move_results=move_results,
        full_set=full_set,
        check_orca=check_orca,
    )

    if not folders_run:
        print(f"No folders found in {job_file}")
        return []

    # save refined job file
    with open(refined_job_file, "w") as f:
        for folder in folders_run:
            f.write(f"{folder}\n")

    if orphaned_check:
        count_orphaned = 0
        for folder_inputs in folders_run:
            # find jobs in root_omol_results that have
            if (
                root_omol_inputs
                and root_omol_results
                and folder_inputs.startswith(root_omol_inputs)
            ):
                folder_relative = folder_inputs[len(root_omol_inputs) :].lstrip(os.sep)
                folder_outputs = root_omol_results + os.sep + folder_relative
            else:
                folder_outputs = folder_inputs
            # check if folder exists with more than 3 files
            if os.path.exists(folder_outputs):
                # check if there is no orca.wfn file in the folder and that there is no gbw_analysis.log file
                if not os.path.exists(
                    os.path.join(folder_outputs, "orca.wfn")
                ) and os.path.exists(os.path.join(folder_outputs, "gbw_analysis.log")):
                    print(
                        f"Orphaned job detected: {folder_outputs} missing orca.wfn w log present"
                    )
                    count_orphaned += 1

                # num_files = len(os.listdir(folder_outputs))
                # if num_files > 3:
                #    print(f"Orphaned job detected: {folder_outputs} with {num_files} files")
                #    count_orphaned += 1

        print(f"Total orphaned jobs detected: {count_orphaned}")


if __name__ == "__main__":
    raise SystemExit(main())

"""
stragglers nakb, dna, droplet, mo_hydrides

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/mo_hydrides_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/mo_hydrides_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
1 left - 2/20

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/droplet_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/droplet_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
2 left - running rn (2/20)

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/nakb_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/nakb_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
22 left - running rn (2/20)

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/dna_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/dna_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
5 left  - running rn (2/20)

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/orbnet_denali_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/orbnet_denali_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
2 left - running rn (2/20)

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/spice_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/spice_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
1397 left - running rn (2/20)



# PACKAGED TOGETHER
python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/packaged_together.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/packaged_together.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 

python refine_list_of_jobs.py --root_omol_inputs /lus/eagle/projects/OMol25/ \
--job_file /lus/eagle/projects/generator/jobs_by_topdir/geom_orca6_refined.txt --move_results \
--refined_job_file /lus/eagle/projects/generator/jobs_by_topdir/geom_orca6_refined.txt \
    --full_set 0 --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ 
5 left - running rn (2/20)

cat geom_orca6_refined.txt >> packaged_together.txt

cat spice_refined.txt >> packaged_together.txt


# tuo
python refine_list_of_jobs.py --root_omol_inputs /p/lustre5/bennion1/Omol2025-4M-DiversitySet/ \
--job_file /p/lustre5/vargas58/generator_working/job_lists/ml_mo_refined.txt --move_results \
--refined_job_file /p/lustre5/vargas58/generator_working/job_lists/ml_mo_refined.txt \
    --full_set 0 --root_omol_results /p/lustre5/vargas58/OMol4M/



cat pmechdb_refined.txt >> packaged_together.txt
cat rmechdb_refined.txt >> packaged_together.txt
cat electrolytes_scaled_sep_refined.txt >> packaged_together.txt
cat electrolytes_reactivity_refined.txt >> packaged_together.txt
cat rgd_uks_refined.txt >> packaged_together.txt
cat rpmd_refined.txt >> packaged_together.txt
cat pdb_fragments_300K_refined.txt >> packaged_together.txt
cat pdb_pockets_400K_refined.txt >> packaged_together.txt
cat low_spin_23_refined.txt >> packaged_together.txt
cat pdb_fragments_400K_refined.txt >> packaged_together.txt
cat rna_refined.txt >> packaged_together.txt
cat tm_react_refined.txt >> packaged_together.txt
cat scaled_separations_exp_refined.txt >> packaged_together.txt


full-runner-parsl-alcf --num_folders 1 --orca_2mkl_cmd $HOME/orca_6_0_0/orca_2mkl \
     --multiwfn_cmd $HOME/Multiwfn_3_8/Multiwfn_noGUI --full_set 0 --type_runner local \
    --n_threads 220 --safety_factor 1.0 --move_results --preprocess_compressed --timeout_hr 12 \
    --queue workq-route  --restart  --n_nodes 1 --job_file ../jobs_by_topdir/spice_test.txt  --n_threads_per_job 10 \
    --preprocess_compressed --root_omol_results /lus/eagle/projects/generator/OMol25_postprocessing/ --root_omol_inputs /lus/eagle/projects/OMol25/

"""
