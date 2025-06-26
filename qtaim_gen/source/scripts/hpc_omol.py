from qtaim_gen.source.core.omol import gbw_analysis
import os
import logging
import random


def main():
    overrun_running = False  # set to True if you want to run the script even if there are running jobs
    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000"  # can also be set in settings.ini
    # set mem
    os.system(
        "ulimit -s unlimited"
    )  # this sometimes doesn't work and I need to manually set this in cmdline
    # os.system("export Multiwfnpath=/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/")  # change this

    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # change this
    multiwfn_cmd = "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"  # change this

    folder_file = os.path.join("./out.txt")
    # read folder file and randomly select a folder
    with open(folder_file, "r") as f:
        folders = f.readlines()
    folders = [f.strip() for f in folders if f.strip()]  # remove empty lines
    if not folders:
        print("No folders found in out.txt")
        return

    hard_stop = len(folders) * 2
    for i in range(hard_stop):
        run_root = random.choice(folders)
        # check three states - not started, running, finished
        # check if there is a file called timings.json, qtaim.json, other.json, fuzzy_full,json, and charge.json
        if (
            os.path.exists(os.path.join(run_root, "timings.json"))
            or os.path.exists(os.path.join(run_root, "qtaim.json"))
            or os.path.exists(os.path.join(run_root, "other.json"))
            or os.path.exists(os.path.join(run_root, "fuzzy_full.json"))
            or os.path.exists(os.path.join(run_root, "charge.json"))
        ):
            print(f"Skipping {run_root} - already processed")
            continue

        # check if there are multiple .*mwfn files
        mwfn_files = [f for f in os.listdir(run_root) if f.endswith(".mwfn")]
        if len(mwfn_files) > 1 and not overrun_running:
            print(f"Skipping {run_root} - multiple mwfn files found")
            continue

        else:
            print(f"Processing {run_root} - {i+1}/{hard_stop}")
            try:
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(run_root, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )

                gbw_analysis(
                    folder=run_root,
                    orca_2mkl_cmd=orca6_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=False,
                    separate=True,
                    overwrite=True,
                    orca_6=True,
                    clean=False,
                    restart=False,
                    debug=False,
                    logger=logging.getLogger("gbw_analysis"),
                    preprocess_compessed=True,
                )  # works!
            except Exception as e:
                print(f"Error in gbw_analysis for {run_root}: {e}")

    pass


main()
