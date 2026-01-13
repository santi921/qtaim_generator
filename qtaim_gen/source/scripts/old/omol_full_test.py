from qtaim_gen.source.core.omol import gbw_analysis, write_settings_file
import os
import argparse
import logging

# add path for shared version
# export PATH="/home/santiagovargas/dev/orca5/:$PATH"; export LD_LIBRARY_PATH="/home/santiagovargas/dev/orca5/:$LD_LIBRARY_PATH"


def main():
    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000"
    # set mem
    os.system("ulimit -s unlimited")
    os.system("export LD_LIBRARY_PATH=/home/santiagovargas/dev/orca5/:$LD_LIBRARY_PATH")
    os.system(
        "export Multiwfnpath=/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/"
    )

    parser = argparse.ArgumentParser(
        description="Run gbw_analysis on ORCA output files."
    )
    parser.add_argument(
        "test_case",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, -1, -2, -3],
        help="Test case number to run (0-6) or -1, -2, -3.",
    )

    parser.add_argument(
        "threads",
        type=int,
        help="Number of threads to use for the analysis.",
        default=8,
    )

    args = parser.parse_args()
    test_case = args.test_case
    threads = int(args.threads)

    orca_base = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/base_test/".format(
        threads
    )

    orca_6 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca6/".format(
        threads
    )
    orca_5 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca5/".format(
        threads
    )
    orca_6_rks = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca6_rks/".format(
        threads
    )
    orca_6_uks = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca6_uks/".format(
        threads
    )
    orca_5_rks = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca5_rks/".format(
        threads
    )
    orca_5_uks = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/{}_threads/orca5_uks/".format(
        threads
    )

    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # orca --molden
    orca5_2mkl = "/home/santiagovargas/dev/orca5/orca_2mkl"

    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    )
    Multiwfnpath = "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/"
    os.environ["Multiwfnpath"] = Multiwfnpath

    parse_only = False
    separate = True
    overwrite = False
    restart=True
    clean = True
    debug = False
    # create logger in target folder

    print(f"Running test case {test_case}...")

    def run_case(test_case):
        if test_case == 0:
            try:
                folder = orca_6_rks
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(folder, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!
                gbw_analysis(
                    folder=folder,
                    orca_2mkl_cmd=orca6_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite,
                    orca_6=True,
                    clean=clean,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:

                print("Error in gbw_analysis - case 2")

        elif test_case == 1:
            try:
                folder = orca_5_uks
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(folder, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!
                gbw_analysis(
                    folder=folder,
                    orca_2mkl_cmd=orca5_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite,
                    orca_6=False,
                    clean=clean,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:

                print("Error in gbw_analysis - case 6")

        elif test_case == 2:
            try:
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(orca_6_uks, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!

                gbw_analysis(
                    folder=orca_6_uks,
                    orca_2mkl_cmd=orca6_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite,
                    orca_6=True,
                    clean=clean,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:

                print("Error in gbw_analysis - case 3")

        elif test_case == 3:
            try:
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(orca_5_rks, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!
                gbw_analysis(
                    folder=orca_5_rks,
                    orca_2mkl_cmd=orca5_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite,
                    orca_6=False,
                    clean=clean,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:

                print("Error in gbw_analysis - case 5")

        elif test_case == 4:
            print(orca_base)
            # create logger in target folder
            logging.basicConfig(
                filename=os.path.join(orca_base, "gbw_analysis.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

            # works!
            gbw_analysis(
                folder=orca_base,
                orca_2mkl_cmd=orca6_2mkl,
                multiwfn_cmd=multiwfn_cmd,
                parse_only=parse_only,
                separate=separate,
                overwrite=False,
                orca_6=True,
                clean=False,
                restart=True,
                debug=debug,
                logger=logging.getLogger("gbw_analysis"),
                mem=400000000,
                n_threads=threads,
            )  # works!
            try:
                print(orca_base)
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(orca_base, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                """
                # works!
                gbw_analysis(
                    folder=orca_base,
                    orca_2mkl_cmd=orca5_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=True,
                    orca_6=True,
                    clean=clean,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!"""
            except:
                print("Error in gbw_analysis - case 0")

        elif test_case == 5:
            try:

                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(orca_6, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!
                gbw_analysis(
                    folder=orca_6,
                    orca_2mkl_cmd=orca6_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=parse_only,
                    separate=separate,
                    overwrite=overwrite,
                    orca_6=True,
                    clean=True,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:

                print("Error in gbw_analysis - case 1")

        elif test_case == 6:

            try:
                # create logger in target folder
                logging.basicConfig(
                    filename=os.path.join(orca_5, "gbw_analysis.log"),
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )
                # works!
                gbw_analysis(
                    folder=orca_5,
                    orca_2mkl_cmd=orca5_2mkl,
                    multiwfn_cmd=multiwfn_cmd,
                    parse_only=False,
                    separate=True,
                    overwrite=True,
                    orca_6=False,
                    clean=False,
                    restart=False,
                    debug=debug,
                    logger=logging.getLogger("gbw_analysis"),
                    mem=400000000,
                    n_threads=threads,
                )  # works!
            except:
                print("Error in gbw_analysis - case 4")

    if test_case == -1:
        run_case(0)
        run_case(1)
        run_case(2)

    elif test_case == -2:
        run_case(3)
        run_case(4)
        run_case(5)
        run_case(6)

    elif test_case == -3:
        run_case(6)
        run_case(3)
        run_case(4)
        run_case(5)
        run_case(0)
        run_case(1)
        run_case(2)

    else:
        run_case(test_case)


main()


# Parameters
# SBATCH --account=m4298_g
# SBATCH --cpus-per-task=5
# SBATCH --error=/pscratch/sd/s/santiago/fairchem/debug/202506-1914-3342-d94e/logs/%j_0_log.err
# SBATCH --gpus-per-node=1
# SBATCH --job-name=uma
# SBATCH --mem=80GB
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --open-mode=append
# SBATCH --output=/pscratch/sd/s/santiago/fairchem/debug/202506-1914-3342-d94e/logs/%j_0_log.out
# SBATCH --qos=premium
# SBATCH --signal=USR2@90
# SBATCH --time=60
# SBATCH --wckey=submitit


# Parameters
# SBATCH --account=m4298_g
# SBATCH --constraint=gpu
# SBATCH --cpus-per-task=5
# SBATCH --error=/pscratch/sd/s/santiago/MinDScAIP/debug/202506-1914-1835-b49a/logs/%j_0_log.err
# SBATCH --gpus-per-node=1
# SBATCH --job-name=mindscaip_medium
# SBATCH --mem=80GB
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --open-mode=append
# SBATCH --output=/pscratch/sd/s/santiago/MinDScAIP/debug/202506-1914-1835-b49a/logs/%j_0_log.out
# SBATCH --qos=premium
# SBATCH --signal=USR2@90
# SBATCH --time=60
# SBATCH --wckey=submitit
