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
        "-test_case",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        help="Test case number to run (0-6) or -1, -2, -3.",
    )

    parser.add_argument(
        "-threads",
        type=int,
        help="Number of threads to use for the analysis.",
        default=4,
    )

    args = parser.parse_args()
    test_case = args.test_case
    threads = int(args.threads)

    orca_base = (
        "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test/"
    )
    orca_base_2 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_2/"
    orca_base_3 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_3/"
    orca_base_4 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_4/"
    orca_base_5 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_5/"
    orca_base_6 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_6/"
    orca_base_7 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_7/"
    orca_base_8 = "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_8/"

    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # orca --molden

    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    )
    Multiwfnpath = "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/"
    os.environ["Multiwfnpath"] = Multiwfnpath

    parse_only = False
    separate = True
    overwrite = True
    clean = True
    debug = False
    # create logger in target folder

    print(f"Running test case {test_case}...")

    def run_case(test_case, threads):
        """
        Runs gbw_analysis for the specified test_case (0-16).
        Args:
            test_case (int): Which test case to run (0-16).
            threads (int): Number of threads to use.
        """
        # List of folder paths for test cases 0 to 16
        base_folders = [
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_1/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_2/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_3/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_4/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_5/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_6/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_7/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_8/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_9/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_10/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_11/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_12/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_13/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_14/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_15/",
            "/home/santiagovargas/dev/qtaim_generator/data/omol_profiling/concur/base_test_16/",
            # Add more if needed
        ]

        orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"
        multiwfn_cmd = (
            "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
        )
        Multiwfnpath = "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/"
        os.environ["Multiwfnpath"] = Multiwfnpath

        parse_only = False
        separate = True
        overwrite = True
        clean = True
        debug = False

        if test_case < 0 or test_case > 16:
            print("Invalid test case number. Please choose between 0 and 16.")
            return

        folder = base_folders[test_case]
        try:
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
                logger=None,
                mem=400000000,
                nthreads=threads,
            )
        except Exception as e:
            print(f"Error in gbw_analysis - case {test_case}: {e}")

    run_case(test_case, threads)


main()
