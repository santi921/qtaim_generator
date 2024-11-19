from qtaim_gen.source.core.omol import gbw_analysis
import os


def main():
    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000"
    # set mem
    os.system("ulimit -s unlimited")

    orca_6 = "/home/santiagovargas/dev/qtaim_generator/data/orca6/"
    orca_5 = "/home/santiagovargas/dev/qtaim_generator/data/orca5/"
    test_folder = "/home/santiagovargas/dev/qtaim_generator/tests/test_files/omol/3/"

    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # orca --molden
    orca5_2mkl = "/home/santiagovargas/dev/orca5/orca_2mkl"

    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    )

    parse_only = False
    separate = True
    overwrite = True
    clean = True

    try:
        # works!
        gbw_analysis(
            folder=orca_6,
            orca_2mkl_cmd=orca6_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=True,
            clean=clean,
            restart=False,
        )  # works!
    except:
        print("Error in gbw_analysis - case 1")

    try:
        gbw_analysis(
            folder="/home/santiagovargas/dev/qtaim_generator/data/orca_convert/",
            orca_2mkl_cmd=orca5_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=False,
            separate=True,
            overwrite=True,
            orca_6=False,
            clean=True,
            restart=False,
        )
    except:
        print("Error in gbw_analysis - case 3")

    try:
        gbw_analysis(
            folder=orca_5,
            orca_2mkl_cmd=orca5_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=False,
            clean=clean,
            restart=False,
        )
    except:
        print("Error in gbw_analysis - case 2")


main()
