from qtaim_gen.source.core.omol import gbw_analysis
import os


def main():
    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000"
    # set mem
    os.system("ulimit -s unlimited")

    orca_base = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/base_test/"
    orca_6 = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca6/"
    orca_5 = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca5/"
    orca_6_rks = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca6_rks/"
    orca_6_uks = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca6_uks/"
    orca_5_rks = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca5_rks/"
    orca_5_uks = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/orca5_uks/"

    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # orca --molden
    orca5_2mkl = "/home/santiagovargas/dev/orca5/orca_2mkl"

    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    )

    parse_only = True
    separate = True
    overwrite = True
    clean = False
    debug = False
    """
    try:
        # works!
        gbw_analysis(
            folder=orca_6_rks,
            orca_2mkl_cmd=orca6_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=True,
            clean=clean,
            restart=False,
            debug=debug,
        )  # works!
    except:

        print("Error in gbw_analysis - case 2")
    """
    """
    try:
        # works!
        gbw_analysis(
            folder=orca_5_uks,
            orca_2mkl_cmd=orca5_2mkl,
            multiwfn_cmd=multiwfn_cmd,
            parse_only=parse_only,
            separate=separate,
            overwrite=overwrite,
            orca_6=False,
            clean=clean,
            restart=False,
            debug=debug
        )  # works!
    except:

        print("Error in gbw_analysis - case 6")
    
    try:
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
            debug=debug
        )  # works!
    except:

        print("Error in gbw_analysis - case 3")


    try:
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
            debug=debug
        )  # works!
    except:

        print("Error in gbw_analysis - case 5")
    """
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
            debug=debug
        )  # works!
    try:
        print(orca_base)
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
            debug=debug
        )  # works!
    except:
        print("Error in gbw_analysis - case 0")
    
    """
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
            debug=debug
        )  # works!
    except:

        print("Error in gbw_analysis - case 1")


    try:
        # works!
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
            debug=debug
        )  # works!
    except:

        print("Error in gbw_analysis - case 4")

    """


main()
