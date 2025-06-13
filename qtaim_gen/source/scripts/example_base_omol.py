from qtaim_gen.source.core.omol import gbw_analysis
import os
import logging



def main():
    # Only needed for orca5
    #os.system("export LD_LIBRARY_PATH=/home/santiagovargas/dev/orca5/:$LD_LIBRARY_PATH")
    
    # set environment variables
    os.environ["OMP_STACKSIZE"] = "64000000" # can also be set in settings.ini
    # set mem
    os.system("ulimit -s unlimited") # this sometimes doesn't work and I need to manually set this in cmdline
    os.system("export Multiwfnpath=/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI")  # change this
    
    orca6_2mkl = "/home/santiagovargas/orca_6_0_0/orca_2mkl"  # change this
    multiwfn_cmd = (
        "/home/santiagovargas/dev/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI"
    ) # change this
    orca_base = "/home/santiagovargas/dev/qtaim_generator/data/omol_tests/base_test/"  # change this


    try:
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
            parse_only=False,
            separate=True,
            overwrite=True,
            orca_6=True,
            clean=False,
            restart=False,
            debug=False,
            logger=logging.getLogger("gbw_analysis"),
        )  # works!
    except:
        print("Error in gbw_analysis - case 0")



main()

