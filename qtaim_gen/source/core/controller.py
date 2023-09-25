import os, random, threading, subprocess, argparse
from glob import glob

# folders = [name for name in os.listdir("./") if os.path.isdir(os.path.join("./", name))]


class ThreadWithResult(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def controller_single(folder_choice, redo_qtaim=False, just_dft=False, reaction=True):
    """
    Runs, with the option of parallelization, the DFT and QTAIM calculations for a single random folder in the active directory.
    Takes:
        folder_choice: the folder to run the calculations in
        redo_qtaim: whether to redo the QTAIM calculations
        just_dft: whether to just do the DFT calculations
        reaction: whether to do a reaction or not

    """

    # check if folder has *wfn file
    print("dir_active {}".format(folder_choice))

    if reaction:
        if len(glob(folder_choice + "/reactants" + "/*.wfn")) > 0:
            print("dft calc already done - reactants")

        else:
            os.system(
                "orca "
                + folder_choice
                + "/reactants"
                + "/input.in > "
                + folder_choice
                + "/reactants/output.out"
            )
        if len(glob(folder_choice + "/products" + "/*.wfn")) > 0:
            print("dft calc already done - products")
        else:
            os.system(
                "orca "
                + folder_choice
                + "/products"
                + "/input.in > "
                + folder_choice
                + "/products/output.out"
            )

        if not just_dft:
            if (
                len(glob(folder_choice + "/reactants/CPprop.txt")) > 0 and not redo_qtaim
            ):
                print("cp calc already done - reactants")

            else:
                subprocess.run(folder_choice + "/reactants/props.sh")
                os.system("mv " + "./CPprop.txt " + folder_choice + "/reactants/")

            if (
                len(glob(folder_choice + "/products/CPprop.txt")) > 0 and not redo_qtaim
            ):
                print("cp calc already done - products")

            else:
                subprocess.run(folder_choice + "/products/props.sh")
                os.system("mv " + "./CPprop.txt " + folder_choice + "/products/")

    else:
        if len(glob(folder_choice + "/*.wfn")) > 0:
            print("dft calc already done ")

        else:
            os.system(
                "orca " + folder_choice + "/input.in > " + folder_choice + "/output.out"
            )

        if not just_dft:
            if len(glob(folder_choice + "/CPprop.txt")) > 0 and not redo_qtaim:
                print("cp calc already done")

            else:
                subprocess.run(folder_choice + "/props.sh")
                os.system("mv ./CPprop.txt " + folder_choice)
