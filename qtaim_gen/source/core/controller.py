import os, threading, subprocess, shlex
from glob import glob
from pathlib import Path

# folders = [name for name in os.listdir("./") if os.path.isdir(os.path.join("./", name))]


class ThreadWithResult(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def controller_single(
    folder_choice, redo_qtaim=False, just_dft=False, reaction=True, orca_path=""
):
    """
    Runs, with the option of parallelization, the DFT and QTAIM calculations for a single random folder in the active directory.
    Takes:
        folder_choice: the folder to run the calculations in
        redo_qtaim: whether to redo the QTAIM calculations
        just_dft: whether to just do the DFT calculations
        reaction: whether to do a reaction or not
        orca_path: absolute path to orca (required for parallel calculations)

    """

    # check if folder has *wfn file
    print("dir_active {}".format(folder_choice))

    if orca_path:
        orca = orca_path
    else:
        orca = "orca"

    if reaction:
        if len(glob(folder_choice + "/reactants" + "/*.wfn")) > 0:
            print("dft calc already done - reactants")

        else:
            os.system(
                "{} ".format(orca)
                + shlex.quote(folder_choice + "/reactants/input.in")
                + " > "
                + shlex.quote(folder_choice + "/reactants/output.out")
            )
        if len(glob(folder_choice + "/products" + "/*.wfn")) > 0:
            print("dft calc already done - products")
        else:
            os.system(
                "{} ".format(orca)
                + shlex.quote(folder_choice + "/products/input.in")
                + " > "
                + shlex.quote(folder_choice + "/products/output.out")
            )

        if not just_dft:
            if (
                len(glob(folder_choice + "/reactants/CPprop.txt")) > 0
                and not redo_qtaim
            ):
                print("cp calc already done - reactants")

            else:
                cwd = os.getcwd()
                os.chdir(folder_choice + "/reactants")
                subprocess.run(["bash", folder_choice + "/reactants/props.sh"])
                os.chdir(cwd)

            if len(glob(folder_choice + "/products/CPprop.txt")) > 0 and not redo_qtaim:
                print("cp calc already done - products")

            else:
                cwd = os.getcwd()
                os.chdir(folder_choice + "/products")
                subprocess.run(["bash", folder_choice + "/products/props.sh"])
                os.chdir(cwd)

    else:
        if len(glob(folder_choice + "/*.wfn")) > 0:
            print("dft calc already done!")

        else:
            os.system(
                "{} ".format(orca)
                + shlex.quote(folder_choice + "/input.in")
                + " > "
                + shlex.quote(folder_choice + "/output.out")
            )

        if not just_dft:
            if len(glob(folder_choice + "/CPprop.txt")) > 0 and not redo_qtaim:
                print("cp calc already done")

            else:
                cwd = os.getcwd()
                os.chdir(folder_choice)
                subprocess.run(["bash", folder_choice + "/props.sh"])
                os.chdir(cwd)
