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


def controller_single(dir_active, redo_qtaim=False, just_dft=False, reaction=True):
    """
    Runs, with the option of parallelization, the DFT and QTAIM calculations for a single random folder in the active directory.

    """

    folders = [
        name
        for name in os.listdir(dir_active)
        if os.path.isdir(os.path.join(dir_active, name))
    ]
    # while(cond):
    folder_choice = random.choice(folders)
    # check if folder has *wfn file
    print("dir_active {}".format(folder_choice))

    if reaction:
        if len(glob(dir_active + folder_choice + "/reactants" + "/*.wfn")) > 0:
            print("dft calc already done - reactants")

        else:
            os.system(
                "orca "
                + dir_active
                + folder_choice
                + "/reactants"
                + "/input.in > "
                + dir_active
                + folder_choice
                + "/reactants/output.out"
            )
        if len(glob(dir_active + folder_choice + "/products" + "/*.wfn")) > 0:
            print("dft calc already done - products")
        else:
            os.system(
                "orca "
                + dir_active
                + folder_choice
                + "/products"
                + "/input.in > "
                + dir_active
                + folder_choice
                + "/products/output.out"
            )

        if not just_dft:
            if (
                len(
                    glob(dir_active + folder_choice + "/reactants/*.CPprop.txt")
                    and not redo_qtaim
                )
                > 0
            ):
                print("cp calc already done - reactants")

            else:
                subprocess.run(dir_active + folder_choice + "/reactants/props.sh")
                os.system(
                    "mv " + "./CPprop.txt " + dir_active + folder_choice + "/reactants/"
                )

            if (
                len(
                    glob(dir_active + folder_choice + "/products/*.CPprop.txt")
                    and not redo_qtaim
                )
                > 0
            ):
                print("cp calc already done - products")

            else:
                subprocess.run(dir_active + folder_choice + "/products/props.sh")
                os.system(
                    "mv " + "./CPprop.txt " + dir_active + folder_choice + "/products/"
                )

    else:
        if len(glob(dir_active + folder_choice + "/*.wfn")) > 0:
            print("dft calc already done ")

        else:
            os.system(
                "orca "
                + dir_active
                + folder_choice
                + "/input.in > "
                + dir_active
                + folder_choice
                + "/output.out"
            )

        if not just_dft:
            if (
                len(
                    glob(dir_active + folder_choice + "/*.CPprop.txt")
                    and not redo_qtaim
                )
                > 0
            ):
                print("cp calc already done - reactants")

            else:
                subprocess.run(dir_active + folder_choice + "/props.sh")
                os.system("mv ./CPprop.txt " + dir_active + folder_choice)
