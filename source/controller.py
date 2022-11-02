import os, random, threading, subprocess
from glob import glob

folders = [name for name in os.listdir("./") if os.path.isdir(os.path.join("./", name))]


class ThreadWithResult(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def controller_single(dir_active):

    # get random folder from dir_active
    for i in range(1000):
        folders = [
            name
            for name in os.listdir(dir_active)
            if os.path.isdir(os.path.join(dir_active, name))
        ]

        # while(cond):
        folder_choice = random.choice(folders)
        print(folder_choice)
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
        # cond_qtaim = os.path.exists(folder_choice + "/out")

        subprocess.run(dir_active + folder_choice + "/reactants" + "/props.sh")
        os.system("mv " + "./CPprop.txt " + dir_active + folder_choice + "/reactants/")

        subprocess.run(dir_active + folder_choice + "/products" + "/props.sh")
        os.system("mv " + "./CPprop.txt " + dir_active + folder_choice + "/products/")
        # move CPprop.txt to folder


def main():

    dir_active = "../data/hydro/QTAIM/"
    for i in range(1000):
        counter = 0
        t1 = ThreadWithResult(
            target=controller_single, kwargs={"dir_active": dir_active}
        )
        t2 = ThreadWithResult(
            target=controller_single, kwargs={"dir_active": dir_active}
        )
        t3 = ThreadWithResult(
            target=controller_single, kwargs={"dir_active": dir_active}
        )
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()


main()
