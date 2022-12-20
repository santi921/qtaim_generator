import os, random, threading, subprocess, argparse
from glob import glob

folders = [name for name in os.listdir("./") if os.path.isdir(os.path.join("./", name))]


class ThreadWithResult(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


def controller_single(dir_active, redo_qtaim=False, just_dft = False):

    # get random folder from dir_active
    #for i in range(1000):
    folders = [
        name
        for name in os.listdir(dir_active)
        if os.path.isdir(os.path.join(dir_active, name))
    ]

    
    # while(cond):
    folder_choice = random.choice(folders)
    # check if folder has *wfn file
    
    print(folder_choice)
    
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
        if len(glob(dir_active + folder_choice + "/reactants/*.CPprop.txt") and not redo_qtaim) > 0:
            print("cp calc already done - reactants")
            
        else: 
            subprocess.run(dir_active + folder_choice + "/reactants/props.sh")
            os.system("mv " + "./CPprop.txt " + dir_active + folder_choice + "/reactants/")
        
        if len(glob(dir_active + folder_choice + "/products/*.CPprop.txt") and not redo_qtaim) > 0:
            print("cp calc already done - products")
            
        else: 
            subprocess.run(dir_active + folder_choice + "/products/props.sh")
            os.system("mv " + "./CPprop.txt " + dir_active + folder_choice + "/products/")
        # move CPprop.txt to folder


def main():
    
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('-hydro', '--hydro', action='store_true')
    parser.add_argument('-redo_qtaim', '--redo_qtaim', action='store_true')
    parser.add_argument('-just_dft', '--just_dft', action='store_true')
    args = parser.parse_args()

    if args.hydro:
        dir_active = "../data/hydro/QTAIM/"
    else:
        dir_active = "../data/mg2/QTAIM/"
    redo_qtaim = args.redo_qtaim
    just_dft = args.just_dft
    
    for i in range(10000):
        counter = 0
        t1 = ThreadWithResult(
            target=controller_single, kwargs={"dir_active": dir_active, "redo_qtaim": redo_qtaim, "just_dft": just_dft}
        )
        #t2 = ThreadWithResult(
        #    target=controller_single, kwargs={"dir_active": dir_active, "redo_qtaim": redo_qtaim}
        #)
        #t3 = ThreadWithResult(
        #    target=controller_single, kwargs={"dir_active": dir_active, "redo_qtaim": redo_qtaim}
        #)
        t1.start()
        #t2.start()
        #t3.start()
        t1.join()
        #t2.join()
        #t3.join()


main()
