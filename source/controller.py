import os, random, threading, subprocess 
from glob import glob

folders = [ name for name in os.listdir("./") if os.path.isdir(os.path.join('./', name)) ]

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


def controller_single(dir_active):
    #folders = [ name for name in os.listdir("./") if os.path.isdir(os.path.join('./', name)) ]
    cond = True 

    while(cond):
        folder_choice = random.choice(folders)
        print(os.path.exists(folder_choice + "/input.wfn"))
        cond =  os.path.exists(folder_choice + "/input.wfn")


    print(folder_choice)    
    os.system("orca " + "./" + folder_choice +  "/input.in")
    #cond_qtaim = os.path.exists(folder_choice + "/out")
    subprocess.run("./"+folder_choice + '/props.sh')
    # move CPprop.txt to folder 
    os.system("mv " + "./CPprop.txt " + dir_active + folder_choice)


def main():

    dir_active = "../data/hydro/QTAIM/"
    for i in range(1000):
        counter = 0 
        t1 = ThreadWithResult(target=controller_single, kwargs={"dir_active" : dir_active})
        t2 = ThreadWithResult(target=controller_single, kwargs={"dir_active" : dir_active})
        t3 = ThreadWithResult(target=controller_single, kwargs={"dir_active" : dir_active})
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()

main()
