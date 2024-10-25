import os, random, threading, subprocess, argparse
from glob import glob
from qtaim_gen.source.core.controller import (
    controller_single,
    ThreadWithResult,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-redo_qtaim", "--redo_qtaim", action="store_true", help="redo QTAIM"
    )
    parser.add_argument(
        "-just_dft",
        "--just_dft",
        action="store_true",
        help="only perform DFT calculaton",
    )
    parser.add_argument(
        "--reactions", action="store_true", help="expect reactants and products"
    )
    parser.add_argument(
        "-dir_active",
        type=str,
        default="./",
        help="absolute path to active directory, such as '/p/work1/wgee/TMC_QTAIM/QTAIM/'",
    )
    parser.add_argument(
        "-orca_path",
        type=str,
        default="./",
        help="absolute path to orca executable, such as '/p/home/wgee/Software/orca_5_0_4/orca'",
    )
    parser.add_argument(
        "-num_threads",
        type=int,
        default=1,
        help="number of threads, preferably divides the number of folders to crawl",
    )
    parser.add_argument(
        "-folders_to_crawl", help="number of folders to check", type=int, default=20000
    )

    args = parser.parse_args()
    redo_qtaim = args.redo_qtaim
    just_dft = args.just_dft
    reactions = args.reactions
    dir_active = args.dir_active
    orca_path = args.orca_path
    num_threads = int(args.num_threads)
    folders_to_crawl = round(int(args.folders_to_crawl) / num_threads)

    print("active dir: {}".format(dir_active))
    # folders_to_crawl_len = len(folders_to_crawl)
    folders_in_dir = os.listdir(dir_active)
    for _ in range(folders_to_crawl):
        threads = {}
        for n in range(num_threads):
            choice_sub = random.choice(folders_in_dir)
            folder_choice = dir_active + "/" + choice_sub
            # remove folder from list
            folders_in_dir.remove(choice_sub)
            thread_n = f"t{n}"
            threads[thread_n] = ThreadWithResult(
                target=controller_single,
                kwargs={
                    "folder_choice": folder_choice,
                    "redo_qtaim": redo_qtaim,
                    "just_dft": just_dft,
                    "reaction": reactions,
                    "orca_path": orca_path,
                },
            )

        for key, thread in threads.items():
            thread.start()

        for key, thread in threads.items():
            thread.join()


main()
