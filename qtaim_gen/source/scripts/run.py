import os, random, threading, subprocess, argparse
from glob import glob
from qtaim_gen.source.core.controller import (
    controller_single,
    ThreadWithResult,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-redo_qtaim", "--redo_qtaim", action="store_true")
    parser.add_argument("-just_dft", "--just_dft", action="store_true")
    parser.add_argument("--reactions", action="store_true")
    parser.add_argument("-dir_active", type=str, default="./")
    parser.add_argument(
        "-folders_to_crawl", help="number of folders to check", type=int, default=20000
    )

    args = parser.parse_args()
    redo_qtaim = args.redo_qtaim
    just_dft = args.just_dft
    reactions = args.reactions
    dir_active = args.dir_active
    folders_to_crawl = int(args.folders_to_crawl)

    print("active dir: {}".format(dir_active))
    # folders_to_crawl_len = len(folders_to_crawl)
    folders_in_dir = os.listdir(dir_active)
    for _ in range(folders_to_crawl):
        choice_sub = random.choice(folders_in_dir)
        folder_choice = dir_active + "/" + choice_sub
        t1 = ThreadWithResult(
            target=controller_single,
            kwargs={
                "dir_active": dir_active,
                "folder_choice": folder_choice,
                "redo_qtaim": redo_qtaim,
                "just_dft": just_dft,
                "reaction": reactions,
            },
        )

        t1.start()
        t1.join()
        # remove folder from list
        folders_in_dir.remove(choice_sub)
        #
        # remove


main()
