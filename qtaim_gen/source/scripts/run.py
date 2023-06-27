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
    parser.add_argument("-reactions", type=bool, default=True)
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

    for _ in range(folders_to_crawl):
        t1 = ThreadWithResult(
            target=controller_single,
            kwargs={
                "dir_active": dir_active,
                "redo_qtaim": redo_qtaim,
                "just_dft": just_dft,
                "reaction": reactions,
            },
        )
        t1.start()
        t1.join()


main()
