import os, bson, argparse
from glob import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="../data/rapter/QTAIM/",
        help="",
    )

    args = parser.parse_args()
    root = args.root
    folders = glob(root + "/*")

    count = 0
    for ind in range(len(folders)):
        folder = folders[ind]
        files = glob(folder + "/*.wfn")
        for file in files:
            if os.path.exists(file[:-4] + ".res"):
                count += 1
            else:
                print(file)

    print("Wfns computed: {} / {}".format(count, len(folders)))


main()
