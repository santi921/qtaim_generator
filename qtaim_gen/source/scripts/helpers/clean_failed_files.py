#!/usr/bin/env python3

import os, argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--reaction", action="store_true")
    args = parser.parse_args(argv)
    reaction = bool(args.reaction)
    root = args.root

    # iterate through each line in the file
    with open("./fail_id.txt") as file:
        lines = file.readlines()

    for line in lines:
        print(line)
        try:
            if reaction:
                os.remove(root + line.strip() + "/reactants/" + "CPprop.txt")
                os.remove(root + line.strip() + "/products/" + "CPprop.txt")
                print("deleted!")
            else:
                os.remove(root + line.strip() + "/CPprop.txt")
                print("deleted!")
        except Exception:
            print("failed")


if __name__ == "__main__":
    raise SystemExit(main())
