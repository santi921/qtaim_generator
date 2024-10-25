import os, argparse
from random import choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-root_folder",
        type=str,
        default="/home/santiagovargas/dev/qtaim_generator/tests/test_files/omol",
    )
    parser.add_argument("-folders_to_crawl", type=int, default=10)
    parser.add_argument("-overwrite", type=bool, default=False)

    args = parser.parse_args()
    root_folder = args.root_folder
    folders_to_crawl = args.folders_to_crawl
    overwrite = args.overwrite

    order_of_operations = ["qtaim", "fuzzy", "bond", "charge", "other"]
    options_folders = os.listdir(root_folder)

    count_folders = 0
    while True:
        folder = choice(options_folders)
        print(folder)
        count_folders += 1
        if count_folders > folders_to_crawl:
            break

        folder_full_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_full_path):
            # check that there is not .wfn file in the folder
            wfn_present = False
            for file in os.listdir(folder_full_path):
                if file.endswith(".wfn"):
                    wfn_present = True
                if file.endswith("convert.in"):
                    conv_file = os.path.join(folder_full_path, file)

            if not wfn_present:
                # run conversion script
                os.system("{}".format(conv_file))

            # run multiwfn scripts
            for order in order_of_operations:
                result_file = os.path.join(folder_full_path, "{}.out".format(order))
                # check that result file does not exist and isn't empty
                if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                    if not overwrite:
                        print(
                            "File {} already exists and is not empty".format(
                                result_file
                            )
                        )
                        continue

                mfwn_file = os.path.join(
                    folder_full_path, "props_{}.mfwn".format(order)
                )
                os.system("{}".format(mfwn_file))


main()
