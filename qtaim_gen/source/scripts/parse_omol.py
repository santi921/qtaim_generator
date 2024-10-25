import os, json, argparse

from qtaim_gen.source.core.parse_multiwfn import (
    parse_charge_doc,
    parse_bond_order_doc,
    parse_fuzzy_doc,
    parse_other_doc,
)


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

    # reintegrate qtaim parser here
    routine_list = ["fuzzy", "bond", "charge", "other"]

    for folder in os.listdir(root_folder):
        folder_full_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_full_path):
            print(folder_full_path)
            for file in os.listdir(folder_full_path):
                if file.endswith(".out"):
                    file_full_path = os.path.join(folder_full_path, file)
                    for routine in routine_list:
                        print(
                            routine,
                        )
                        if routine in file:

                            json_file = file_full_path.replace(".out", ".json")

                            if routine == "fuzzy":
                                data = parse_fuzzy_doc(file_full_path)
                                with open(json_file, "w") as f:
                                    json.dump(data, f)

                            elif routine == "bond":
                                data = parse_bond_order_doc(file_full_path)

                                with open(json_file, "w") as f:
                                    json.dump(data, f)

                            elif routine == "other":
                                data = parse_other_doc(file_full_path)
                                with open(json_file, "w") as f:
                                    json.dump(data, f)

                            elif routine == "charge":
                                (
                                    charge_dict_overall,
                                    atomic_dipole_dict_overall,
                                    dipole_info,
                                ) = parse_charge_doc(file_full_path)
                                charge_dict_overall = {
                                    "charge": charge_dict_overall,
                                    "dipole": dipole_info,
                                    "atomic_dipole": atomic_dipole_dict_overall,
                                }
                                with open(json_file, "w") as f:
                                    json.dump(charge_dict_overall, f)


main()
