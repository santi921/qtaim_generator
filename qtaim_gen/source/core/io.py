def write_input_file(folder, lines, n_atoms, options):
    """
    Write input file for Multiwfn.
    Takes:
        folder: folder to write input file to
        lines: lines from xyz file
        n_atoms: number of atoms in xyz file
        options: dictionary of options for input file
    Returns:
        None
    """
    with open(folder + "/input.in", "w") as f:
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        f.write("*xyz {} {}\n".format(options["charge"], options["spin"]))
        for ind in range(n_atoms):
            f.write(
                str(lines[ind + 2].split()[0])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[1])
                + "\t"
                + str(lines[ind + 2].split()[2])
                + "\n"
            )
        f.write("*\n")


def write_input_file_from_pmg_molecule(folder, molecule, options):
    n_atoms = int(len(molecule["sites"]))
    # print(options)
    with open(folder + "/input.in", "w") as f:
        f.write("!{} {} AIM\n\n".format(options["functional"], options["basis"]))
        f.write("%SCF\n")
        f.write("    MaxIter 1000\n")
        f.write("END\n")
        f.write(
            "* xyz {} {}\n".format(
                molecule["charge"],
                molecule["spin_multiplicity"],
            )
        )
        for ind in range(n_atoms):
            xyz = molecule["sites"][ind]["xyz"]
            atom = molecule["sites"][ind]["species"][0]["element"]
            f.write(
                "{}\t{: .4f}\t{: .4f}\t{: .4f}\n".format(atom, xyz[0], xyz[1], xyz[2])
            )
        f.write("*\n")


def complete_folder_molecule_to_json(folder):
    """
    Takes:
        folder: folder to complete
    Returns:
        row: row of pandas dataframe - contains all information molecule in folder
    """
    # TODO
