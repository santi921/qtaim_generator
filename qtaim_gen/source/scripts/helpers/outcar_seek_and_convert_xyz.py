import os
import shutil
import tarfile
import gzip
import glob
import numpy as np


def ef_out(folder):
    # check if folder has ef.out in the folder above this one
    ef_out = glob.glob(os.path.join(folder + "/../", "ef.out"))
    # ef_out = glob.glob(os.path.join(folder + "/", "ef.out"))

    if len(ef_out) == 0:
        return False
    else:
        # read ef.out, it's a 4 column text file
        # first column is iterations, second is forces, third is energy, fourth is delta energy
        # return the max force
        ef_out = ef_out[0]
        with open(ef_out, "r") as f:
            lines = f.readlines()
            forces = [float(line.split()[1]) for line in lines]
            # print(forces)
            return max(forces)


def copy_and_rename_ef_and_neb_dat(folder, target_dir):
    ef_out = glob.glob(os.path.join(folder + "/../", "ef.out"))
    neb_dat = glob.glob(os.path.join(folder + "/../", "neb.dat"))

    if len(ef_out) > 0:
        print("found ef.out")
        modified_name = ef_out[0]
        modified_name_neb = neb_dat[0]
        replace_list = [
            "/00/",
            "/01/",
            "/02/",
            "/03/",
            "/04/",
            "/05/",
            "/06/",
            "/07/",
            "/08/",
            "/09/",
            "/10/",
            "/11/",
            "/12/",
            "/13/",
            "/14/",
            "/15/",
        ]
        for i in replace_list:
            modified_name = modified_name.replace(i, "/")
            modified_name_neb = modified_name_neb.replace(i, "/")
        modified_name = modified_name.replace("/", "_")
        modified_name_neb = modified_name_neb.replace("/", "_")
        # remove /00/ or /01/ ...
        # remove . from modified_name
        modified_name = modified_name.replace("..", "")
        modified_name_neb = modified_name_neb.replace("..", "")

        # copy ef.out to target_dir with modified name
        shutil.copy(ef_out[0], target_dir + "/" + modified_name[2:])
        shutil.copy(neb_dat[0], target_dir + "/" + modified_name_neb[2:])


def convert_poscar_to_xyz(poscar_file, output_directory):
    # Get the filename without the directory path
    # file_name = os.path.basename(poscar_file)
    file_name = os.path.basename(poscar_file)
    # Modify the file name to replace '/' with '_' and remove '.'
    modified_name = poscar_file.replace("/", "_").replace(".", "")[2:]

    # Construct the output file name by replacing the extension and modifying the filename
    output_file = os.path.splitext(modified_name)[0] + ".xyz"

    # Read the POSCAR file and convert it to XYZ format
    with open(poscar_file, "r") as in_file:
        lines = in_file.readlines()
        direct_tf = False
        lattice_info = lines[2:5]
        lattice_info = [line.split() for line in lattice_info]
        lattice_info = [[float(i) for i in line] for line in lattice_info]
        # print(lattice_info)
        for i in range(len(lines)):
            if lines[i].strip().startswith("Direct"):
                direct_tf = True

    with open(poscar_file, "r") as in_file, open(output_file, "w") as out_file:
        # Skip the header lines
        if direct_tf:
            for _ in range(5):
                next(in_file)
        else:
            for _ in range(4):
                next(in_file)
            # todo

        # Read the atomic symbols
        atomic_elem = next(in_file).split()
        atomic_counts = next(in_file).split()
        atomic_counts = [int(i) for i in atomic_counts]
        cart_direct = next(in_file)
        # print(atomic_elem, atomic_counts)
        atomic_symbols = [
            element
            for element, repeat in zip(atomic_elem, atomic_counts)
            for _ in range(repeat)
        ]
        # print(atomic_symbols)
        # Read the atomic positions in direct coordinates
        direct_coordinates = []
        for line in in_file:
            # print(line)
            if not line.strip():
                break

            coordinates = line.split()[:3]
            coord_temp = np.array([float(coord) for coord in coordinates])
            if direct_tf:  # dot product
                coord_temp = np.dot(coord_temp, lattice_info)

            direct_coordinates.append(coord_temp)
            # direct_coordinates.append()
            # if you get to an empty line, break out of the loop

        out_file.write("{}\n".format(np.sum(atomic_counts)))
        # Write the XYZ file header
        out_file.write("Converted from POSCAR file: {}\n".format(poscar_file))

        # Write the atomic positions in XYZ format
        for symbol, coordinates in zip(atomic_symbols, direct_coordinates):
            cartesian_coordinates = [
                coord for coord in coordinates
            ]  # Convert to Cartesian coordinates (Angstroms)
            out_file.write(
                "{} {:.4f} {:.4f} {:.4f}\n".format(symbol, *cartesian_coordinates)
            )

    # Move the XYZ file to the output directory
    shutil.move(output_file, os.path.join(output_directory, output_file))


def convert_outcar_to_xyz(outcar_file, output_directory):
    # Get the filename without the directory path
    file_name = os.path.basename(outcar_file)

    # Modify the file name to replace '/' with '_' and remove '.'
    modified_name = outcar_file[2:].replace("/", "_").replace(".", "")
    # print(modified_name)
    # Construct the output file name by replacing the extension and modifying the filename
    output_file = os.path.splitext(modified_name)[0] + ".xyz"

    # Read the OUTCAR file and convert it to XYZ format
    with open(outcar_file, "r") as in_file, open(output_file, "w") as out_file:
        # Skip the initial lines until the atomic positions section
        for line in in_file:
            # print(line)
            if "POSITION" in line:
                break

        # Write the XYZ file header
        out_file.write("Converted from OUTCAR file: {}\n".format(outcar_file))

        # Write the atomic positions in XYZ format
        for line in in_file:
            if line.strip().startswith("-"):
                break
            values = line.split()
            atom_type = values[0]
            coordinates = [float(coord) for coord in values[1:4]]
            out_file.write("{} {:.4f} {:.4f} {:.4f}\n".format(atom_type, *coordinates))

    # Move the XYZ file to the output directory
    shutil.move(output_file, os.path.join(output_directory, output_file))


def untar_file(tar_path):
    print(tar_path[:-3])
    with gzip.open(tar_path, "rb") as f_in:
        with open(tar_path[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main(argv=None):
    # Set the folder path to traverse
    folder_path = "./H0"

    # Set the output directory where the converted XYZ files will be placed
    output_directory = "./xyz_" + folder_path[2:]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Find all OUTCAR.gz files in the folder structure using glob
    outcar_gz_files = glob.glob(
        os.path.join(folder_path, "**", "*OUTCAR.gz"), recursive=True
    )
    # print(outcar_gz_files)
    # Convert the OUTCAR.gz files
    for gz_file in outcar_gz_files:
        untar_file(gz_file)

    # Find all remaining OUTCAR files in the folder structure using glob
    outcar_files = glob.glob(os.path.join(folder_path, "**", "*OUTCAR"), recursive=True)
    contcar_files = glob.glob(
        os.path.join(folder_path, "**", "*CONTCAR"), recursive=True
    )
    # print(outcar_files)
    # Convert the remaining OUTCAR files
    """for outcar_file in outcar_files:
        # if there's a CONTCAR file, use that instead
        if outcar_file.replace("OUTCAR", "CONTCAR") in contcar_files:
            convert_poscar_to_xyz(
                outcar_file.replace("OUTCAR", "CONTCAR"), output_directory
            )
        else:
            convert_outcar_to_xyz(outcar_file, output_directory)
    """
    outcar_files = glob.glob(os.path.join(folder_path, "**", "*POSCAR"), recursive=True)
    # print(outcar_files)
    for outcar_file in outcar_files:
        # get the folder path of the file
        folder_path = os.path.dirname(outcar_file)
        # print(outcar_file)
        # print(folder_path)
        ef_out_out = ef_out(folder_path)
        if ef_out_out == False:
            pass
        elif ef_out_out > 0.04:
            print("skipping ef file error too high, max error: ", ef_out_out)
        else:
            copy_and_rename_ef_and_neb_dat(folder_path, output_directory)

            if outcar_file.replace("OUTCAR", "CONTCAR") in contcar_files:
                print("found cont car")
                convert_poscar_to_xyz(
                    outcar_file.replace("OUTCAR", "CONTCAR"), output_directory
                )
                # convert_poscar_to_xyz(outcar_file, output_directory)
            else:
                convert_poscar_to_xyz(outcar_file, output_directory)


if __name__ == "__main__":
    raise SystemExit(main())
