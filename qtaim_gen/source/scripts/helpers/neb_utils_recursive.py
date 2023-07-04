import os

# Define the directory where you want to start running the command
starting_directory = "/home/santiagovargas/Downloads/gas_hase/H3/"

# Define the maximum depth for recursion
max_depth = 5


# Recursive function to run the command in each subdirectory
def run_nebbarrier(directory, depth):
    # Change to the subdirectory
    os.chdir(directory)
    print(directory)
    # Check if the "00" folder exists
    if os.path.exists("00"):
        # Change to the "00" folder
        # os.chdir("00")

        # Run the command in the subdirectory
        os.system("nebbarrier.pl")
        os.system("perl /home/santiagovargas/Downloads/gas_hase/nebef.pl > ef.out")
        # Change back to the parent directory
        os.chdir("..")

    # Recursively enter subdirectories up to the maximum depth
    if depth < max_depth:
        for root, dirs, files in os.walk(directory):
            # print(root, dirs, files)
            for subdir in dirs:
                subdirectory = os.path.join(root, subdir)
                run_nebbarrier(subdirectory, depth + 1)

    # Change back to the parent directory
    os.chdir("..")


# Start running the command from the starting directory
run_nebbarrier(starting_directory, 1)
