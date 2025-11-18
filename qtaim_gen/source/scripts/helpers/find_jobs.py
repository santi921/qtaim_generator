
import os

def find_zst_folders(root_dir, output_file, alternative_root=None, max_depth=3):
    """
    Traverse up to max_depth levels from root_dir, saving paths containing 'orca.tar.zst'.
    """
    count = 0
    def traverse(current_dir, current_depth):
        nonlocal count
        if current_depth > max_depth:
            return
        try:
            with os.scandir(current_dir) as it:
                found_zst = False
                for entry in it:
                    if entry.is_file() and entry.name == 'orca.tar.zst':
                        found_zst = True
                    elif entry.is_dir():
                        traverse(entry.path, current_depth + 1)
                if found_zst:
                    if alternative_root:
                        relative_path = os.path.relpath(current_dir, root_dir)
                        new_path = os.path.join(alternative_root, relative_path)
                        out_file.write(new_path + '\n')
                    else:
                        out_file.write(current_dir + '\n')
                    count += 1
        except PermissionError:
            pass

    with open(output_file, 'w') as out_file:
        traverse(root_dir, 1)
    print(f"Paths saved to {output_file}, count: {count}")

# Example usage
root_directory = '/global/scratch/users/santiagovargas/omol_w_cleaning/'  # Replace with your target folder
output_file = 'out.txt'
alternative_root = None  # Replace with your alternative root if needed

find_zst_folders(root_directory, output_file, alternative_root, max_depth=3)