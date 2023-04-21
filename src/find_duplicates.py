import os
import hashlib

def find_duplicates(folder_path, delete=False):
    """
    Find duplicate images in a folder
    :param folder_path: path to the folder containing the images
    :param delete: whether to delete one of the duplicate files (default: False)
    """
    # dictionary to store file hashes and paths
    file_hashes = {}
    duplicate_count = 0
    
    # iterate over files in folder
    for file_name in os.listdir(folder_path):
        # skip subdirectories
        if os.path.isdir(os.path.join(folder_path, file_name)):
            continue
            
        # calculate hash of file
        with open(os.path.join(folder_path, file_name), 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        # check if hash already exists in dictionary
        if file_hash in file_hashes:
            #print(f"Duplicate found: {file_name} and {file_hashes[file_hash]}")
            duplicate_count += 1
            
            # delete one of the duplicate files if delete=True
            if delete:
                input_path = os.path.join(folder_path, file_name)
                output_base_filename = os.path.splitext(os.path.basename(input_path))[0]
                output_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "singles")
                for suffix in ["_0.0", "_90.0", "_180.0", "_270.0"]:
                    output_filename = output_base_filename + suffix + ".jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    os.remove(output_path)
                os.remove(os.path.join(folder_path, file_name))
                print(f'Removing {os.path.join(folder_path, file_name)}... and its 4 rotations')

        else:
            # add hash and path to dictionary
            file_hashes[file_hash] = file_name
    
    # print duplicate count for folder
    print(f"Total duplicates found in {folder_path[7:-6]}: {duplicate_count}")