import os

# Set your directory path here
directory = "/lab/projects/fire_smoke_awr/data/detection/datasets/data_mining/early_fire_A_only/images/test"

def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
 
if __name__ == "__main__":
    count = count_files_in_directory(directory)
    print(f"Number of files in '{directory}': {count}")
