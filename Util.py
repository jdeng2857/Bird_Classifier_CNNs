import shutil
import pandas as pd

def all_class_images():
    csv_df = pd.read_csv("birds.csv")
    dataset = pd.DataFrame(csv_df).to_numpy()
    filepaths = dataset[:, 1]
    class_ids = dataset[:, 0]
    train_ids = class_ids[:81950]
    train_filepaths = filepaths[:81950]
    id_to_filepaths = dict(zip(train_ids, train_filepaths))
    print(id_to_filepaths)

    new_filepaths = [];
    old_filepaths = [];
    for i, filepath in id_to_filepaths.items():
        new_filepath = "classes/" + str(int(i)) + ".jpg"
        new_filepaths.append(new_filepath)
        old_filepaths.append(filepath)
        shutil.copyfile(filepath,new_filepath)


CLASS_IMAGES = all_class_images()

# # Get the source file path
# source_file_path = "/path/to/source/file.txt"
#
# # Get the destination file path
# destination_file_path = "/path/to/destination/file.txt"
#
# # Copy the file from source to destination
# shutil.copyfile(source_file_path, destination_file_path)
