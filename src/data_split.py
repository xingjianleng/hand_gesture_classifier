from pathlib import Path
from random import shuffle
import shutil


# split all the labelled data
data_path = "../labelled_data"
all_files = [file_name for file_name in Path(data_path).rglob("*.csv")]
shuffle(all_files)
train_size = int(0.8 * len(all_files))  # 80% of data in training set
train_file_names, test_file_names = all_files[:train_size], all_files[train_size:]

train_path = Path("../train_data")
test_path = Path("../test_data")

# cleanup the directory
for filename in train_path.rglob("*.txt"):
    filename.unlink()
for filename in test_path.rglob("*.txt"):
    filename.unlink()

# copy data into folders
for train_file in train_file_names:
    shutil.copy(train_file, f"{train_path}/{train_file.name}")

for test_file in test_file_names:
    shutil.copy(test_file, f"{test_path}/{test_file.name}")
