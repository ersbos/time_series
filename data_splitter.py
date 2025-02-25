import os
import re
import random
import shutil

# Define the source base directory containing the person folders
# (e.g. folders like "person_eight", "person_one", etc.)
base_dir = "data/train"  # You can change this to your source directory path

# Define the target directory where files will be moved. It will contain folders like "person_eight".
target_base_dir = os.path.join(base_dir, "target")
if not os.path.exists(target_base_dir):
    os.makedirs(target_base_dir)

# Define a mapping from numeric strings to their word counterparts.
# Extend this dictionary as needed.
num_to_word = {
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
    '10': 'ten'
}

# List all directories in the base directory that start with "person"
person_dirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("person")
]

# Compile a regex pattern to extract the sensor identifier.
# For example, from "person_8_sensor_1_speed_1_amplification_1_trace_1_with_headers.csv"
# it will extract "1" as the sensor id.
pattern_sensor = re.compile(r'_sensor_(\d+)_')

# Compile a regex pattern to extract the person number from the file name.
# It expects that the file name starts with "person_<number>".
pattern_person = re.compile(r'^person_(\d+)')

# Process each person folder found in the source base directory.
for person_dir in person_dirs:
    source_person_path = os.path.join(base_dir, person_dir)
    print(f"\nProcessing folder: {person_dir}")

    # Group files by sensor id.
    sensor_files = {}
    for filename in os.listdir(source_person_path):
        if filename.endswith(".csv"):
            match_sensor = pattern_sensor.search(filename)
            if match_sensor:
                sensor_id = match_sensor.group(1)
                sensor_files.setdefault(sensor_id, []).append(os.path.join(source_person_path, filename))
            else:
                print(f"File '{filename}' does not match the expected sensor pattern and will be skipped.")

    # For each sensor group, randomly move 20% of the files.
    for sensor_id, files in sensor_files.items():
        num_files = len(files)
        num_to_move = int(num_files * 0.2)
        if num_to_move == 0:
            print(f"For sensor_{sensor_id} in {person_dir}: {num_files} file(s) detected. Nothing to move (20% is 0 files).")
        else:
            files_to_move = random.sample(files, num_to_move)
            for file_path in files_to_move:
                filename = os.path.basename(file_path)
                # Extract the person number from the filename.
                match_person = pattern_person.match(filename)
                if match_person:
                    person_num = match_person.group(1)
                    # Map the numeric person id to its text form.
                    person_word = num_to_word.get(person_num, person_num)  # fallback to the number if no mapping exists
                    target_person_folder = f"person_{person_word}"
                    destination_folder = os.path.join(target_base_dir, target_person_folder)
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    destination = os.path.join(destination_folder, filename)
                    try:
                        shutil.move(file_path, destination)
                        print(f"Moved file:\n  {file_path}\n  -> {destination}")
                    except Exception as e:
                        print(f"Error moving {file_path}: {e}")
                else:
                    print(f"File '{filename}' does not match the expected person pattern; cannot determine target folder.")