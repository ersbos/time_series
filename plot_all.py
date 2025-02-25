#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

def plot_csv_files_in_directory(base_directory):
    # Walk through all folders and files in the provided directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_file = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_file)

                    # Check if required columns are present
                    if 'timestamp' not in df.columns or 'voltage' not in df.columns:
                        print(f"Skipping '{csv_file}': Required columns not found.")
                        continue

                    # Optional: Convert voltage values to a torch tensor
                    voltage = df['voltage'].values
                    signal = torch.tensor(voltage).float().unsqueeze(0)

                    # Add a row_number column (starting from 1)
                    df['row_number'] = df.index + 1

                    # Determine title elements from the folder and file name
                    folder_name = os.path.basename(root)

                    # Create a new figure for each CSV file
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['timestamp'], df['voltage'], label="Voltage", color='blue')
                    plt.xlabel("Timestamp (s)")
                    plt.ylabel("Voltage")
                    plt.title(f"Folder: {folder_name}  —  File: {file}")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    # Display the plot; close the window to proceed to the next plot
                    plt.show()

                except Exception as e:
                    print(f"Error processing '{csv_file}': {e}")

if __name__ == "__main__":
    base_dir = "data/train"
    if not os.path.isdir(base_dir):
        print("The provided path is not a valid directory.")
    else:
        plot_csv_files_in_directory(base_dir)
# %%
