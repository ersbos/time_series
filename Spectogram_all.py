#%%
import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms as audio
import plotly.express as px

# Set the base directory containing CSV files.
base_directory = 'D:/Bitirme Projesi/main/new simulation(data ambient person)/time_series/data/train'

# Create a recursive glob pattern to search all CSV files in the directory and its subdirectories.
pattern = os.path.join(base_directory, '**', '*.csv')
csv_files = glob.glob(pattern, recursive=True)

# Instantiate the transforms once to reuse for all files.
spectrogram_transform = audio.Spectrogram(n_fft=4096, win_length=200, hop_length=128)
mel_transform = audio.MelSpectrogram(sample_rate=1000, n_fft=4096, win_length=200, hop_length=128, f_min=0, f_max=200, n_mels=100)

print(f"Found {len(csv_files)} CSV files.")

# Loop over each CSV file found.
for csv_file in csv_files:
    try:
        # Read the CSV file into a DataFrame.
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading file {csv_file}: {e}")
        continue

    # Check for expected columns.
    if 'voltage' not in df.columns:
        print(f"Skipping {csv_file}: 'voltage' column not found.")
        continue

    print(f"Processing file: {csv_file}")
    voltage = df['voltage'].values

    # Convert the voltage data into a Torch tensor with a shape of (1, time).
    signal = torch.tensor(voltage).float().unsqueeze(0)

    # Compute both spectrogram and mel spectrogram.
    spec = spectrogram_transform(signal)
    mel_spec = mel_transform(signal)

    # Remove the channel dimension and convert to a NumPy array.
    spec_db = 10 * np.log10(spec.squeeze(0).numpy() + 1e-10)
    mel_spec_db = 10 * np.log10(mel_spec.squeeze(0).numpy() + 1e-10)

    # Plot the mel spectrogram using Matplotlib.
    plt.figure(figsize=(10, 6))
    plt.imshow(spec_db, cmap='jet', origin='lower', aspect='auto')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title(f'Mel Spectrogram for {os.path.basename(csv_file)}')
    plt.colorbar(label='Power (dB)')
    plt.show()

    # If the CSV contains a "timestamp" column, add the row number and create a Plotly line plot.
    if 'timestamp' in df.columns:
        df['row_number'] = df.index + 1
        fig = px.line(
            df,
            x='timestamp',
            y='voltage',
            hover_data=['row_number'],
            title=f'Voltage Over Time ({os.path.basename(csv_file)})'
        )
        fig.update_xaxes(title_text='Timestamp (s)')
        fig.update_yaxes(title_text='Voltage')
        fig.show()
    else:
        print(f"Skipping Plotly plot for {csv_file}: 'timestamp' column not found.")
# %%
