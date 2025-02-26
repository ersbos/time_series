#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms as audio
import plotly.express as px

csv_file = 'D:/Bitirme Projesi/main/new simulation(data ambient person)/time_series/data/person_eight/person_8_sensor_1_speed_1_amplification_1_trace_1_with_headers.csv'


df = pd.read_csv(csv_file)


voltage = df['voltage'].values


signal = torch.tensor(voltage).float().unsqueeze(0)  # shape becomes (1, time)

# Instantiate the spectrogram transform with specified parameters.
transform = audio.Spectrogram(n_fft=2048, win_length=500, hop_length=128)
mel_transform = audio.MelSpectrogram(1000,2048,200,128,0,500,128)
# Compute the spectrogram.
spectrogram = transform(signal)
mel_spectrogram = mel_transform(signal)

# Remove the channel dimension and convert to a NumPy array.
spectrogram = spectrogram.squeeze(0).numpy()
mel_spectrogram = mel_spectrogram.squeeze(0).numpy()
# Optional: Convert amplitude values to dB for better visualization.
spectrogram_db = 10 * np.log10(spectrogram + 1e-10)
mel_spectrogram_db = 10 * np.log10(mel_spectrogram + 1e-10)
# Plot the spectrogram using Matplotlib with the 'jet' colormap.
plt.figure(figsize=(10, 6))
plt.imshow(mel_spectrogram_db, cmap='jet', origin='lower', aspect='auto')
plt.xlabel('Time Frames')
plt.ylabel('Frequency Bins')
plt.title('Spectrogram')
plt.colorbar(label='Power (dB)')
plt.show()
# Add a column for row number (starting at 1)
df['row_number'] = df.index + 1

# Create a line plot using Plotly Express and include the row number in the hover data
fig = px.line(df, x='timestamp', y='voltage', hover_data=['row_number'],
              title='Voltage Over Time (Timestamp in Seconds)')

# Update axis titles if needed
fig.update_xaxes(title_text='Timestamp (s)')
fig.update_yaxes(title_text='Voltage')

# Display the plot
fig.show()

# %%
