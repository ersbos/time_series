#%%
import os
import random
import torch
import matplotlib.pyplot as plt
from utilities import PreloadedSeismicDataset



data_dir = "data"
window_size = 5120      
window_stride = 2560      


dataset = PreloadedSeismicDataset(data_dir=data_dir, window_size=window_size, window_stride=window_stride, transform=None)


if len(dataset) < 10:
    raise ValueError("The dataset contains fewer than 10 samples.")


sample_indices = random.sample(range(len(dataset)), 100)


for idx in sample_indices:
    window_signal, label = dataset[idx]
    print(f"Index: {idx}, Label: {label}, Data Shape: {window_signal.shape}")

 
    signal_np = window_signal.squeeze().numpy()

    plt.figure()
    plt.plot(signal_np)
    plt.title(f"Index: {idx}, Label: {label}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

#%%
label_to_plot = 2

indices_with_label = [i for i in range(len(dataset)) if dataset[i][1] == label_to_plot]

if not indices_with_label:
    print(f"No samples found with label {label_to_plot}.")
else:
    print(f"Found {len(indices_with_label)} samples with label {label_to_plot}.")

    # Optionally, sample a few indices if there are many.
    sample_size = min(100, len(indices_with_label))
    sample_indices = random.sample(indices_with_label, sample_size)

    for idx in sample_indices:
        window_signal, label = dataset[idx]
    
        signal_np = window_signal.squeeze().numpy()

        # Plotting the seismic signal.
        plt.figure()
        plt.plot(signal_np)
        plt.title(f"Index: {idx}, Label: {label}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
# %%
