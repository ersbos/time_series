import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def detect_step_events(signal_data, window_size=512, threshold_factor=3, noise_sigma=1.0):
    """
    Detect step events using a chi-square based threshold.

    Parameters:
        signal_data (np.array): The seismic signal data.
        window_size (int): The number of samples per window.
        threshold_factor (float): The number of standard deviations above the mean.
        noise_sigma (float): The assumed standard deviation of the noise.

    Returns:
        List of indices in the original signal where step events occur.
    """
    windowed_energy = []
    step_events = []

    # Number of complete windows
    num_windows = len(signal_data) // window_size
    for i in range(num_windows):
        start = i * window_size
        window = signal_data[start:start + window_size]

        # Calculate energy: mean squared amplitude in the window.
        energy = np.sum(window ** 2) / window_size
        windowed_energy.append(energy)

    # Theoretical energy based on noise model (chi-square properties):
    mean_energy = noise_sigma ** 2
    std_energy = noise_sigma ** 2 * np.sqrt(2 / window_size)
    threshold = mean_energy + threshold_factor * std_energy
    #print(f"Using noise model: mean_energy = {mean_energy:.4f}, std_energy = {std_energy:.4f}, threshold = {threshold:.4f}")

    # If a window's energy exceeds the threshold, record the starting index of that window.
    for i, energy in enumerate(windowed_energy):
        if energy > threshold:
            step_events.append(i * window_size)

    return step_events

def detect_step_events_old(signal_data, window_size=100, threshold_factor=3):
    # Sliding window to calculate energy
    windowed_energy = []
    step_events = []
    for i in range(0, len(signal_data), window_size):
        window = signal_data[i:i + window_size]
        energy = np.sum(window ** 2) / len(window)  # Mean energy in the window
        windowed_energy.append(energy)
    
    # Set threshold based on noise model
    noise_energy_mean = np.mean(windowed_energy)
    noise_energy_std = np.std(windowed_energy)
    threshold = noise_energy_mean + threshold_factor * noise_energy_std
    
    # Detect steps based on threshold
    for i, energy in enumerate(windowed_energy):
        if energy > threshold:
            step_events.append(i * window_size)
    
    return step_events

def plot_labeled_windows(signal_data, labeled_windows, window_size, title="Labeled Windows"):
    """
    Plots a signal with red dots indicating the midpoint of labeled windows.

    Parameters:
        signal_data (list or np.array): The signal values.
        labeled_windows (list): A list of starting indices for windows considered as labeled.
        window_size (int): The size of each window.
        title (str): The title of the plot.

    Returns:
        None; displays an interactive Plotly plot.
    """
    fig = go.Figure()

    # Create a continuous line plot of the signal.
    x_axis = list(range(len(signal_data)))
    fig.add_trace(go.Scatter(x=x_axis, y=signal_data, mode='lines', name='Signal'))

    # Calculate the midpoint for each labeled window and get the corresponding signal value.
    red_dot_x = []
    red_dot_y = []
    for start in labeled_windows:
        midpoint = int(start + window_size / 2)
        # Only plot if midpoint exists in the signal
        if midpoint < len(signal_data):
            red_dot_x.append(midpoint)
            red_dot_y.append(signal_data[midpoint])

    # Plot red dots to indicate detected step events.
    fig.add_trace(go.Scatter(
        x=red_dot_x,
        y=red_dot_y,
        mode='markers',
        marker=dict(color='red', size=10),
        name='Detected Step'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Signal Value"
    )
    fig.show()

if __name__ == "__main__":
    # Directory containing CSV files
    data_folder = os.path.join("data", "person_two")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        exit(1)

    # For demonstration, we'll load the first CSV file.
    csv_path = csv_files[0]
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        exit(1)

    # Assumes the CSV contains a column named "voltage"
    if "voltage" not in data.columns:
        print(f"CSV file {csv_path} does not have a 'voltage' column.")
        exit(1)

    signal = data["voltage"].values.astype(np.float32)

    # Parameters for step detection
    window_size = 512         # Adjust based on expected step window
    threshold_factor = 3      # +3 standard deviations in the noise model
    noise_sigma = 25.0         # Set as appropriate for your sensor

    detected_windows = detect_step_events(signal, window_size=window_size,
                                          threshold_factor=threshold_factor,
                                         noise_sigma=noise_sigma)
    print("Detected step events at indices:", detected_windows)

    # Plot the signal and overlay detected step events.
    plot_labeled_windows(signal, detected_windows, window_size,
                         title=f"Labeled Windows for {os.path.basename(csv_path)}")