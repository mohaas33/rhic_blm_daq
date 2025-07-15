import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Define base path
base_path = r"C:/Users/shulg/OneDrive - Brookhaven National Laboratory/Work/BLM_data/waveforms_npz"

# Step 1: Find latest directory
def get_latest_dir(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    dated_dirs = []
    for d in dirs:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d_%H-%M-%S")
            dated_dirs.append((dt, d))
        except ValueError:
            continue
    latest_dt, latest_dir_name = max(dated_dirs, key=lambda x: x[0])
    return os.path.join(base_path, latest_dir_name), latest_dir_name

latest_dir, latest_dir_name = get_latest_dir(base_path)
print("Latest directory:", latest_dir)

# Step 2: Load first 10 files
files = sorted([f for f in os.listdir(latest_dir) if f.endswith(".npz")])[:10]

# Initialize channel storage
channels = {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': []}
t = None  # time axis, same for all

# Step 3: Load data and collect for histograms
for file in files:
    data = np.load(os.path.join(latest_dir, file))
    buffer_size = data["buffer_size"]
    sample_rate = data["sample_rate"]
    t = np.arange(buffer_size) / sample_rate

    for ch in channels:
        channels[ch].append(data[ch])

# Convert lists to arrays
for ch in channels:
    channels[ch] = np.array(channels[ch])  # shape: (10, buffer_size)

# Step 4: Plot time-series
plt.figure(figsize=(12, 10))
for i, ch in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
    plt.subplot(4, 1, i+1)
    for waveform in channels[ch]:
        plt.plot(t, waveform, alpha=0.5)
    plt.title(f"{ch} Time-Series")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

plt.tight_layout()
ts_plot_path = os.path.join(base_path, f"{latest_dir_name}_timeseries.png")
plt.savefig(ts_plot_path)
print(f"Saved time-series plot to: {ts_plot_path}")
plt.close()

# Step 5: Plot histograms
plt.figure(figsize=(12, 10))
for i, ch in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
    plt.subplot(4, 1, i+1)
    all_values = channels[ch].flatten()
    plt.hist(all_values, bins=100, alpha=0.7)
    plt.title(f"{ch} Value Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")

plt.tight_layout()
hist_plot_path = os.path.join(base_path, f"{latest_dir_name}_histograms.png")
plt.savefig(hist_plot_path)
print(f"Saved histogram plot to: {hist_plot_path}")
plt.close()
