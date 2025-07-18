import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from daq_utils import analyze_buffer

# Define base path
#input_dir = r"/Users/evgenyshulga/Library/CloudStorage/OneDrive-BrookhavenNationalLaboratory/Work/BLM_data/waveforms_npz/2025-07-17_10-04-05/"
input_dir = r"/Users/evgenyshulga/Library/CloudStorage/OneDrive-BrookhavenNationalLaboratory/Work/BLM_data/waveforms_npz/2025-07-16_16-22-40/"
base_path = r"./"
# Step 2: Load first 10 files
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])[-100:]


# Initialize channel storage

# Step 3: Load data and collect for histograms
for file in files:
    data = np.load(os.path.join(input_dir, file))
    buffer_size = data["buffer_size"]
    sample_rate = data["sample_rate"]
    channels = {'ch1': [], 'ch2': [], 'ch3': [], 'ch4': []}
    t = None  # time axis, same for all

    t = np.arange(buffer_size) / sample_rate

    for ch in channels:
        channels[ch].append(data[ch])

    results = []
    trig_threshold = [1, 0.035, 0.0012, 0.002]
    # Convert lists to arrays
    for ich, ch in enumerate(channels):
        channels[ch] = np.array(channels[ch])  # shape: (10, buffer_size)
        print(channels[ch][0])
        result = analyze_buffer(
                channels[ch][0],
                channels['ch1'][0],
                0.5,
                trig_threshold[ich] ,
                1,
                1,
                [1,2],
            )
        print(result)
        results.append(result)
    # Step 4: Plot time-series
    plt.figure(figsize=(12, 10))
    for i, ch in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
        plt.subplot(4, 1, i+1)
        for waveform in channels[ch]:
            plt.plot(t, waveform, alpha=0.5)
            plt.plot(results[i]['peak_indicies']/sample_rate, results[i]['peak_heights'], color='red', marker='o')
        plt.title(f"{ch} Time-Series")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
    ts_plot_path = os.path.join(base_path, f"timeseries.png")
    plt.savefig(ts_plot_path)
    print(f"Saved time-series plot to: {ts_plot_path}")
    plt.close()

