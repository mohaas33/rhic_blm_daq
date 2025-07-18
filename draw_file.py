import numpy as np

d = np.load("C:/Users/shulg/OneDrive - Brookhaven National Laboratory/Work/BLM_data/waveforms_npz/2025-07-18_17-10-58\waveform_00000_CH1.npz")
print(d)
buffer_size = d["buffer_size"]
sample_rate = d["sample_rate"]
ch1 = d["ch1"]
ch2 = d["ch2"]
peaks = d["peaks"]
print("Peaks: ",peaks)
t = np.arange(buffer_size) / sample_rate

timestamp = d["timestamp"]
#print(t, ch1)
import matplotlib.pyplot as plt
plt.plot(t*1e6, ch1, label="CH1")
plt.plot(t*1e6, ch2, label="CH2")

plt.xlabel("Time [ns]")
plt.ylabel("Voltage [V]")
plt.title(f"Waveform at {timestamp}")
plt.legend()
plt.grid(True)
plt.show()