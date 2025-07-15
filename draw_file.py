import numpy as np

d = np.load("waveforms_npz/waveform_00000.npz")
print(d)
buffer_size = d["buffer_size"]
sample_rate = d["sample_rate"]
ch1 = d["ch1"]
ch2 = d["ch2"]
ch3 = d["ch3"]
ch4 = d["ch4"]
t = np.arange(buffer_size) / sample_rate

timestamp = d["timestamp"]
print(t, ch1)
import matplotlib.pyplot as plt
plt.plot(t*1e6, ch1, label="CH1")
plt.plot(t*1e6, ch2, label="CH2")
plt.plot(t*1e6, ch3, label="CH3")
plt.plot(t*1e6, ch4, label="CH4")
plt.xlabel("Time [ms]")
plt.ylabel("Voltage [V]")
plt.title(f"Waveform at {timestamp}")
plt.legend()
plt.grid(True)
plt.show()