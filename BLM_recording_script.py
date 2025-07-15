import time
import numpy as np 
from ctypes import *
from dwfconstants import *  # Import constants from the WaveForms SDK
import os
import datetime
from sys import platform                 # this is needed to check the OS type
# Plotting optional
# import matplotlib.pyplot as plt

# Load DWF
# load the dynamic library (the path is OS specific)
if platform.startswith("win"):
    # on Windows
    dwf = cdll.dwf
elif platform.startswith("darwin"):
    # on macOS
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    # on Linux
    dwf = cdll.LoadLibrary("libdwf.so")

# Open device
hdwf = c_int()
trigger_conditions = c_int()

cDevice = c_int()
dwf.FDwfEnum(enumfilterType.value|enumfilterNetwork.value, None)
time.sleep(1)
dwf.FDwfEnum(enumfilterType.value|enumfilterNetwork.value, byref(cDevice))
if cDevice.value < 1:
    time.sleep(30)
    dwf.FDwfEnum(enumfilterType.value|enumfilterNetwork.value, byref(cDevice))
dwf.FDwfDeviceOpen(0, byref(hdwf))

if hdwf.value == 0:
    print("Failed to open device")
    quit()

print("Device opened")

# Set up waveform generator: sine wave
waveform_generator = True
if waveform_generator:
    frequency = 1000  # 1 kHz
    amplitude = 1.0   # 1 V peak

    dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_bool(True))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(frequency))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(amplitude))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_bool(True))  # Start Wavegen

    print(f"Started sine wave on Wavegen Channel 1: {frequency} Hz, {amplitude} V")

    coupling = c_int()
    channel = 0  # Channel 1 (0-indexed)

    # Call the function
    dwf.FDwfAnalogInChannelCouplingGet(hdwf, c_int(channel), byref(coupling))

    # Interpret result
    if coupling.value == DwfAnalogCouplingDC:
        print("Channel coupling: DC")
    elif coupling.value == DwfAnalogCouplingAC:
        print("Channel coupling: AC")
    else:
        print(f"Unknown coupling: {coupling.value}")

# Set up analog input
sample_rate = 250e6     # 100 MS/s
#buffer_size = 8192      # Number of samples per channel
trigger_level = 0.5     # Trigger threshold in volts

dwf.FDwfAnalogInFrequencySet(hdwf, c_double(sample_rate))
# Query buffer size limits
min_buf = c_int()
max_buf = c_int()
dwf.FDwfAnalogInBufferSizeInfo(hdwf, byref(min_buf), byref(max_buf))

print(f"Minimum buffer size: {min_buf.value}")
print(f"Maximum buffer size: {max_buf.value}")

buffer_size = max_buf.value # Number of samples per channel
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(buffer_size))

# Enable both channels
for ch in [0, 1, 2, 3]:
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(ch), c_bool(True))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), c_double(5.0))
    # Set input impedance to 50 Ohm
    dwf.FDwfAnalogInChannelImpedanceSet(hdwf, c_int(ch), c_double(50))

# Set up trigger: rising edge on CH1
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcDetectorAnalogIn)
dwf.FDwfAnalogInTriggerAutoTimeoutSet(hdwf, c_double(0))  # Wait indefinitely
dwf.FDwfAnalogInTriggerTypeSet(hdwf, trigtypeEdge)
dwf.FDwfAnalogInTriggerChannelSet(hdwf, c_int(0))  # Channel 1
dwf.FDwfAnalogInTriggerLevelSet(hdwf, c_double(trigger_level))
dwf.FDwfAnalogInTriggerConditionSet(hdwf, trigcondRisingPositive)

# Set trigger position: 50% of buffer before trigger
trigger_position = -0.5 * (buffer_size / sample_rate)
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(trigger_position))

# Allocate buffers
samples_ch1 = (c_double * buffer_size)()
samples_ch2 = (c_double * buffer_size)()
samples_ch3 = (c_double * buffer_size)()
samples_ch4 = (c_double * buffer_size)()

# Output directory
outdir = "waveforms_npz"
os.makedirs(outdir, exist_ok=True)

#t = np.arange(buffer_size) / sample_rate

print("Starting continuous acquisition... Press Ctrl+C to stop.")
i = 0
try:
    while True:
        # Start acquisition
        dwf.FDwfAnalogInConfigure(hdwf, c_bool(False), c_bool(True))

        # Wait for trigger
        sts = c_byte()
        while True:
            dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
            #print('sts.value = ',sts.value)
            if sts.value == DwfStateDone.value:
                break
            time.sleep(0.01)

        # Read both channels
        dwf.FDwfAnalogInStatusData(hdwf, c_int(0), samples_ch1, buffer_size)
        dwf.FDwfAnalogInStatusData(hdwf, c_int(1), samples_ch2, buffer_size)
        dwf.FDwfAnalogInStatusData(hdwf, c_int(2), samples_ch3, buffer_size)
        dwf.FDwfAnalogInStatusData(hdwf, c_int(3), samples_ch4, buffer_size)

        ch1 = np.fromiter(samples_ch1, dtype=np.float64)
        ch2 = np.fromiter(samples_ch2, dtype=np.float64)
        ch3 = np.fromiter(samples_ch3, dtype=np.float64)
        ch4 = np.fromiter(samples_ch4, dtype=np.float64)
        timestamp = datetime.datetime.now().isoformat()

        # Save as compressed npz
        filename = os.path.join(outdir, f"waveform_{i:05d}.npz")
        #np.savez_compressed(filename, ch1=ch1, ch2=ch2, timestamp=timestamp)
        np.savez_compressed(filename, ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4, timestamp=timestamp, sample_rate=sample_rate, trigger_level=trigger_level, buffer_size=buffer_size)
        print(f"Saved: {filename}")
        i += 1



except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    dwf.FDwfDeviceClose(hdwf)
    print("Device closed.")
