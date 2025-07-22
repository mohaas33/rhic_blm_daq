import time
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ctypes import *
from dwfconstants import *  # Import constants from the WaveForms SDK
import os
import datetime
from sys import *                 # this is needed to check the OS type
# Plotting optional
# import matplotlib.pyplot as plt
import sys
# Peak finding algorithm
from daq_utils import analyze_buffer

# total arguments
n = len(argv)
print(argv)
if n<3:
    print("Not enough input variables: trigger_chn trigger_level ch_label")
    quit()    

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



plot_test = True#False
trigger_chn = int(argv[1])
sample_rate = 2*750e6     # 100 MS/s
buffer_size = 1_000_000      # Number of samples per channel
trigger_level = float(argv[2])    # Trigger threshold in volts
#trigger_level1 = 0.0012    # Trigger threshold in volts
#trigger_level2 = 0.002    # Trigger threshold in volts
ch_label = argv[3]
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
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0)) # 0 = the device will only be configured when FDwf###Configure is calle

# Set up waveform generator: sine wave
waveform_generator = False
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
#else:
print(f"Unknown coupling: {coupling.value}")


# Enable channels
for ch in [0, trigger_chn]:
    dwf.FDwfAnalogInChannelEnableSet(hdwf, c_int(ch), c_bool(True))
    dwf.FDwfAnalogInChannelCouplingSet(hdwf, c_int(ch), DwfAnalogCouplingDC) #DC coupling
    if ch==0:
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), c_double(5))
    if ch==1:
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), c_double(0.5))
    if ch==2:
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), c_double(0.01))
    if ch==3:
        dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), c_double(0.01))
    # Set input impedance to 50 Ohm
    dwf.FDwfAnalogInChannelImpedanceSet(hdwf, c_int(ch), c_double(50))

# Set up analog input

dwf.FDwfAnalogInAcquisitionModeSet(hdwf, acqmodeRecord)
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(sample_rate))
# Query buffer size limits
min_buf = c_int()
max_buf = c_int()
dwf.FDwfAnalogInBufferSizeInfo(hdwf, byref(min_buf), byref(max_buf))

print(f"Minimum buffer size: {min_buf.value}")
print(f"Maximum buffer size: {max_buf.value}")

#buffer_size = max_buf.value # Number of samples per channel
#buffer_size = 10000 # Number of samples per channel
dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(buffer_size))



# Set up trigger: rising edge on CH1
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcDetectorAnalogIn)
dwf.FDwfAnalogInTriggerAutoTimeoutSet(hdwf, c_double(0))  # Wait indefinitely
dwf.FDwfAnalogInTriggerTypeSet(hdwf, trigtypeEdge)
dwf.FDwfAnalogInTriggerChannelSet(hdwf, c_int(trigger_chn))  # Channel set 
dwf.FDwfAnalogInTriggerLevelSet(hdwf, c_double(trigger_level))
dwf.FDwfAnalogInTriggerConditionSet(hdwf, trigcondRisingPositive)

# Set trigger position: 50% of buffer before trigger
#trigger_position = -0.5 * (buffer_size / sample_rate)
#dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(trigger_position))

#dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))
#time.sleep(2)

# Allocate buffers
#samples_ch1 = (c_double * buffer_size)()
#samples_ch2 = (c_double * buffer_size)()
#samples_ch3 = (c_double * buffer_size)()
#samples_ch4 = (c_double * buffer_size)()

# Output directory
TS_FORMAT = "%Y%m%d_%H%M%S%f"
DATA_DIR = "./data/"

outdir = "C:/Users/shulg/OneDrive - Brookhaven National Laboratory/Work/BLM_data/waveforms_npz/"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

        channels_wf = []
        # Read all channels
        for chi in [0, trigger_chn]:
            samples_ch = (c_double * buffer_size)()
            dwf.FDwfAnalogInStatusData(hdwf, c_int(chi), samples_ch, buffer_size)
            #dwf.FDwfAnalogInStatusData(hdwf, c_int(1), samples_ch2, buffer_size)
            #dwf.FDwfAnalogInStatusData(hdwf, c_int(2), samples_ch3, buffer_size)
            #dwf.FDwfAnalogInStatusData(hdwf, c_int(3), samples_ch4, buffer_size)

            ch = np.fromiter(samples_ch, dtype=np.float64)
            channels_wf.append(ch)
            #ch2 = np.fromiter(samples_ch2, dtype=np.float64)
            #ch3 = np.fromiter(samples_ch3, dtype=np.float64)
            #ch4 = np.fromiter(samples_ch4, dtype=np.float64)

        #Get triggered peak position
        result = analyze_buffer(
            channels_wf[1],
            channels_wf[0], 
            0.5,
            trigger_level ,
            1,
            2,
            [1,2],
        )

        timestamp = datetime.datetime.now().isoformat()
        if plot_test:
            # Plot results (example for channel 1)
            time_axis = np.arange(buffer_size) / sample_rate
            #plt.plot(time_axis * 1e6, channels_data[0])  # Time in microseconds
            #plt.plot(time_axis * 1e6, samples_ch1)  # Time in microseconds
            for samples_ch in channels_wf:
                plt.plot(time_axis * 1e9, samples_ch)  # Time in microseconds
            plt.plot(result['peak_indicies']/sample_rate*1e9, result['peak_heights'], color='red', marker='o')    
            #plt.plot(time_axis * 1e9, samples_ch2)  # Time in microseconds
            #plt.plot(time_axis * 1e9, samples_ch3)  # Time in microseconds
            #plt.plot(time_axis * 1e9, samples_ch4)  # Time in microseconds
            plt.title("Channel 1 Scope Data at 250 MS/s")
            plt.xlabel("Time (ns)")
            plt.ylabel("Voltage (V)")
            plt.grid(True)
            plt.show()

        peaks = pd.DataFrame(result)

        # Save as compressed npz
        filename = os.path.join(outdir, f"waveform_{i:05d}_{ch_label}.npz")
        #np.savez_compressed(filename, ch1=ch1, ch2=ch2, timestamp=timestamp)
        np.savez_compressed(filename, peaks=peaks, 
                            ch1=channels_wf[0], 
                            ch2=channels_wf[1], 
                            timestamp=timestamp, 
                            sample_rate=sample_rate, 
                            buffer_size=buffer_size, 
                            trigger_chn=trigger_chn, 
                            trigger_level=trigger_level
                            )
        filename_peaks = os.path.join(outdir, f"peak_{i:05d}_{ch_label}.npz")
        np.savez_compressed(filename_peaks, peak_indicies  = peaks['peak_indicies'], 
                            peak_heights = peaks['peak_heights'], 
                            widths = peaks['widths'],
                            dist_to_revsig = peaks['dist_to_revsig'],
                            integrals      = peaks['integrals'], 
                            timestamp=timestamp, 
                            sample_rate=sample_rate, 
                            buffer_size=buffer_size, 
                            trigger_chn=trigger_chn, 
                            trigger_level=trigger_level
                            )

        #print(peaks)
        print(f"Saved: {filename}")
        print(f"Saved: {filename_peaks}")
        i += 1



except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    dwf.FDwfDeviceClose(hdwf)
    print("Device closed.")
