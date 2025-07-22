from pathlib import Path
from datetime import datetime, timedelta
import re
import numpy as np
import pandas as pd

def get_df(ch_match = "peak*CH1.npz"):
    # Define base path
    base_path = r"C:/Users/shulg/OneDrive - Brookhaven National Laboratory/Work/BLM_data/waveforms_npz/"
    base_path = r"/Users/evgenyshulga/Library/CloudStorage/OneDrive-BrookhavenNationalLaboratory/Work/BLM_data/waveforms_npz/"
    # Base directory
    base_dir = Path(base_path)
    # Time window setup
    now = datetime.now()
    target_time = now - timedelta(hours=3)  # N hours ago
    time_window_start = target_time - timedelta(minutes=10)
    time_window_end = target_time + timedelta(minutes=10)

    # Regex pattern for datetime in folder name: e.g., 2025-07-21_03-25-57
    datetime_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')


    # Collect matching files
    matching_files = []

    for subdir in base_dir.iterdir():
        #print(subdir)
        if subdir.is_dir():
            match = datetime_pattern.search(subdir.name)
            #print(subdir.name)
            if match:
                datetime_str  = match.group()
                try:
                    dt = datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
                    if time_window_start <= dt <= time_window_end:
                        # Find matching files in this dir
                        for file in subdir.glob(ch_match):
                            matching_files.append(file.resolve())
                except ValueError:
                    continue

    all_data = []
    # Print full paths of matching files
    for file_path in matching_files:
        print(file_path)
        #d = np.load(file_path)
        with np.load(file_path) as data:
            # Assuming each .npz has 'array1' and 'array2'
            # and we want them as columns in the DataFrame
            all_data.append({
                            'peak_indicies'  :data['peak_indicies'].tolist(),  
                            'peak_heights'   :data['peak_heights'].tolist(),
                            'widths'         :data['widths'].tolist(), 
                            'dist_to_revsig' :data['dist_to_revsig'].tolist(), 
                            'integrals'      :data['integrals'].tolist() ,       
                            'timestamp'      :data['timestamp'].tolist(), 
                            'sample_rate'    :data['sample_rate'].tolist(), 
                            'buffer_size'    :data['buffer_size'].tolist(), 
                            'trigger_chn'    :data['trigger_chn'].tolist(), 
                            'trigger_level'  :data['trigger_level'].tolist()
                            })
        
    # 4. Create the DataFrame
    df = pd.DataFrame(all_data)
    return df
df_1 = get_df(ch_match = "peak*CH1.npz")
print(df_1)
df_2 = get_df(ch_match = "peak*CH2.npz")
print(df_2)
df_3 = get_df(ch_match = "peak*CH3.npz")
print(df_2)

import matplotlib.pyplot as plt

# Assuming your DataFrame is called df

# Extract single values from list columns
df_1['peak_heights_val']   = df_1['peak_heights'].apply(lambda x: x[0])
df_1['dist_to_revsig_val'] = df_1['dist_to_revsig'].apply(lambda x: x[0])
df_2['peak_heights_val']   = df_2['peak_heights'].apply(lambda x: x[0])
df_2['dist_to_revsig_val'] = df_2['dist_to_revsig'].apply(lambda x: x[0])
df_3['peak_heights_val']   = df_3['peak_heights'].apply(lambda x: x[0])
df_3['dist_to_revsig_val'] = df_3['dist_to_revsig'].apply(lambda x: x[0])

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(df_1['dist_to_revsig_val'], df_1['peak_heights_val'], color='blue')
plt.scatter(df_2['dist_to_revsig_val'], df_2['peak_heights_val'], color='red')
plt.scatter(df_3['dist_to_revsig_val'], df_3['peak_heights_val'], color='darkgreen')
plt.xlabel('Distance to RevSig')
plt.ylabel('Peak Height')
plt.title('Peak Height vs Distance to RevSig')
plt.grid(True)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Flatten the dist_to_revsig values (assumes each cell is a list)
all_distances_1 = df_1['dist_to_revsig'].explode().astype(float)
all_distances_2 = df_2['dist_to_revsig'].explode().astype(float)
all_distances_3 = df_3['dist_to_revsig'].explode().astype(float)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_distances_1, bins=2000, color='skyblue', edgecolor='blue')
plt.hist(all_distances_2, bins=2000, color='red', edgecolor='red')
plt.hist(all_distances_3, bins=2000, color='green', edgecolor='green')
plt.xlabel('Distance to RevSig')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to RevSig')
plt.grid(True)
plt.tight_layout()
plt.show()