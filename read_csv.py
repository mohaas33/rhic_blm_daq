import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import ast

import glob


col = {1:'blue', 2:'red', 3:'darkgreen'}
ch_label = {1:'CVD BLM', 2:'Plastic BLM', 3:'Quartz BLM'}

df_ini = 0
# Find all CSV files in the folder
csv_files = glob.glob('../Data/peaks_*.csv')

# Read and concatenate
dfs = [pd.read_csv(f) for f in csv_files]
df_ini = pd.concat(dfs, ignore_index=True)
date_string = '2025-07-18_22' 
# Helper function: parse array-like strings with space separators
def parse_array(s):
    return np.fromstring(s.strip("[]"), sep=' ')

### Path to your CSV
#date_string = '2025-07-22' 
#csv_path = f'../Data/peaks_{date_string}.csv'
#
## Load the CSV into a DataFrame
#df_ini = pd.read_csv(csv_path)

# Preview the first few rows
print(df_ini.head())

## Helper function to parse string arrays into Python lists of floats or ints
#def parse_array_str(s):
#    s = s.strip('[]')
#    # split by space, handle empty string case
#    if s == '':
#        return []
#    # Try to convert to int if possible else float
#    items = s.split()
#    try:
#        return [int(i) for i in items]
#    except ValueError:
#        return [float(i) for i in items]
#
## Columns that contain these array-like strings
#array_cols = ['peak_indicies', 'peak_heights', 'widths', 'dist_to_revsig', 'integrals']
#
## Parse each of those columns
#for col in array_cols:
#    df_ini[col] = df_ini[col].apply(parse_array_str)
# Your array-like columns
array_cols = ['peak_indicies', 'peak_heights', 'widths', 'dist_to_revsig', 'integrals']

# Step 1: Parse the strings into lists
def parse_array_str(s):
    if not isinstance(s, str):
        return []
    s = s.strip('[]').strip()
    if not s:
        return []
    return s.split()

for col in array_cols:
    df_ini[col] = df_ini[col].apply(parse_array_str)

# Step 2: Clean and align arrays row-by-row
def clean_row(row):
    arrays = {col: row[col] for col in array_cols}
    
    # Mark valid indices based on whether 'peak_indicies' are integers
    valid_indices = []
    for i, val in enumerate(arrays['peak_indicies']):
        try:
            int(val)
            valid_indices.append(i)
        except ValueError:
            continue

    # Now filter each array to keep only valid indices
    for col in array_cols:
        arr = arrays[col]
        arrays[col] = [arr[i] for i in valid_indices if i < len(arr)]

    # Optional: convert to int/float
    try:
        arrays['peak_indicies'] = [int(x) for x in arrays['peak_indicies']]
    except:
        arrays['peak_indicies'] = []
    try:
        arrays['peak_heights'] = [float(x) for x in arrays['peak_heights']]
        arrays['widths'] = [float(x) for x in arrays['widths']]
        arrays['dist_to_revsig'] = [float(x) for x in arrays['dist_to_revsig']]
        arrays['integrals'] = [float(x) for x in arrays['integrals']]
    except:
        pass  # You can choose to drop or log here

    # Return updated row
    for col in array_cols:
        row[col] = arrays[col]
    return row

# Apply row cleaning
df_ini = df_ini.apply(clean_row, axis=1)

# Now explode each of those columns simultaneously
# First, ensure that each list in array columns has the same length per row
assert all(
    (df_ini[array_cols[0]].str.len() == df_ini[col].str.len()).all()
    for col in array_cols[1:]
), "Array columns have differing lengths within rows!"

# Explode the dataframe by repeating rows for each element in the arrays
df = df_ini.explode(array_cols).reset_index(drop=True)

print(df)

import matplotlib.pyplot as plt


print(df['dist_to_revsig'])
# Apply parsing
#df['dist_to_revsig_val'] = df['dist_to_revsig'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())
#df['peak_heights_val']   = df['peak_heights'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ').tolist())

#df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
bad_timestamps = df[df['timestamp'].isna()]
print("Bad timing: ",bad_timestamps)

# Convert peak_heights to numeric if it's string/object
df['peak_heights'] = pd.to_numeric(df['peak_heights'], errors='coerce')



avg_min = 6 #data taking is 2 min each, should be >6 minutes
# Count occurrences per 2 minutes for each channel
counts = df.groupby([
    pd.Grouper(key='timestamp', freq=f'{avg_min}min'),  # round timestamps to avg_min bins
    'ch'
]).size().reset_index(name='count')
# Group and compute mean peak_heights
avg_heights = df.groupby([
    pd.Grouper(key='timestamp', freq=f'{avg_min}min'),
    'ch'
])['peak_heights'].mean().reset_index(name='avg_peak_height')

# Merge rate and average height by time + channel
combined = pd.merge(counts, avg_heights, on=['timestamp', 'ch'], how='left')

# Optional: calculate rate per minute (divide by avg_min)
counts['rate_per_min'] = counts['count'] / (avg_min/3)
counts['rate_per_sec'] = counts['count'] / (avg_min/3*60)
# Optional: calculate rate per minute (divide by avg_min)
combined['rate_per_min'] = combined['count'] / (avg_min/3)
combined['rate_per_sec'] = combined['count'] / (avg_min/3*60)


plt.figure(figsize=(8, 5))
col = {1:'blue', 2:'red', 3:'darkgreen'}
for ch in combined['ch'].unique():
    if ch==1:
        df_ch = combined[(counts['ch'] == ch) & (combined['rate_per_sec'] > 0) & (combined['avg_peak_height'] > 0.08)]
    else:
        df_ch = combined[(counts['ch'] == ch) & (combined['rate_per_sec'] > 0) & (combined['avg_peak_height'] > 0.08)]
    plt.plot(df_ch['timestamp'], df_ch['rate_per_sec'], label=ch_label[ch], marker='o', linestyle='', color=col[ch],  alpha=0.6)

plt.xlabel('Time')
plt.ylabel('Rate [Hz]')
plt.title(f'Channel Activity Rate ({avg_min} min bins)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f"rate_vs_time_{date_string}.png", dpi=300)

plt.show()

plt.figure(figsize=(8, 5))
col = {1:'blue', 2:'red', 3:'darkgreen'}
for ch in combined['ch'].unique():
    df_ch = combined[(combined['ch'] == ch) & (combined['rate_per_sec'] > 0)]
    plt.plot(df_ch['avg_peak_height'], df_ch['rate_per_sec'], label=ch_label[ch], marker='o', linestyle='', color=col[ch],  alpha=0.6)

plt.xlabel('peak_heights')
plt.ylabel('Rate [Hz]')
plt.title(f'Channel Activity Rate ({avg_min} min bins)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f"peak_heights_vs_rate_{date_string}.png", dpi=300)

plt.show()



# Group by channel
channels = df['ch'].unique()
df['dist_to_revsig_ns'] = -1*df['dist_to_revsig'] * 1.0/df['sample_rate'] * 1e9
print(df['dist_to_revsig_ns'])
print(df.head())
# Plot
plt.figure(figsize=(8, 5))
        
plt.scatter(df[df['ch']==1]['dist_to_revsig_ns'], df[df['ch']==1]['peak_heights'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(df[df['ch']==2]['dist_to_revsig_ns'], df[df['ch']==2]['peak_heights'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(df[df['ch']==3]['dist_to_revsig_ns'], df[df['ch']==3]['peak_heights'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('Distance to RevSig [ns]')
plt.ylabel('Amplitude [V]')
plt.title('Peak Height vs Distance to RevSig')
plt.grid(True)
plt.xlim(-1, 13000)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig(f"amp_vs_revsig_{date_string}.png", dpi=300)

plt.show()

plt.figure(figsize=(8, 5))
        
plt.scatter(df[df['ch']==1]['timestamp'], df[df['ch']==1]['peak_heights'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(df[df['ch']==2]['timestamp'], df[df['ch']==2]['peak_heights'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(df[df['ch']==3]['timestamp'], df[df['ch']==3]['peak_heights'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('time stamp')
plt.ylabel('Amplitude [V]')
plt.title('Peak hights vs time')
plt.grid(True)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig(f"amp_vs_time_{date_string}.png", dpi=300)

plt.show()

# Flatten the dist_to_revsig values (assumes each cell is a list)
all_distances_1 = df[(df['ch']==1 ) & ( df['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)
all_distances_2 = df[(df['ch']==2 ) & ( df['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)
all_distances_3 = df[(df['ch']==3 ) & ( df['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_distances_1, bins=2000, color='skyblue', edgecolor='blue', label=f"Ch 1", alpha=0.6)
plt.hist(all_distances_2, bins=2000, color='red', edgecolor='red'     , label=f"Ch 2", alpha=0.6)
plt.hist(all_distances_3, bins=2000, color='green', edgecolor='green' , label=f"Ch 3", alpha=0.6)
plt.xlabel('Distance to RevSig [ns]')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to RevSig')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f"revsig_{date_string}.png", dpi=300)

plt.show()

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(df[df['ch']==1]['peak_heights'], bins=200, color='skyblue', edgecolor='blue', label=f"Ch 1", alpha=0.6)
plt.hist(df[df['ch']==2]['peak_heights'], bins=200, color='red', edgecolor='red'     , label=f"Ch 2", alpha=0.6)
plt.hist(df[df['ch']==3]['peak_heights'], bins=200, color='green', edgecolor='green' , label=f"Ch 3", alpha=0.6)

plt.xlabel('Amplitude [V]')
plt.ylabel('#')
plt.title('Histogram of Amplitudes')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

plt.legend()

plt.tight_layout()
plt.savefig(f"amp_{date_string}.png", dpi=300)
plt.show()

# Load your main data (assuming it's already in variable df)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load the intervals CSV
intervals = pd.read_csv('stable_beam_intervals_2025_0716-0724.csv')
intervals['start'] = pd.to_datetime(intervals['start'])
intervals['end'] = pd.to_datetime(intervals['end'])

# Create a mask to keep only rows in df within any interval
mask = pd.Series(False, index=df.index)

for _, row in intervals.iterrows():
    mask |= (df['timestamp'] >= row['start']) & (df['timestamp'] <= row['end'])

# Filtered DataFrame
df_filtered = df[mask].copy()

print(df_filtered)

# Flatten the dist_to_revsig values (assumes each cell is a list)
all_distances_1_filtered = df_filtered[(df_filtered['ch']==1 ) & ( df_filtered['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)
all_distances_2_filtered = df_filtered[(df_filtered['ch']==2 ) & ( df_filtered['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)
all_distances_3_filtered = df_filtered[(df_filtered['ch']==3 ) & ( df_filtered['dist_to_revsig']<=0)]['dist_to_revsig_ns'].explode().astype(float)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_distances_1_filtered, bins=2000, color='skyblue', edgecolor='blue', label=f"Ch 1", alpha=0.6)
plt.hist(all_distances_2_filtered, bins=2000, color='red', edgecolor='red'     , label=f"Ch 2", alpha=0.6)
plt.hist(all_distances_3_filtered, bins=2000, color='green', edgecolor='green' , label=f"Ch 3", alpha=0.6)
plt.xlabel('Distance to RevSig [ns]')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to RevSig')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(f"filtered_revsig_{date_string}.png", dpi=300)

plt.show()

plt.figure(figsize=(8, 5))
        
plt.scatter(df_filtered[df_filtered['ch']==1]['timestamp'], df_filtered[df_filtered['ch']==1]['peak_heights'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(df_filtered[df_filtered['ch']==2]['timestamp'], df_filtered[df_filtered['ch']==2]['peak_heights'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(df_filtered[df_filtered['ch']==3]['timestamp'], df_filtered[df_filtered['ch']==3]['peak_heights'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('time stamp')
plt.ylabel('Amplitude [V]')
plt.title('Peak hights vs time')
plt.grid(True)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig(f"amp_vs_time_{date_string}.png", dpi=300)

plt.show()




# Define column names
cols = ['date', 'time', 'flag']
use_dates = '2025_0716-0724'
# Read both files
#df_start = pd.read_csv(f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_startramp_{use_dates}.dat', sep='\s+', names=cols)
#df_end   = pd.read_csv(f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_endramp_{use_dates}.dat', sep='\s+', names=cols)

#------------ Read RHIC data with combining date/time
# Get beam current data
df_current = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_beamcurrent_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'bluDCCTtotal', 'yelDCCTtotal', 'RhicState', 'Fill']
)
df_start = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_startramp_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'Flag']
)
df_end = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_endramp_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'Flag']
)
# Get BLM data
df_blm = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_blm_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'Array', 'Counts1', 'Counts2']
)

# Get beam abort data
df_abort = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_abort_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'Abort']
)

# Get blue beam emitx data
df_blue_nemitx = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_blue_emitx_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'nEmit']
)

# Get blue beam emity data
df_blue_nemity = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_blue_emity_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'nEmit']
)

# Get yellow beam emitx data
df_yellow_nemitx = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_yellow_emitx_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'nEmit']
)

# Get yellow beam emity data
df_yellow_nemity = pd.read_csv(
    f'/Users/evgenyshulga/Work/EIC/BLM/BEAM_RHIC_DATA/rhic_data_yellow_emity_{use_dates}.dat',
    sep=r'\s+',
    comment='#',
    header=None,
    names=['Date', 'Time', 'nEmit']
)

# Combine and convert with datetime.strptime
df_current['Timestamp'] = df_current.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_abort['Timestamp'] = df_abort.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_blm['Timestamp'] = df_blm.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_blue_nemitx['Timestamp'] = df_blue_nemitx.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_blue_nemity['Timestamp'] = df_blue_nemity.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_yellow_nemitx['Timestamp'] = df_yellow_nemitx.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_yellow_nemity['Timestamp'] = df_yellow_nemity.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_start['Timestamp'] = df_start.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)
df_end['Timestamp'] = df_end.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", '%m/%d/%Y %H:%M:%S'),axis=1)


# Drop the original columns and reorder
df_current.drop(columns=['Date', 'Time'], inplace=True)
df_current = df_current[['Timestamp', 'bluDCCTtotal', 'yelDCCTtotal', 'RhicState', 'Fill']]
df_blm.drop(columns=['Date', 'Time', 'Array'], inplace=True)
df_blm = df_blm[['Timestamp', 'Counts1', 'Counts2']]
df_abort.drop(columns=['Date', 'Time'], inplace=True)
df_abort = df_abort[['Timestamp', 'Abort']]
df_blue_nemitx.drop(columns=['Date', 'Time'], inplace=True)
df_blue_nemitx = df_blue_nemitx[['Timestamp', 'nEmit']]
df_blue_nemity.drop(columns=['Date', 'Time'], inplace=True)
df_blue_nemity = df_blue_nemity[['Timestamp', 'nEmit']]
df_yellow_nemitx.drop(columns=['Date', 'Time'], inplace=True)
df_yellow_nemitx = df_yellow_nemitx[['Timestamp', 'nEmit']]
df_yellow_nemity.drop(columns=['Date', 'Time'], inplace=True)
df_yellow_nemity = df_yellow_nemity[['Timestamp', 'nEmit']]
df_start.drop(columns=['Date', 'Time'], inplace=True)
df_start = df_start[['Timestamp', 'Flag']]
df_end.drop(columns=['Date', 'Time'], inplace=True)
df_end = df_end[['Timestamp', 'Flag']]



import pandas as pd

# Ensure Timestamp is datetime and set index
df_current['Timestamp'] = pd.to_datetime(df_current['Timestamp'])
df_current = df_current.set_index('Timestamp')

## Thresholds
#blu_hi, blu_lo = 210, 50
#yel_hi, yel_lo = 210, 50
#
## Function to find intervals where current drops through thresholds
#def find_drop_intervals(series, high_thresh, low_thresh):
#    in_drop = False
#    start_time = None
#    intervals = []
#
#    for time, val in series.items():
#        if not in_drop and val >= high_thresh:
#            start_time = time
#            in_drop = True
#        elif in_drop and val <= low_thresh:
#            intervals.append((start_time, time))
#            in_drop = False
#
#    return intervals

## Find individual drop intervals
#blu_intervals = find_drop_intervals(df_current['bluDCCTtotal'], blu_hi, blu_lo)
#yel_intervals = find_drop_intervals(df_current['yelDCCTtotal'], yel_hi, yel_lo)
#
## Find overlapping intervals
#overlapping_intervals = []
#for blu_start, blu_end in blu_intervals:
#    for yel_start, yel_end in yel_intervals:
#        start = max(blu_start, yel_start)
#        end = min(blu_end, yel_end)
#        if start < end:
#            overlapping_intervals.append((start, end))
#            break  # optional: only first match per blu interval



# Ensure Timestamp is datetime and sorted
#df_current['Timestamp'] = pd.to_datetime(df_current['Timestamp'])
#df_current = df_current.sort_values('Timestamp').set_index('Timestamp')

current = df_current['bluDCCTtotal']
intervals = []

i = 0
while i < len(current) - 1:
    # Find first drop below or equal to 200 from above
    if current.iloc[i] > 210 and current.iloc[i + 1] <= 210:
        drop_200_time = current.index[i + 1]

        # Now look forward for drop below or equal to 50
        j = i + 1
        while j < len(current) and current.iloc[j] > 50:
            j += 1
        if j < len(current):
            drop_50_time = current.index[j]
            intervals.append(
                (drop_200_time,
                drop_50_time)
            )
            i = j  # skip ahead to after this drop
        else:
            break
    else:
        i += 1

# Count occurrences per 2 minutes for each channel
counts = df_filtered.groupby([
    pd.Grouper(key='timestamp', freq=f'{avg_min}min'),  # round timestamps to avg_min bins
    'ch'
]).size().reset_index(name='count')


# Optional: calculate rate per minute (divide by avg_min)
counts['rate_per_min'] = counts['count'] / (avg_min/3)
counts['rate_per_sec'] = counts['count'] / (avg_min/3*60)

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df_current.index, df_current['bluDCCTtotal']/50, label='Blue Beam Current /50', color='blue')
plt.plot(df_current.index, df_current['yelDCCTtotal']/50, label='Yellow Beam Current /50', color='orange')
#plt.plot(df_start['Timestamp'], df_start['Flag'], label='Ramp START', marker='o', linestyle='-')
#plt.plot(df_end['Timestamp'], df_end['Flag'], label='Ramp END', marker='o', linestyle='-')
#plt.plot(df_abort['Timestamp'], df_abort['Abort'], label='Abort', marker='o', linestyle='-')
for start, end in intervals:#overlapping_intervals:
    plt.axvspan(start, end, color='gray', alpha=0.9)

for ch in counts['ch'].unique():
    #df_ch = counts[counts['ch'] == ch]
    #plt.plot(df_ch['timestamp'], df_ch['rate_per_sec'], label=ch_label[ch], color=col[ch],  alpha=0.9)
    df_ch = counts[(counts['ch'] == ch) & (counts['rate_per_sec'] > 0)]
    plt.plot(df_ch['timestamp'], df_ch['rate_per_sec'], label=ch_label[ch], marker='o', linestyle='', color=col[ch],  alpha=0.6)
# Convert list of dicts to DataFrame
intervals_df = pd.DataFrame(intervals,columns=['start','end'])

# Save to CSV
intervals_df.to_csv(f'stable_beam_intervals_{use_dates}.csv', index=False)

plt.xlabel("Time")
plt.ylabel("DCCT Current [units?]")
plt.title("Beam Currents with Drop Intervals Highlighted")
#plt.ylim(1e-2, 300)

#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f'stable_beam_intervals_{use_dates}.png", dpi=300)

plt.show()

print(f"[INFO] Beam current data: \n{df_current.head()}")
print(f"[INFO] C-AD BLM data: \n {df_blm.head()}")
print(f"[INFO] C-AD Abort data: \n {df_abort.head()}")
print(f"[INFO] C-AD Ramp up START data: \n {df_start.head()}")
print(f"[INFO] C-AD Ramp up END data: \n {df_end.head()}")
print(f"[INFO] C-AD Blue nEmitX data: \n {df_blue_nemitx.head()}")
print(f"[INFO] C-AD Blue nEmitY data: \n {df_blue_nemity.head()}")
print(f"[INFO] C-AD Yellow nEmitX data: \n {df_yellow_nemitx.head()}")
print(f"[INFO] C-AD Yellow nEmitY data: \n {df_yellow_nemity.head()}")


# Plot
# If not already done: combine Date and Time into datetime
#df_start['datetime'] = pd.to_datetime(df_start['Date'] + ' ' + df_start['Time'])
#df_end['datetime'] = pd.to_datetime(df_end['Date'] + ' ' + df_end['Time'])







