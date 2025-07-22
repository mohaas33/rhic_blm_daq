import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


# Helper function: parse array-like strings with space separators
def parse_array(s):
    return np.fromstring(s.strip("[]"), sep=' ')

# Path to your CSV
csv_path = 'peaks_2025-07-18.csv'

# Load the CSV into a DataFrame
df_ini = pd.read_csv(csv_path)

# Preview the first few rows
print(df_ini.head())

# Helper function to parse string arrays into Python lists of floats or ints
def parse_array_str(s):
    s = s.strip('[]')
    # split by space, handle empty string case
    if s == '':
        return []
    # Try to convert to int if possible else float
    items = s.split()
    try:
        return [int(i) for i in items]
    except ValueError:
        return [float(i) for i in items]

# Columns that contain these array-like strings
array_cols = ['peak_indicies', 'peak_heights', 'widths', 'dist_to_revsig', 'integrals']

# Parse each of those columns
for col in array_cols:
    df_ini[col] = df_ini[col].apply(parse_array_str)

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

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Count occurrences per 16 minutes for each channel
counts = df.groupby([
    pd.Grouper(key='timestamp', freq='16min'),  # round timestamps to 16-min bins
    'ch'
]).size().reset_index(name='count')


# Optional: calculate rate per minute (divide by 16)
counts['rate_per_min'] = counts['count'] / (16)
counts['rate_per_sec'] = counts['count'] / (16*60)


plt.figure(figsize=(8, 5))

for ch in counts['ch'].unique():
    df_ch = counts[counts['ch'] == ch]
    plt.plot(df_ch['timestamp'], df_ch['rate_per_sec'], label=f"Ch {ch}")

plt.xlabel('Time')
plt.ylabel('Rate [Hz]')
plt.title('Channel Activity Rate (16 min bins)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig("rate_vs_time.png", dpi=300)

plt.show()


# Group by channel
channels = df['ch'].unique()

print(df['dist_to_revsig'])
print(df.head())
# Plot
plt.figure(figsize=(8, 5))
        
plt.scatter(-1.33*df[df['ch']==1]['dist_to_revsig'], df[df['ch']==1]['peak_heights'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(-1.33*df[df['ch']==2]['dist_to_revsig'], df[df['ch']==2]['peak_heights'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(-1.33*df[df['ch']==3]['dist_to_revsig'], df[df['ch']==3]['peak_heights'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('Distance to RevSig [ns]')
plt.ylabel('Amplitude [V]')
plt.title('Peak Height vs Distance to RevSig')
plt.grid(True)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.savefig("amp_vs_revsig.png", dpi=300)

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
plt.savefig("amp_vs_time.png", dpi=300)

plt.show()

# Flatten the dist_to_revsig values (assumes each cell is a list)
all_distances_1 = -1.33*df[df['ch']==1]['dist_to_revsig'].explode().astype(float)
all_distances_2 = -1.33*df[df['ch']==2]['dist_to_revsig'].explode().astype(float)
all_distances_3 = -1.33*df[df['ch']==3]['dist_to_revsig'].explode().astype(float)

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
plt.savefig("revsig.png", dpi=300)

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
plt.savefig("amp.png", dpi=300)
plt.show()

