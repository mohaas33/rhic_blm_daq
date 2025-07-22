import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


# Helper function: parse array-like strings with space separators
def parse_array(s):
    return np.fromstring(s.strip("[]"), sep=' ')

# Path to your CSV
csv_path = 'peaks_2025-07-18_first35.csv'

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Preview the first few rows
print(df.head())

import matplotlib.pyplot as plt



# Apply parsing
df['dist_to_revsig_val'] = df['dist_to_revsig'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')[0])
df['peak_heights_val']   = df['peak_heights'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')[0])

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Count occurrences per 15 minutes for each channel
counts = df.groupby([
    pd.Grouper(key='timestamp', freq='6min'),  # round timestamps to 15-min bins
    'ch'
]).size().reset_index(name='count')


# Optional: calculate rate per minute (divide by 6)
counts['rate_per_min'] = counts['count'] / (6)
counts['rate_per_sec'] = counts['count'] / (6*60)


plt.figure(figsize=(8, 5))

for ch in counts['ch'].unique():
    df_ch = counts[counts['ch'] == ch]
    plt.plot(df_ch['timestamp'], df_ch['rate_per_sec'], label=f"Ch {ch}")

plt.xlabel('Time')
plt.ylabel('Rate [Hz]')
plt.title('Channel Activity Rate (15 min bins)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


# Group by channel
channels = df['ch'].unique()

print(df['dist_to_revsig'])
print(df.head())
# Plot
plt.figure(figsize=(8, 5))
        
plt.scatter(df[df['ch']==1]['dist_to_revsig_val'], df[df['ch']==1]['peak_heights_val'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(df[df['ch']==2]['dist_to_revsig_val'], df[df['ch']==2]['peak_heights_val'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(df[df['ch']==3]['dist_to_revsig_val'], df[df['ch']==3]['peak_heights_val'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('Distance to RevSig')
plt.ylabel('Peak Height')
plt.title('Peak Height vs Distance to RevSig')
plt.grid(True)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
        
plt.scatter(df[df['ch']==1]['timestamp'], df[df['ch']==1]['peak_heights_val'], color='blue', label=f"Ch 1", s=10, alpha=0.6)
plt.scatter(df[df['ch']==2]['timestamp'], df[df['ch']==2]['peak_heights_val'], color='red', label=f"Ch 2", s=10, alpha=0.6)
plt.scatter(df[df['ch']==3]['timestamp'], df[df['ch']==3]['peak_heights_val'], color='darkgreen', label=f"Ch 3", s=10, alpha=0.6)
plt.xlabel('time stamp')
plt.ylabel('Peak Height')
plt.title('Peak hights vs time')
plt.grid(True)
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()

# Flatten the dist_to_revsig values (assumes each cell is a list)
all_distances_1 = df[df['ch']==1]['dist_to_revsig_val'].explode().astype(float)
all_distances_2 = df[df['ch']==2]['dist_to_revsig_val'].explode().astype(float)
all_distances_3 = df[df['ch']==3]['dist_to_revsig_val'].explode().astype(float)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_distances_1, bins=2000, color='skyblue', edgecolor='blue', label=f"Ch 1", alpha=0.6)
plt.hist(all_distances_2, bins=2000, color='red', edgecolor='red'     , label=f"Ch 2", alpha=0.6)
plt.hist(all_distances_3, bins=2000, color='green', edgecolor='green' , label=f"Ch 3", alpha=0.6)
plt.xlabel('Distance to RevSig')
plt.ylabel('Frequency')
plt.title('Histogram of Distance to RevSig')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

