

import os
import numpy as np
import pandas as pd

# Path to list of directories
#list_files = ['list_2025-07-19.txt', 'list_2025-07-20.txt', 'list_2025-07-21.txt', ]
list_files = ['list_2025-07-22.txt']

# Output CSV file
#output_csvs = ['peaks_2025-07-19.csv', 'peaks_2025-07-20.csv', 'peaks_2025-07-21.csv']
output_csvs = ['peaks_2025-07-22.csv']

for list_file, output_csv in zip(list_files,output_csvs):
    # Storage for all waveforms
    all_rows = []

    # Read all directories from file
    with open(list_file, 'r', encoding='utf-16') as f:
        directories = [line.strip() for line in f if line.strip()]

    print('Number of directories: ', len(directories))
    dirN = 0
    # Loop over each directory
    for dir_path in directories:
        #print(dir_path)
        dirN+=1
        print(f'{dirN}/{len(directories)}')
        try:
            fileN=0
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    fileN+=1
                    #if fileN % 20 == 0:
                    #    print(f"files processed: {fileN}")
                    if (
                        entry.is_file()
                        and entry.name.startswith('peak')
                        and entry.name.endswith('.npz')
                    ):
                        try:
                            ch = 0
                            filepath = entry.name
                            if filepath.endswith('CH1.npz'):
                                ch=1
                            if filepath.endswith('CH2.npz'):
                                ch=2
                            if filepath.endswith('CH3.npz'):
                                ch=3

                            data = np.load(entry.path)
                            row = {
                                'ch'  :ch,  
                                'peak_indicies'  :data['peak_indicies'],  
                                'peak_heights'   :data['peak_heights'],
                                'widths'         :data['widths'], 
                                'dist_to_revsig' :data['dist_to_revsig'], 
                                'integrals'      :data['integrals'] ,       
                                'timestamp'      :data['timestamp'], 
                                'sample_rate'    :data['sample_rate'], 
                                'buffer_size'    :data['buffer_size'], 
                                'trigger_chn'    :data['trigger_chn'], 
                                'trigger_level'  :data['trigger_level']
                            }
                            all_rows.append(row)

                        except Exception as e:
                            print(f"⚠️ Failed to load {entry.path}: {e}")
        except Exception as e:
            print(f"⚠️ Cannot access directory {dir_path}: {e}")

    # Convert to DataFrame and save
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved CSV: {output_csv}")
    else:
        print("❌ No valid .npz files found.")



