import os
import pandas as pd
import numpy as np
import re

from coactivation.dataset_definition import left_right_hand_dict
from generate_viz import preprocess_data, df_to_coactivation_map, plot_coactivation_map
 
# one df per hand for healthy people

# iterate through

df_right, df_left = pd.DataFrame(), pd.DataFrame()

df_dict = {
    ('right'): df_right,
    ('left'): df_left,
}

unknown = []
hand = ''

data_dir = 'collected_data'
for root, subdirs, files in os.walk(data_dir):
    print(root)
    if root != data_dir and root[15] == '2':
        print(f"\nIn subdirectory: {root}")
        if (int(root[15:19]) > 2022) and files: 
            print("Files:")
            subject_id = root.split('/')[-1].split('_')[-1]
            if subject_id not in left_right_hand_dict:
                hand = 'unknown'
                unknown.append(subject_id)
                continue
            elif bool(re.match(r'^p\d+$',subject_id)):
                # is patient, skip
                continue
            else:
                hand = left_right_hand_dict[subject_id]

            for file in files:
                if file[-4:] == '.csv':
                    print(file)
                    # extract 11 or 13
                    n = file[-7:5]
                    if n not in ['11','13']:
                        continue
                    
                    # get main df to append to
                    df_main = df_dict[hand]

                    # get map of individual dataset
                    path = os.path.join(f"{root}", file)
                    print(path)
                    df = preprocess_data(path)
                    cm = df_to_coactivation_map(df, metric='var')
                    cm_melted = pd.melt(cm.reset_index(), id_vars=['gt'])

                    # merge to master dataset to average later
                    if df_main.empty:
                        df_dict[hand] = cm_melted  # Update directly in the dictionary
                    else:
                        df_dict[hand] = pd.merge(df_main, cm_melted, on=['gt', 'emg'])

# final mean operation
for key, final_df in df_dict.items():
    final_df = pd.DataFrame(final_df.set_index(['gt','emg']).mean(axis=1)).reset_index().pivot(index='gt', columns='emg', values=0)
    final_df.to_csv(key + 'var.csv')