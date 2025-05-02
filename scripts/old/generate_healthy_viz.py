import os
import pandas as pd
import numpy as np
import re

from coactivation.dataset_definition import left_right_hand_dict
from generate_viz import preprocess_data, df_to_coactivation_map, plot_coactivation_map
 
# create six dfs of flattened coactivation map df: (right, left)  x  (11, 12, 13)

# iterate through

df_r11, df_r12, df_r13, df_l11, df_l12, df_l13 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_dict = {
    ('right', '11'): df_r11,
    ('right', '13'): df_r13,
    ('left', '11'): df_l11,
    ('left', '13'): df_l13,
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
                if left_right_hand_dict[subject_id] == "left":
                    hand = 'left'
                else: 
                    hand = 'right'

            for file in files:
                if file[-4:] == '.csv':
                    print(file)
                    # extract 11 or 13
                    n = file[-7:5]
                    if n not in ['11','13']:
                        continue
                    
                    # get main df to append to
                    df_main = df_dict.get((hand,n))

                    # get map of individual dataset
                    path = os.path.join(f"{root}", file)
                    print(path)
                    df = preprocess_data(path)
                    cm = df_to_coactivation_map(df, metric='std')
                    cm_melted = pd.melt(cm.reset_index(), id_vars=['gt'])

                    # merge to master dataset to average later
                    if df_main.empty:
                        df_dict[(hand, n)] = cm_melted  # Update directly in the dictionary
                    else:
                        df_dict[(hand, n)] = pd.merge(df_main, cm_melted, on=['gt', 'emg'])

# final mean operation
for key, final_df in df_dict.items():
    final_df = pd.DataFrame(final_df.set_index(['gt','emg']).mean(axis=1)).reset_index().pivot(index='gt', columns='emg', values=0)
    final_df.to_csv(key[0] + key[1] + '.variance.csv')
                    
# save to csvs
# df_r11.to_csv('df_r11.csv')
# df_r13.to_csv('df_r13.csv')
# df_l11.to_csv('df_l11.csv')
# df_l13.to_csv('df_l13.csv')
