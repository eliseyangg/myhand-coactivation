import numpy as np
import pandas as pd

import os
import re

from coactivation.dataset_definition import left_right_hand_dict
from coactivation.process import preprocess_emgs, get_stratified_cm, get_coactivation_map, emgs

import warnings
warnings.simplefilter('error', FutureWarning)

# get coactivations of every single gt window from each session into one large dataset
data_dir = 'collected_data'
columns = np.append(emgs, ['gt','hand','is_patient','subject_id','date','n','window'])
df_list = []
for root, _, files in os.walk(data_dir):
    # print(root)
    if root != data_dir and root[15] == '2':
        # print(f"\nIn subdirectory: {root}")
        if (int(root[15:19]) > 2022) and files: 
            print(root)
            subject_id = root.split('/')[-1].split('_')[-1]
            if subject_id == 'Augmen':
                continue
            date = (root.split('/')[-1])[:10]
            is_patient = bool(re.match(r'^p\d+$',subject_id))
            if subject_id not in left_right_hand_dict:
                hand = 'unknown'
            else:
                hand = left_right_hand_dict[subject_id]
            for file in files:
                if file[-4:] == '.csv':
                    n = file[-7:5]
                    if n not in ['11','12','13']:
                        continue
                    df_preprocessed = preprocess_emgs(os.path.join(f"{root}", file))
                    # df_preprocessed['window'] = np.cumsum(np.diff(df_preprocessed['gt'], prepend=0)!=0)
                    # df_to_concat = df_preprocessed[np.append(emgs, ['window','gt'])].groupby(['window','gt']).median().reset_index()
                    df_to_concat = get_stratified_cm(df_preprocessed)
                    
                    len_df = len(df_to_concat)
                    df_to_concat['hand'] = np.repeat(hand, len_df).astype(str)
                    df_to_concat['is_patient'] = np.repeat(is_patient, len_df).astype(bool)
                    df_to_concat['subject_id'] = np.repeat(subject_id, len_df).astype(str)
                    df_to_concat['date'] = np.repeat(date, len_df).astype(str)
                    df_to_concat['n'] = np.repeat(n, len_df).astype(str)
                    
                    df_list.append(df_to_concat)

df_healthy = pd.concat(df_list, ignore_index=True)
df_healthy.to_csv('coactivations_by_gt.csv')