import numpy as np
import pandas as pd

import os
import re

from coactivation.dataset_definition import left_right_hand_dict
from coactivation.process import preprocess_emgs, get_coactivation_map, emgs

# get coactivations of every single session into one large dataset
data_dir = 'collected_data'
columns = np.append(np.char.add(np.array(['gt0','gt1','gt2'])[:,None], emgs).flatten(), ['hand','is_patient', 'subject_id','n'])
df_healthy = pd.DataFrame(columns=columns)
for root, _, files in os.walk(data_dir):
    # print(root)
    if root != data_dir and root[15] == '2':
        # print(f"\nIn subdirectory: {root}")
        if (int(root[15:19]) > 2022) and files: 
            subject_id = root.split('/')[-1].split('_')[-1]
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
                    coactivation_map = get_coactivation_map(df_preprocessed)
                    values = coactivation_map.to_numpy().flatten()
                    if len(values)!=24:
                        continue
                    df_to_concat = pd.DataFrame([np.append([values], [hand, is_patient, subject_id, n])], columns=columns)
                    df_healthy = pd.concat([df_healthy, df_to_concat])
                    # display(df_healthy)
df_healthy.to_csv('coactivations.csv')