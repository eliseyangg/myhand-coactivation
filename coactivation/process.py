import pandas as pd
import numpy as np

import os
import re

import warnings
warnings.simplefilter('error', FutureWarning)


emgs = np.char.add('emg', np.arange(0, 8).astype(str))

left_right_hand_dict = {
    "hz": "right",
    "jx": "right",
    "av": "right",
    "ec": "right",
    "ac": "right",
    "ts": "right",
    "cl": "right",
    "dy": "right",
    "fa": "left",
    "wx": "left",
    "gk": "right",
    "jp": "right",
    "xw": "right",
    "yc": "right",
    "ae": "right",
    "gk2": "right",
    "jo": "left",
    "si": "left",
    "as": "left",
    "im": "left",
    # hr changed
    "hr": "right",
    "is": "right",
    "p1": "right",
    "p3": "left",
    "p4": "right",
    "p7": "right",
    "p8": "right",
    "p10": "left",
    "p12": "right",
    "p13": "left",
    # Some patient has identifiers
    "rt": "right", #p4
    "dm": "right", #p1
    "av": "left", #p3
    # multimodal-ctrl identifiers
    "multimodal-ctrl-004": "right", #p4
    "multimodal-ctrl-005": "right", #p8
    "multimodal-ctrl-006": "right" #p7
}


def generate_coactivation_by_gt(data_dir = 'collected_data'):
    '''Generates a dataset of coactivations by gt window for each session for 200hz data 
    
    returns:
    -------
    pd.DataFrame()
        cols: emg0-emg7, gt, window, date, subject_id, set_num, subset, hand, is_patient
    '''

    cols = np.append(['folder','subject_id','hand','set_num','subset','is_patient','gt','window'], emgs)
    l = len(data_dir) + 1
    df_list = []
    for root, _, files in os.walk(data_dir):
        if root != data_dir and root[l] == '2':
            date = int(root[l:l+4] + root[l+5:l+7] + root[l+8:l+10])
            if date >= 20250205 and date != 20250401: 
                print(root)
                folder = root.split('/')[-1]
                if len(folder) > 10:
                    continue
                for file in files:
                    print(file)
                    if file[-4:] == '.csv':
                        try: 
                            subject_id, set_num, subset = re.match(r'^([^_]+)_(.*)_(.+)$', file[:-4]).groups()
                            is_patient = bool(re.match(r'^p\d+$',subject_id))
                            if subject_id not in left_right_hand_dict:
                                hand = np.nan
                            else:
                                hand = left_right_hand_dict[subject_id]

                            # emg coactivation extraction
                            df_preprocessed = preprocess_emgs(os.path.join(f"{root}", file), hz=200)
                            df_to_concat = get_stratified_cm(df_preprocessed)
                            len_df = len(df_to_concat)

                            # add metadata
                            df_to_concat['folder'] = np.repeat(folder, len_df).astype(str)
                            df_to_concat['subject_id'] = np.repeat(subject_id, len_df).astype(str)
                            df_to_concat['set_num'] = np.repeat(set_num, len_df).astype(str)
                            df_to_concat['subset'] = np.repeat(subset, len_df).astype(str)
                            df_to_concat['subset'] = np.where(df_to_concat['set_num'] == 'grasp', 'grasping', df_to_concat['set_num'])
                            df_to_concat['subset'] = np.where(df_to_concat['set_num'] == 'wrist', 'wrist_movement', df_to_concat['set_num'])
                            df_to_concat['hand'] = np.repeat(hand, len_df).astype(str)
                            df_to_concat['is_patient'] = np.repeat(is_patient, len_df).astype(bool)
                            
                            df_list.append(df_to_concat[cols])

                        except: # doesn't follow format
                            continue

    df = pd.concat(df_list, ignore_index=True)
    return df

def generate_coactivation_dataset(data_dir = 'collected_data'):
    '''generates a dataset of coactivations across all sessions

    returns:
    -------
    pd.DataFrame
        cols: gt0emg0-gt2emg7, folder, subject_id, hand, set_num, subset, is_patient
    '''
    l = len(data_dir) + 1
    columns = np.append(np.char.add(np.array(['gt0','gt1','gt2'])[:,None], emgs).flatten(), ['folder','subject_id','hand','set_num','subset','is_patient'])
    df = pd.DataFrame(columns=columns)
    for root, _, files in os.walk(data_dir):
        # print(root)
        if root != data_dir and root[l] == '2':
            date = int(root[l:l+4] + root[l+5:l+7] + root[l+8:l+10])
            if date >= 20250202 and date != 20250401:  
                folder = root.split('/')[-1]
                if len(folder) > 10:
                    continue

                for file in files:
                    if file[-4:] == '.csv':
                        try:
                            subject_id, set_num, subset = re.match(r'^([^_]+)_(.*)_(.+)$', file[:-4]).groups()
                            is_patient = bool(re.match(r'^p\d+$',subject_id))
                            if subject_id not in left_right_hand_dict:
                                hand = np.nan
                            else:
                                hand = left_right_hand_dict[subject_id]

                            df_preprocessed = preprocess_emgs(os.path.join(f"{root}", file), hz=200)
                            coactivation_map = get_coactivation_map(df_preprocessed)
                            values = coactivation_map.to_numpy().flatten()
                            if len(values)!=24:
                                continue
                            df_to_concat = pd.DataFrame([np.append([values], [folder, subject_id, hand, set_num, subset, is_patient])], columns=columns)
                            df = pd.concat([df, df_to_concat])

                        except: # doesn't follow format
                            continue

    for emg in np.char.add(np.array(['gt0','gt1','gt2'])[:,None], emgs).flatten():
        df[emg] = df[emg].astype(float)

    return df


def preprocess_emgs(path, hz = 200):
    '''
    process emg data
    
    parameters
    ----------
    path : str
        path to .csv file of emg data

    returns
    -------
    df : pd.DataFrame
    '''
    df_preprocessed = pd.read_csv(path, index_col=0)
    if df_preprocessed.columns[1] != 'gt':
        df_preprocessed = pd.read_csv(path)
    if hz == 50:
        df_preprocessed[emgs] = df_preprocessed[emgs] / df_preprocessed[emgs].max(axis=0)
    else:
        df_preprocessed[emgs] = np.abs(df_preprocessed[emgs])
        df_preprocessed[emgs] = df_preprocessed[emgs] / 128

    return df_preprocessed

def get_coactivation_map(df, metric='median'):
    '''
    compute coactivation map from preprocessed data

    parameters
    ----------
    df : pd.DataFrame
        preprocessed emg dataframe with columns ['gt'] + emg0-7
    metric : str or callable
        aggregation metric applied to each group

    returns
    -------
    pd.DataFrame
        pivoted coactivation map indexed by gt, columns emg0-7
    '''
    df_melt = df[np.append(['gt'], emgs)].melt(id_vars='gt',value_vars=emgs,var_name='emg').groupby(['gt','emg']).apply(metric).reset_index()
    coactivation_map = df_melt.pivot(index='gt', columns='emg', values='value')
    return coactivation_map

def cm_to_array(cm):
    '''flatten a coactivation matrix into a 1d numpy array'''
    return cm.values.flatten()

def array_to_cm(array):
    '''reshape a flat array into a coactivation matrix dataframe'''
    cm = pd.DataFrame(array.reshape(3, 8), index=[0, 1, 2], columns=[f'emg{i}' for i in range(8)])
    cm.index.name = 'gt'
    return cm

def get_stratified_cm(df_preprocessed):
    '''compute stratified coactivation maps by window and gt'''
    df_preprocessed['window'] = np.cumsum(np.diff(df_preprocessed['gt'], prepend=0)!=0)
    cm_stratified = df_preprocessed[np.append(emgs, ['window','gt'])].groupby(['window','gt']).median().reset_index()
    return cm_stratified

def get_all_healthy_map(dfgt=pd.DataFrame(), metric='median', data_dir='collected_data'):
    '''compute the average healthy coactivation map grouped by gt'''
    if dfgt.empty:
        dfgt = generate_coactivation_by_gt(data_dir)
    return mirror_emg(dfgt)[np.append(emgs, 'gt')].groupby('gt').apply(metric)


_MIRROR_IDX = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 0, 7: 7}

def mirror_emg(df: pd.DataFrame,
               hand_col: str = 'hand',
               left_label: str = 'left') -> pd.DataFrame:
    '''
    mirror emg channels horizontally for left-hand rows
    
    parameters
    ----------
    df : pd.DataFrame
        input dataframe with either emg0-7 or gt#emg0-7 columns
    hand_col : str
        column indicating left or right hand label
    left_label : str
        value in hand_col to trigger mirroring

    returns
    -------
    pd.DataFrame
        dataframe with mirrored emg columns for left-hand rows
    '''
    df_out = df.copy()
    mask = df_out[hand_col] == left_label

    single_cols = [f'emg{i}' for i in range(8)]
    multi_cols  = [f'gt{g}emg{i}' for g in range(3) for i in range(8)]

    if set(single_cols).issubset(df_out.columns):
        tmp = df_out.loc[mask, single_cols].copy()
        for tgt_i, src_i in _MIRROR_IDX.items():
            tgt, src = f'emg{tgt_i}', f'emg{src_i}'
            df_out.loc[mask, tgt] = tmp[src]

    elif set(multi_cols).issubset(df_out.columns):
        for g in range(3):
            emg_cols = [f'gt{g}emg{i}' for i in range(8)]
            tmp = df_out.loc[mask, emg_cols].copy()
            for tgt_i, src_i in _MIRROR_IDX.items():
                tgt = f'gt{g}emg{tgt_i}'
                src = f'gt{g}emg{src_i}'
                df_out.loc[mask, tgt] = tmp[src]

    return df_out


from coactivation.similarity import mi_cm, g_cm, gmi_cm, ruzicka_cm

def get_similarity_metric(cm1, cm2, metric='ruzicka'):
    '''compare two coactivation maps using specified similarity metrics'''
    if metric == 'ruzicka':
        return ruzicka_cm(cm1, cm2)
    if metric == 'g':
        return g_cm(cm1, cm2)
    if metric == 'mi':
        return mi_cm(cm1, cm2)
    if metric =='gmi':
        return gmi_cm(cm1, cm2)

def apply_similarity_metric(x, comparison_cm, metric='ruzicka'):
    return (get_similarity_metric(array_to_cm(x.values), comparison_cm, metric=metric)[1])


#======= Discard Pile ========#
'''
def get_coactivation_bool_map(cm, threshold=0.45):
    return cm/np.max(cm) > threshold

def get_alignment_map(bm1, bm2):
    alignment_map = pd.DataFrame(0, index=bm1.index, columns=bm1.columns) 

    conditions = [
        (bm1 == 0) & (bm2 == 0),  # both exhibit no activations
        (bm1 == 1) & (bm2 == 1),  # both exhibit an activation
        (bm1 == 0) & (bm2 == 1),  # bm2 exhibits an activation but bm1 does not
        (bm1 == 1) & (bm2 == 0)   # bm2 doesn't exhibit an activation but bm1 does
    ]
    choices = [0, 1, 2, 3]
    alignment_map[:] = np.select(conditions, choices, default=np.nan)
    return alignment_map
'''
#==============================#