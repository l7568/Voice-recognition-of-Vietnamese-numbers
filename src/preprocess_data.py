import os

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import zscore
import librosa

from config import Config

c = Config()


def correct_label(df):
    df['label'].replace('lnh', 'linh', inplace=True)
    df['label'].replace('trsm', 'tram', inplace=True)
    df['label'].replace('sli', 'sil', inplace=True)
    df['label'].replace('tram ', 'tram', inplace=True)
    df['label'].replace('3   ', '3', inplace=True)
    df['label'].replace('9  ', '9', inplace=True)
    df['label'].replace(' tram', 'tram', inplace=True)
    df['label'].replace('nam', '5', inplace=True)
    df['label'].replace('ngin', 'nghin', inplace=True)
    df['label'].replace('nghin ', 'nghin', inplace=True)
    df['label'].replace('8\\', '8', inplace=True)
    df.dropna(subset=["label"], inplace=True)
    return df


def compute_mfcc_feature(df):
    def extract_mfcc_feature(x, sound, sr, hop_length):
        s, e = int(np.floor(x.loc["start"] * sr)), int(np.ceil(x.loc["end"] * sr))

        # Extract features
        mfcc = librosa.feature.mfcc(y=sound[s:e], sr=c.sr, n_mfcc=c.n_mfcc,
                                    hop_length=c.hop_length, n_fft=c.n_fft)
        delta = librosa.feature.delta(mfcc, width=3)
        delta_2 = librosa.feature.delta(mfcc, order=2, width=3)

        # Normalize z-scores for each type of feature
        mfcc = zscore(mfcc, axis=1)  # shape (13, n_frame)
        delta = zscore(delta, axis=1)  # shape (13, n_frame)
        delta_2 = zscore(delta_2, axis=1)  # shape (13, n_frame)

        # Combine the features into a comprehensive feature vector
        combined_features = np.vstack((mfcc, delta, delta_2))  # shape (39, n_frame)
        return combined_features.T  # shape (n_frame, 39)

    fids = df['fid'].unique()
    for fid in fids:
        sound_file_path = os.path.join(fid + ".wav")
        # Optimize calculation speed by read files only once
        sound, sr = librosa.load(sound_file_path)
        dfi = df[df["fid"] == fid]
        # Calculate features
        df.loc[df["fid"] == fid, "mfcc_norma"] = dfi.apply(extract_mfcc_feature,
                                                           args=(sound, c.sr, c.hop_length), axis=1)
    return df


def reduce_mfccs_state(df):
    def kmean_reduce(mfcc_norma, label, state_dict):
        if label == 'sil':
            return mfcc_norma

        n_state, b = mfcc_norma.shape
        if n_state > state_dict[label]:
            n_state = state_dict[label]

        # Assign labels to (adjacent) frames using KMeans
        clustering = KMeans(n_clusters=n_state, n_init=12, random_state=0)
        clustering.fit(mfcc_norma)
        state = clustering.labels_

        # Group frames with the same label (adjacent)
        same_states = []
        curr_idx = 0
        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                same_states.append(np.arange(curr_idx, i))
                curr_idx = i

        same_states.append(np.arange(curr_idx, len(state)))
        # Merge identical frames: mean value
        mfcc_feat = [np.mean(mfcc_norma[arr], axis=0) for arr in same_states]

        return np.array(mfcc_feat)

    df['mfcc_feat'] = df.apply(lambda x:
                               kmean_reduce(x.mfcc_norma, x.label, c.state_dict), axis=1)
    return df
