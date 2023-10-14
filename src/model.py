import os

import hmmlearn
from hmmlearn import hmm
import numpy as np

import pickle

# Test hyper-parameter for GMMHMM model
n_states = 8  # Number of hidden states (adjust as needed)
n_mix = 5  # Number of Gaussian distributions in each hidden state
startprobPrior = np.array([0.3, 0.3, 0.1, 0, 0, 0, 0, 0], dtype=np.float64)
tmp_p = 1.0 / 3
transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0, 0, 0, 0, 0],
                          [0, tmp_p, tmp_p, tmp_p, 0, 0, 0, 0],
                          [0, 0, tmp_p, tmp_p, tmp_p, 0, 0, 0],
                          [0, 0, 0, tmp_p, tmp_p, tmp_p, 0, 0],
                          [0, 0, 0, 0, tmp_p, tmp_p, tmp_p, 0],
                          [0, 0, 0, 0, 0, tmp_p, tmp_p, tmp_p],
                          [0, 0, 0, 0, 0, 0, 0.5, 0.5],
                          [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)

covariance_type = 'diag'  # Type of covariance matrix


class GMMHMMModel:
    def __init__(self):
        self.models = {}  # Dict to store HMMs for each label

    def add_model_for_label(self, label):
        self.models[label] = hmm.GMMHMM(n_components=n_states,
                                        n_mix=n_mix,
                                        covariance_type=covariance_type,
                                        transmat_prior=transmatPrior,
                                        startprob_prior=startprobPrior,
                                        n_iter=20)

    # Fit HMM model for label
    def fit_for_label(self, label, sequences, list_sequences_length):
        self.models[label].fit(X=sequences, lengths=list_sequences_length)

    def predict(self, sequences, list_sequences_length):
        for label in self.models.keys():
            self.models[label].predict(sequences, list_sequences_length)

    def save(self, save_path):
        model_name = 'speech_models.pkl'
        with open(os.path.join(save_path, model_name), "wb") as f:
            pickle.dump(self.models, f)
        print(f"Models saved as {model_name}")
