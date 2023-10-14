import pandas as pd

from dataset import load_data, buildDataDictForMFCC
from preprocess_data import compute_mfcc_feature, reduce_mfccs_state, correct_label

from model import GMMHMMModel

from train import train_GMMHMM
from test import test_GMMHMM

from config import Config

c = Config()


def main():
    # Prepare audio data, calculate feature
    df = load_data()
    df = correct_label(df)
    df = compute_mfcc_feature(df)
    df = reduce_mfccs_state(df)

    # Init model, format data for train/test phases
    SpeechModel = GMMHMMModel()
    trainDataDict, testDataDict = buildDataDictForMFCC(df, c.train_size)

    # Training phase
    SpeechModel = train_GMMHMM(SpeechModel, trainDataDict)

    # Testing phase
    test_GMMHMM(SpeechModel, testDataDict)

    # Save trained model
    SpeechModel.save(save_path='../output')


if __name__ == '__main__':
    main()
