from config import Config

c = Config()


def train_GMMHMM(SpeechModel, dataDict):
    for label, data in dataDict.items():
        SpeechModel.add_model_for_label(label)
        sequences, seq_lengths = data
        SpeechModel.fit_for_label(label, sequences, seq_lengths)

    return SpeechModel
