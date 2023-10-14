import numpy as np
class Config:
    train_size = 0.8

    """ Explain param for calcualte mfcc feature
        _ sr : The sample rate of the audio signal, which specifies how many audio samples are captured per second (in Hz).
        _ n_mfcc: The number of MFCC coefficients to calculate. It determines the dimensionality of the MFCC feature vector.
        _ hop_length : The hop length is the number of samples between successive frames when dividing the audio signal into frames.
                              It affects the level of temporal detail in the MFCC features.
        _ n_fft : The number of samples in each FFT (Fast Fourier Transform) window. It determines the frequency resolution of the MFCCs.
    """
    sr = 22050
    n_mfcc = 13
    hop_length = 220
    n_frame = 13
    n_fft = 512

    state_dict = {
        'linh': 18,
        '9': 21,
        '6': 24,
        '5': 21,
        '2': 21,
        '7': 20,
        '3': 21,
        '4': 20,
        'mot': 16,
        '0': 22,
        'tram': 20,
        'tu': 24,
        '1': 19,
        'muoi': 15,
        '8': 21,
        'lam': 21,
        'trieu': 24,
        'nghin': 23,
        'm1': 16,
        'sil': 38
    }