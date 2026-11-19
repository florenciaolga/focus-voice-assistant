import numpy as np

from . import config

PRE_EMPHASIS = 0.97
FRAME_LEN = 400
HOP_LEN = 160
N_FFT = 512
N_MELS = 40
N_MFCC = 13
DELTA_N = 2


def pre_emphasis(x, coef=PRE_EMPHASIS):
    raise NotImplementedError


def frame_signal(x, frame_len=FRAME_LEN, hop_len=HOP_LEN):
    raise NotImplementedError


def power_spectrum(frames, n_fft=N_FFT):
    raise NotImplementedError


def mel_filterbank(sr=config.SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    raise NotImplementedError


def logmel(x, sr=config.SAMPLE_RATE):
    raise NotImplementedError


def dct2(x, n_out=N_MFCC):
    raise NotImplementedError


def delta(feat, n=DELTA_N):
    raise NotImplementedError


def cmvn(feat):
    raise NotImplementedError


def mfcc(x, sr=config.SAMPLE_RATE):
    raise NotImplementedError
