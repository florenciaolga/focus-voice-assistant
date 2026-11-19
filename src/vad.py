import numpy as np

FRAME_LEN = 400
HOP_LEN = 160
ENERGY_MARGIN_DB = 10.0
MIN_SPEECH_MS = 150
HANGOVER_MS = 300


def frame_energy_db(x, frame_len=FRAME_LEN, hop_len=HOP_LEN):
    raise NotImplementedError


def frame_zcr(x, frame_len=FRAME_LEN, hop_len=HOP_LEN):
    raise NotImplementedError


def detect(x, sr=16000):
    raise NotImplementedError


class StreamingVAD:
    def __init__(self, sr=16000):
        self.sr = sr

    def process(self, chunk):
        raise NotImplementedError
