class ASR:
    def __init__(self, size="tiny"):
        self.size = size

    def transcribe(self, audio):
        raise NotImplementedError
