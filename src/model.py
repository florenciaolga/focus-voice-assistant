class Model1:
    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def predict(self, audio):
        raise NotImplementedError
