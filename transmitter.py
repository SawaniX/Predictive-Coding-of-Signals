import numpy as np


class Transmitter:
    def __init__(self, data: np.ndarray):
        print(type(data))
        self.data = data
        self.N = 256
        self.r = 10

    def compress(self):
        words = self._split_to_segments()

    def _split_to_segments(self):
        words = []
        frm = 0
        to = self.N
        while to < self.data.shape[0]:
            words.append(self.data[frm:to])

            frm = to - self.r
            to = frm + self.N
        return words

    def _model_signal(self):
        pass

    def _levinson_durbin(self):
        pass

    def _calculate_residual_errors(self):
        pass

    def _save_file(self):
        pass
    