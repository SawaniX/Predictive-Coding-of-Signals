import numpy as np

from transmitter.prepare_data import PrepareData


class Transmitter:
    def __init__(self, data: np.ndarray):
        self.data = data

        self.preparator = PrepareData(self.data)

    def compress(self):
        segments = self.preparator.prepare()
        self._levinson_durbin(segments)
        

    def _model_signal(self):
        pass

    def _levinson_durbin(self, segments: list):
        p = []
        for segment in segments:
            part = []
            for i, sample in enumerate(segment, 1):
                pi = 0
                for t in range(i+1, len(segment)):
                    pi += segment[t] * segment[t-i]
                part.append(pi)
            p.append(part)

    def _calculate_residual_errors(self):
        pass

    def _save_file(self):
        pass
    