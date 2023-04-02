import numpy as np

from transmitter.prepare_data import PrepareData


class Transmitter:
    def __init__(self, data: np.ndarray):
        self.data = data
        
        self.preparator = PrepareData(self.data)

    def compress(self):
        segments = self.preparator.prepare()
        

    def _model_signal(self):
        pass

    def _levinson_durbin(self):
        pass

    def _calculate_residual_errors(self):
        pass

    def _save_file(self):
        pass
    