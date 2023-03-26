import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from receiver import Receiver
from transmitter import Transmitter


class Data:
    def __init__(self):
        dir_path = os.path.dirname(__file__)
        samplerate, data = wavfile.read(dir_path + '/wiedzmin.wav')
        self.samplerate = samplerate
        self.data = data

    def plot(self):
        time = np.linspace(0., self.data.shape[0] / self.samplerate, self.data.shape[0])
        plt.plot(time, self.data[:, 0], label="Left channel")
        plt.plot(time, self.data[:, 1], label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    
if __name__=='__main__':
    data = Data()

    transmitter = Transmitter(data)

    receiver = Receiver()
