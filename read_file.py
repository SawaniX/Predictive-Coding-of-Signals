import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


class Data:
    def __init__(self):
        dir_path = os.path.dirname(__file__)
        samplerate, data = wavfile.read(dir_path + '/wiedzmin_1_channel.wav')
        self.samplerate = samplerate
        self.data = data.astype(float)
        self.plot()

    def plot(self):
        time = np.linspace(0., self.data.shape[0] / self.samplerate, self.data.shape[0])
        print(self.data.shape[0], self.samplerate)
        plt.plot(time, self.data, label="dsfdl")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        