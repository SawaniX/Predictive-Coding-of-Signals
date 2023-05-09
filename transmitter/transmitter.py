import numpy as np
from scipy import signal, linalg
import statsmodels.api as sm
import pickle
import bitstring
import matplotlib.pyplot as plt
from transmitter.prepare_data import PrepareData
from scipy.io import wavfile

class Transmitter:
    def __init__(self, data: np.ndarray, r: int):
        self.r = r
        self.data = data

        self.preparator = PrepareData(self.data, self.r)

    def compress(self):
        self.segments_with_noise, self.segments_raw = self.preparator.prepare()
        self._calculate_autocorrelation()
        #a = self._yule_walker(autocorr_coefficients)
        self._levinson_durbin()
        self._calculate_residual_errors()
        self._find_max_error()
        self._odtworz()
        #self._quantization()
        #self._save_file()

    def _odtworz(self):
        # bb = [0 for _ in range(250000)]
        # for idx in range(1, len(self.segments_raw)):
        #     for k in range(len(self.segments_raw[0])):
        #         yk = 0
        #         for i in range(1, self.r+1):
        #             if k - i > 0:
        #                 yk = yk + self.all_k[idx-1][i] * bb[k-i]
        #         bb[(idx-1) * 256 + k] = -yk + self.all_e[idx][k]
        # npa = np.asarray(bb[:240895], dtype=np.int16)
        # wavfile.write('rekon.wav', 11025, npa.astype(np.int16))
        # time = np.linspace(0., 240895 / 11025, 240895)
        # plt.plot(time, bb[:240895], label="Left channel")
        # plt.legend()
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        # plt.show()

        seg_len = len(self.segments_raw[0])
        segments_count = len(self.segments_raw)
        signal = []#10 * [0]
        y_prev = 10 * [0]
        for idx, segment in enumerate(self.segments_raw):
            for k in range(seg_len):
                yk = 0
                for i in range(1, self.r+1):
                    if k - i >= 0:
                        #print(i, k, k-i)
                        yk += self.all_k[idx][i] * y_prev[k-i]
                yk = -yk + self.all_e[idx][k]
                print(yk)
                if k < 246:
                    signal.append(yk)
                y_prev.append(yk)

        npa = np.asarray(signal, dtype=np.int16)
        wavfile.write('rekon3.wav', 11025, npa.astype(np.int16))
        time = np.linspace(0., len(signal) / 11025, len(signal))
        plt.plot(time, signal, label="Left channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()




        
    def _calculate_autocorrelation(self):
        self.autocorr_coefficients = []
        for segment in self.segments_with_noise:
            self.autocorr_coefficients.append(sm.tsa.acf(segment, nlags = 10))

    def _levinson_durbin(self):
        self.all_k = []
        for p in self.autocorr_coefficients:
            self.a = {}
            sigma = [p[0]]
            k = [1]     # tu czy 1?
            for i in range(1, len(p)):
                if i == 1:
                    ki = p[1] / p[0]        # tu moze minus
                else:
                    sigma.append((1 - k[i-1] * k[i-1]) * sigma[i-2])
                    ki = (p[i] - self._calc_sum(i, p, k)) / sigma[i-1]
                self.a[f'a{i}{i}'] = ki
                k.append(ki)
            self.all_k.append(k)

    def _calc_sum(self, i: int, p: list, k: list):
        sum = 0
        for j in range(1, i):
            if f'a{j}{i-1}' not in self.a:
                self.a[f'a{j}{i-1}'] = self.a[f'a{j}{i-2}'] - k[i-1] * self.a[f'a{i-1-j}{i-1-1}']
            sum += self.a[f'a{j}{i-1}'] * p[i-j]
        return sum

    def _calculate_residual_errors(self):
        self.all_e = []
        for segment, k in zip(self.segments_raw, self.all_k):
            e = []
            for i in range(len(segment)):
                error = 0
                for j in range(self.r+1):
                    if i-j >= 0:
                        error += segment[i-j] * k[j]
                e.append(error)
            self.all_e.append(e)

    def _find_max_error(self):
        self.max_errors = []
        for e in self.all_e:
            self.max_errors.append(max(e, key=abs))

    def _quantization(self, levels: int = 8):
        self.quants = []
        for idx, e in enumerate(self.all_e):
            step = (2 * self.max_errors[idx]) / (levels - 1)
            if step == 0:
                quant = np.floor(e)
            else:
                quant = np.floor(e/step)   # lub round
            self.quants.append(quant)

    def _save_file(self):
        pass
    