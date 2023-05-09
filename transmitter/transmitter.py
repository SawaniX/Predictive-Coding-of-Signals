import numpy as np
from scipy import signal, linalg
import statsmodels.api as sm
import pickle
import bitstring

from transmitter.prepare_data import PrepareData

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
        #self._quantize_uniform(self.all_e[1], -self.max_errors[1], self.max_errors[1], 8)
        self._quantization()
        #self._save_file()
        
    def _calculate_autocorrelation(self):
        self.autocorr_coefficients = []
        # for segment in self.segments_with_noise:
        #     N = len(segment)
        #     segment_coefficients = [1]
        #     for row in range(1, self.r+1):
        #         p = 0
        #         for t in range(row+1, N):
        #             p += segment[t] * segment[t-1]
        #         R = p / N
        #         segment_coefficients.append(p)
        #     self.autocorr_coefficients.append(segment_coefficients)

        for segment in self.segments_with_noise:
            self.autocorr_coefficients.append(sm.tsa.acf(segment, nlags = 10))
    
    # def _yule_walker(self, autocorr_coefficients: list):
    #     a = []
    #     for p in autocorr_coefficients:
    #         R = linalg.toeplitz(p[:self.r])
    #         r = p[1:self.r+1]
    #         a.append(linalg.inv(R).dot(r))
    #     return a

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

        #print(self.autocorr_coefficients[1])
        # print(self.all_k[1])
        # print()
        # print()
        # si, ar, _, _, _ = sm.tsa.stattools.levinson_durbin(self.autocorr_coefficients[1], nlags=10, isacov=True)
        # print(ar)

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

    def quantize(self, data, levels, maxe):
        max_value = maxe
        min_value = -maxe
        step = (max_value - min_value) / levels
        return np.round((data - min_value) / step) * step + min_value

    def _quantization(self, levels: int = 8):
        levels = 8

        self.quants = []
        for idx, e in enumerate(self.all_e):
            quantized_data = self.quantize(e, levels, self.max_errors[idx])
            print(quantized_data)
            self.quants.append(quantized_data)
        # self.quants = []
        # for idx, e in enumerate(self.all_e):
        #     step = (2 * self.max_errors[idx]) / (levels - 1)
        #     if step == 0:
        #         quant = np.floor(e)
        #     else:
        #         quant = np.floor(e/step)   # lub round
        #     self.quants.append(quant)

    def _save_file(self):
        bity = []
        for e in self.quants[1]:
            bit = bin(int(e))
            print(bit, e)
        # import csv

        # with open("a.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.all_k)
        # rows = [[data] for data in self.max_errors]
        # with open("emax.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(rows)
        # with open("quant.csv", "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(self.quants)

        # with open('a.bin', 'wb') as f:
        #     f.write(np.array(self.all_k).tobytes())
        # with open('emax.bin', 'wb') as f:
        #     f.write(np.array(self.max_errors).tobytes())
        # with open('quants.bin', 'wb') as f:
        #     f.write(np.array(self.quants).tobytes())



        # with open('a.pickle', 'wb') as handle:
        #     pickle.dump(self.all_k, handle)
        # with open('emax.pickle', 'wb') as handle:
        #     pickle.dump(self.max_errors, handle)
        # with open('quants.pickle', 'wb') as handle:
        #     pickle.dump(self.quants, handle)
    