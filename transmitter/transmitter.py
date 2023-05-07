import numpy as np
from scipy import signal, linalg
import statsmodels.api as sm

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
        print(self.all_k[1])
        print()
        print()
        si, ar, _, _, _ = sm.tsa.stattools.levinson_durbin(self.autocorr_coefficients[1], nlags=10, isacov=True)
        print(ar)

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
                e.append(sum([a*b for a,b in zip(k, segment[i:self.r+1])]))
            self.all_e.append(e)

    def _find_max_error(self):
        self.max_errors = []
        for e in self.all_e:
            self.max_errors.append(max(e, key=abs))

    def _quantization(self, levels: int):
        pass

    def _save_file(self):
        pass
    