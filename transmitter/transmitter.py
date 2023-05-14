import numpy as np
from scipy import signal, linalg
import statsmodels.api as sm
import pickle
import bitstring
import struct
import matplotlib.pyplot as plt
from transmitter.prepare_data import PrepareData
from scipy.io import wavfile
import math

class Transmitter:
    def __init__(self, data: np.ndarray, r: int, bits: int):
        self.r = r
        self.data = data
        self.bits = bits
        self.quant_level = 2**self.bits

        self.preparator = PrepareData(self.data, self.r)

    def compress(self):
        self.segments_with_noise, self.segments_raw = self.preparator.prepare()
        self._calculate_autocorrelation()
        self._levinson_durbin()
        #self._calculate_residual_errors()
        #self._find_max_error()
        #self._quantization()
        #self._save_file()
        #self._odtworz()
        self._test()
        self._rysowanie()
        
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

        # for p in self.segments_with_noise:
        #     _, s, _, _, _ = sm.tsa.stattools.levinson_durbin(p, nlags=10, isacov=False)
        #     c = [1]
        #     c.extend(s)
        #     self.all_k.append(c)
        x = [b for c in self.all_k for b in c]
        print(max(x))
        print(min(x))

    def _calc_sum(self, i: int, p: list, k: list):
        sum = 0
        for j in range(1, i):
            if f'a{j}{i-1}' not in self.a:
                self.a[f'a{j}{i-1}'] = self.a[f'a{j}{i-2}'] - k[i-1] * self.a[f'a{i-1-j}{i-1-1}']
            sum += self.a[f'a{j}{i-1}'] * p[i-j]
        return sum

    def _calculate_residual_errors(self):
        self.segmenty_bez_zakladki = []
        for idx, segment in enumerate(self.segments_raw):
            if idx == 0:
                self.segmenty_bez_zakladki.append(segment)
            else:
                self.segmenty_bez_zakladki.append(segment[10:])

        self.all_e = []
        for idx, (segment, k) in enumerate(zip(self.segmenty_bez_zakladki, self.all_k)):
            e = []
            for i in range(len(segment)):
                error = 0
                for j in range(self.r+1):
                    if idx == 0:
                        if i-j >= 0:
                            error += segment[i-j] * k[j]
                    else:
                        if i-j >= 0:
                            error += segment[i-j] * k[j]
                        else:
                            error += self.segmenty_bez_zakladki[idx-1][-j] * k[j]
                e.append(error)
            self.all_e.append(e)



        # self.all_e = []
        # for segment, k in zip(self.segments_raw, self.all_k):
        #     e = []
        #     for i in range(len(segment)):
        #         error = 0
        #         for j in range(self.r+1):
        #             if i-j >= 0:
        #                 error += segment[i-j] * k[j]
        #         e.append(error)
        #     self.all_e.append(e)

    def _find_max_error(self):
        self.max_errors = []
        for e in self.all_e:
            self.max_errors.append(max(e, key=abs))

    def _quantization(self):
        self.quants = []
        for idx, e in enumerate(self.all_e):
            e_plus = [x+self.max_errors[idx] for x in e]
            step = (2 * self.max_errors[idx]) / (self.quant_level - 1)
            if step == 0:
                quant = np.round(e_plus)
            else:
                quant = np.round(e_plus/step)   # lub round
            self.quants.append(quant)

    def _save_file(self):
        a_flat = [i for a in self.all_k for i in a]
        a_struct = struct.pack('f'*len(a_flat), *a_flat)
        with open(f'a{str(self.bits)}.bin', 'wb') as f:
            f.write(a_struct)

        emax_struct = struct.pack('f'*len(self.max_errors), *self.max_errors)
        with open(f'emax{str(self.bits)}.bin', 'wb') as f:
            f.write(emax_struct)

        quants_flat = [i for quant in self.quants for i in quant]
        bit_lista = bitstring.BitArray()
        for liczba in quants_flat:
            bit_lista += bitstring.BitArray(uint=int(liczba), length=self.bits)
        bajty = bit_lista.tobytes()
        with open(f'quants{str(self.bits)}.bin', 'wb') as plik:
            plik.write(bajty)

    def _test(self):
        self.segmenty_bez_zakladki = []
        for idx, segment in enumerate(self.segments_raw):
            if idx == 0:
                self.segmenty_bez_zakladki.append(segment)
            else:
                self.segmenty_bez_zakladki.append(segment[10:])
        
        e = []
        for idx, segment in enumerate(self.segmenty_bez_zakladki):
            e_seg = []
            for ind, yk in enumerate(segment):
                ek = 0
                for i in range(11):
                    if ind - i >= 0:
                        ek += self.all_k[idx][i] * segment[ind - i]
                    elif ind - i < 0 and idx > 0:
                        ek += self.all_k[idx][i] * self.segmenty_bez_zakladki[idx-1][-i]
                e_seg.append(ek)
            e.append(e_seg)

        odtw = []
        for idx, ee in enumerate(e):
            odtw_seg = []
            for ind, eee in enumerate(ee):
                yk = 0
                for i in range(1, 11):
                    if ind - i >= 0:
                        yk += self.all_k[idx][i] * odtw_seg[ind-i]
                    elif ind - i < 0 and idx > 0:
                        yk += self.all_k[idx][i] * odtw[idx-1][-i]
                yk = eee - yk/10
                odtw_seg.append(yk)
            odtw.append(odtw_seg)

        o = [x for a in odtw for x in a]
        #o = o[10000:13000]
        npa = np.asarray(o, dtype=np.int16)
        wavfile.write('absb.wav', 11025, npa.astype(np.int16))
        time = np.linspace(0., len(o)/11025, len(o))
        plt.plot(time, o, label="Odtworzony")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")


    def _odtworz(self):
        sig = 10 * [0]
        for idx, segment in enumerate(self.segmenty_bez_zakladki):
            for k in range(len(segment)):
                yk = 0
                for i in range(1, self.r+1):
                    yk += self.all_k[idx][i] * sig[-i]
                wynik = self.all_e[idx][k] - yk
                sig.append(wynik)
        print(max(sig), min(sig))

        sig = sig[1200:1280]
        npa = np.asarray(sig)
        wavfile.write('rekon8.wav', 11025, npa)
        time = np.linspace(0., len(sig)/11025, len(sig))
        plt.scatter(time, sig, c="red", s=4, label="Odtw")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

    def _rysowanie(self):
        oryg = []
        for idx, segment in enumerate(self.segments_raw):
            if idx == 0:
                oryg.extend(segment)
            else:
                oryg.extend(segment[10:])
        #oryg = oryg[10000:13000]
        time = np.linspace(0., len(oryg)/11025, len(oryg))
        plt.plot(time, oryg, label="Oryginal")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # sygnal = []
        # for idx, segment in enumerate(self.all_e):
        #     sygnal.extend(segment)
        # time = np.linspace(0., len(sygnal)/11025, len(sygnal))
        # plt.scatter(time, sygnal, s=2, c="red", label="Errors")
        # plt.legend()

        plt.show()
    