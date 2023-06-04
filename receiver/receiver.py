from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import struct
import bitstring

class Receiver:
    def __init__(self, bits: int):
        self.r = 10
        self.n = 256
        self.bits = bits
        self._read_files()
        self.reconstruct_signal()

    def _dequantization(self, quants):
        quant_level = max([max(quant) for quant in quants]) + 1
        
        self.all_e = []
        for idx, quant in enumerate(quants):
            result = []
            step = (2 * self.emax[idx]) / (quant_level - 1)
            dic = {i: -self.emax[idx] + i*step for i in range(quant_level)}
            for q in quant:
                result.append(dic[q])
            self.all_e.append(result)

    def _read_files(self):
        with open(f'a{self.bits}.bin', 'rb') as f:
            a = np.fromfile(f, dtype=np.float32)
        a_list = a.tolist()
        self.all_k = [a_list[i:i + self.r+1] for i in range(0, len(a_list), self.r+1)]

        with open(f'emax{self.bits}.bin', 'rb') as f:
            self.emax = np.fromfile(f, dtype=np.float32)

        with open(f'quants{self.bits}.bin', 'rb') as plik:
            bajty = plik.read()
        bit_lista = bitstring.BitArray(bytes=bajty)
        lista = []
        bit_index = 0
        while bit_index < len(bit_lista):
            liczba = bit_lista[bit_index:bit_index+self.bits].uint
            lista.append(liczba)
            bit_index += self.bits
        l = [self.n]
        l.extend(list(range(self.n, len(lista), self.n-10)))
        quants = []
        for idx, i in enumerate(l):
            if idx == 0:
                print("tu")
                quants.append(lista[i:i + self.n])
            else:
                quants.append(lista[i:i + self.n-10])
        self._dequantization(quants)

    def reconstruct_signal(self):
        odtw = []
        for idx, ee in enumerate(self.all_e):
            odtw_seg = []
            for ind, eee in enumerate(ee):
                yk = 0
                for i in range(1, 11):
                    if ind - i >= 0:
                        yk += self.all_k[idx][i] * odtw_seg[ind-i]
                    elif ind - i < 0 and idx > 0:
                        yk += self.all_k[idx][i] * odtw[idx-1][-i]
                yk = eee - yk/4
                odtw_seg.append(yk)
            odtw.append(odtw_seg)
        signal = [x for a in odtw for x in a]
        
        npa = np.asarray(signal, dtype=np.int16)
        wavfile.write(f'{str(self.bits)}bity.wav', 11025, npa.astype(np.int16))

        time = np.linspace(0., len(signal) / 11025, len(signal))
        plt.scatter(time, signal, s=4, c='red', label="Odtworzony")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
