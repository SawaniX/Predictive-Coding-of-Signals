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
        quants = [lista[i:i + self.n] for i in range(0, len(lista), self.n)]
        self._dequantization(quants)

    def reconstruct_signal(self):
        signal = []
        y_prev = 10 * [0]
        for idx, segment in enumerate(self.all_e):
            for k in range(self.n):
                yk = 0
                for i in range(1, self.r+1):
                    if k - i >= 0:
                        yk += self.all_k[idx][i] * y_prev[k-i]
                yk = -yk + self.all_e[idx][k]
                if k < 246:
                    signal.append(yk)
                y_prev.append(yk)
        
        npa = np.asarray(signal, dtype=np.int16)
        wavfile.write(f'{str(self.bits)}bity.wav', 11025, npa.astype(np.int16))
        time = np.linspace(0., len(signal) / 11025, len(signal))
        #time2 = np.linspace(0., len(signal) / 11025, int(len(signal)/len(self.all_e)))
        plt.plot(time, signal, label="Odtworzony")
        #plt.vlines(x = time2, ymin = -5000, ymax = 5000, color = 'b')
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
