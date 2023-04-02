import numpy as np
import math


class PrepareData:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.len = data.shape[0]
        self.N = 256
        self.r = 10

    def prepare(self):
        segments = self._split_to_segments()
        self._flatten(segments)
        self._fill_ends_with_r_zeros(segments)
        return segments

    def _split_to_segments(self):
        segments = self._split()

        if self._is_last_segment_shorter_than_N(segments[-1]):
            segments[-1] = self._fill_last_array_with_zeros_to_N_values(segments[-1])
        return segments
    
    def _split(self):
        segments = [self.data[0:self.N]]
        frm = self.N - self.r
        while frm < self.len:
            segments.append(self.data[frm:frm+self.N])
            frm = frm + self.N - self.r
        return segments
    
    def _is_last_segment_shorter_than_N(self, last_segment: np.ndarray):
        return len(last_segment) < self.N
    
    def _fill_last_array_with_zeros_to_N_values(self, last_segment: np.ndarray):
        return np.concatenate((last_segment, np.zeros(self.N - len(last_segment))))
    
    def _flatten(self, segments: list):
        for segment in segments:
            for k, sample in enumerate(segment, 1):
                w = 0.5 * (1 - math.cos((2*math.pi*k) / (self.N+1)))
                segment[k-1] = sample * w

    def _fill_ends_with_r_zeros(self, segments: list):
        for i in range(len(segments)):
            segments[i] = np.concatenate((segments[i], np.zeros(self.r)))
            segments[i] = np.concatenate((np.zeros(self.r), segments[i]))
