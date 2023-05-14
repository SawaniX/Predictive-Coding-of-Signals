import numpy as np
import math
import copy


class PrepareData:
    def __init__(self, data: np.ndarray, r: int):
        self.data = data
        self.len = data.shape[0]
        self.N = 256
        self.r = r

    def prepare(self):
        segments = self._split_to_segments()
        segments_raw = copy.deepcopy(segments)
        self._flatten(segments)
        self._fill_ends_with_white_noise(segments)
        return segments, segments_raw

    def _split_to_segments(self):
        segments = self._split()

        # if self._is_last_segment_shorter_than_N(segments[-1]):
        #     segments[-1] = self._fill_last_array_with_zeros_to_N_values(segments[-1])
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
        for idx, segment in enumerate(segments):
            for k, sample in enumerate(segment):
                w = 0.5 * (1 - math.cos((2*math.pi*k) / (self.N+1)))
                segments[idx][k] = sample * w

    def _fill_ends_with_white_noise(self, segments: list):
        mean = 0
        std = 1
        num_samples = 10
        for i in range(len(segments)):
            white_noise_start = np.random.normal(mean, std, size=num_samples)
            white_noise_end = np.random.normal(mean, std, size=num_samples)
            segments[i] = np.concatenate((segments[i], white_noise_start))
            segments[i] = np.concatenate((white_noise_end, segments[i]))
