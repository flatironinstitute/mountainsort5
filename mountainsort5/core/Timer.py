import time


class Timer:
    def __init__(self, label: str):
        self._label = label
        self._start_time = time.time()
    def report(self):
        elapsed = time.time() - self._start_time
        print(f'*** MS5 Elapsed time for {self._label}: {elapsed:.3f} seconds ***')
