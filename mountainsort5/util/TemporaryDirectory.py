import shutil
import tempfile


class TemporaryDirectory:
    """A context manager for temporary directories"""
    def __init__(self, dir=None):
        self._dir = None
        self._dir = dir
    def __enter__(self):
        self._dir = tempfile.mkdtemp(dir=self._dir)
        return self._dir
    def __exit__(self, exc_type, exc_value, traceback):
        if self._dir:
            shutil.rmtree(self._dir)
