import importlib.metadata
__version__ = importlib.metadata.version("mountainsort5")

from .schemes.sorting_scheme1 import sorting_scheme1
from .schemes.Scheme1SortingParameters import Scheme1SortingParameters
from .schemes.sorting_scheme2 import sorting_scheme2
from .schemes.Scheme2SortingParameters import Scheme2SortingParameters
from .schemes.sorting_scheme3 import sorting_scheme3
from .schemes.Scheme3SortingParameters import Scheme3SortingParameters