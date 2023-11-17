import importlib.metadata
__version__ = importlib.metadata.version("mountainsort5")

from .schemes.sorting_scheme1 import sorting_scheme1 # noqa: F401
from .schemes.Scheme1SortingParameters import Scheme1SortingParameters # noqa: F401
from .schemes.sorting_scheme2 import sorting_scheme2 # noqa: F401
from .schemes.Scheme2SortingParameters import Scheme2SortingParameters # noqa: F401
from .schemes.sorting_scheme3 import sorting_scheme3 # noqa: F401
from .schemes.Scheme3SortingParameters import Scheme3SortingParameters # noqa: F401
