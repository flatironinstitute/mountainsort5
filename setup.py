from setuptools import setup, find_packages

# read the contents of README.md
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

__version__ = '0.5.8'

setup(
    name='mountainsort5',
    version=__version__,
    author="Jeremy Magland",
    author_email="jmagland@flatironinstitute.org",
    url="https://github.com/flatironinstitute/mountainsort5",
    description="MountainSort 5 spike sorting algorithm",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'spikeinterface>=0.97.1',
        'isosplit6>=0.1.0',
        'scikit-learn',
        'packaging'
    ],
    tests_require=[
        "pytest",
        "pytest-cov"
    ]
)
