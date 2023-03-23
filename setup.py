from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        'spikeinterface>=0.97.1',
        'scikit-learn'
    ]
)