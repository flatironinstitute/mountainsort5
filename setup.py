from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        'spikeinterface>=0.97.1',
        'isosplit5>=0.2.0', # actually depends on isosplit6 alg within isosplit5 repo
        'scikit-learn'
    ]
)