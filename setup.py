from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    install_requires=[
        'spikeinterface>=0.97.1',
        'isosplit6>=0.1.0',
        'scikit-learn'
    ],
    tests_require=[
        "pytest",
        "pytest-cov"
    ]
)