#!/bin/bash
set -ex

# black --check .
pyright
pytest --cov=mountainsort5 --cov-report=xml --cov-report=term tests/