#!/bin/bash
set -ex

pytest --cov=mountainsort5 --cov-report=xml --cov-report=term tests/