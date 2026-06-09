#!/bin/bash

set -e

# Install all required Python packages
pip install torch pytorch-forecasting lightning pandas numpy matplotlib pyarrow

echo "Done"
