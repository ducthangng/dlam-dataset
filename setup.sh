#!/bin/bash

set -e

curl -o ./data https://aws-ducthangng-dataset.s3.amazonaws.com/dlam-dataset/data/train.csv\?X-Amz-Algorithm\=AWS4-HMAC-SHA256\&X-Amz-Credential\=AKIAZRZKF4AZJRVG6EAI%2F20260609%2Fus-east-1%2Fs3%2Faws4_request\&X-Amz-Date\=20260609T143640Z\&X-Amz-Expires\=86400\&X-Amz-SignedHeaders\=host\&X-Amz-Signature\=2f66b116d49c5101a4438e1b95f77a985233344038d095f084ed70b04f434ecb

# Install all required Python packages
pip install torch pytorch-forecasting lightning pandas numpy matplotlib pyarrow

echo "Done"
