#!/bin/bash

# Create a new conda environment named 'basic_gpt' with Python 3.10
conda create --name basic_gpt python=3.10 -y

# Activate the newly created environment
source activate basic_gpt

# Install the packages listed in requirements.txt using pip
pip install -r requirements.txt

# Test if PyTorch MPS device is available
python -c "import torch; print('MPS device available:', torch.backends.mps.is_available())"

echo "Conda environment 'basic_gpt' created, requirements installed, and MPS device availability tested."