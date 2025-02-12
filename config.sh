#!/bin/bash

# Create a new conda environment named 'basic_gpt' with Python 3.10
conda create --name basic_gpt python=3.10 -y

# Activate the newly created environment
source activate basic_gpt

# Install the packages listed in requirements.txt using pip
pip install -r requirements.txt

echo "Conda environment 'basic_gpt' created and requirements installed."