#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate basic_gpt

# Run the predict.py script from the src folder with the given seed
python src/predict.py --seed "This is test seed"