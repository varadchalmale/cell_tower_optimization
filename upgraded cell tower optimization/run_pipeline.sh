#!/bin/bash
echo "=== AI-DRIVEN CELL TOWER PLANNING PIPELINE ==="

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python3."
    exit
fi

# Setup Virtual Environment if not exists
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create required directories
mkdir -p Data/raw outputs

# Run main script
echo "Running the optimization pipeline..."
python main.py

echo "=== Pipeline execution finished ==="
