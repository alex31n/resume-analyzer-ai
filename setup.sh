#!/bin/bash
echo "Setting up Resume Analyzer AI..."

# Create necessary directories
mkdir -p data/raw_data data/processed_data models/checkpoints

# Install dependencies
pip install -r requirements.txt

echo "Setup complete!"