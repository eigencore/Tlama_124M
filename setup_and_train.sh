#!/bin/bash

# Variables
REPO_URL="https://github.com/eigencore/TinyRB.git"
REPO_DIR="TinyRB"
LOG_DIR="log"
ARCHIVE_NAME="logs.tar.gz"

# 1. Clone the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning the repository..."
    git clone "$REPO_URL"
else
    echo "The repository already exists, skipping cloning."
fi

# 2. Enter the directory
cd "$REPO_DIR" || exit 1

# 3. Create the virtual environment
if [ ! -d "env" ]; then
    echo "Creating the virtual environment..."
    python -m venv env
else
    echo "The virtual environment already exists."
fi

# 4. Activate the virtual environment
echo "Activating the virtual environment..."
source env/bin/activate

# 5. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Aborting."
    exit 1
fi

# 6. Run the training script
echo "Starting training..."
python train.py

# 7. Compress logs
if [ -d "$LOG_DIR" ]; then
    echo "Compressing logs into $ARCHIVE_NAME..."
    tar -czf "$ARCHIVE_NAME" "$LOG_DIR"
    echo "Logs successfully compressed."
else
    echo "Log directory not found, skipping compression."
fi

echo "Process completed."
