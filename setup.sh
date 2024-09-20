#!/bin/bash

echo "Installing Python dependencies from requirements.txt..."

# Check if pip or pip3 is available
if command -v pip3 &> /dev/null
then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null
then
    pip install -r requirements.txt
else
    echo "Error: pip or pip3 not found. Please install pip."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please check if pip is installed correctly."
    exit 1
fi

echo "Dependencies installed successfully."
