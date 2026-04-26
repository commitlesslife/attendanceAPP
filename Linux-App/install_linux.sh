#!/bin/bash
echo "Setting up Smart Attendance System for Linux..."

# Install system dependencies (Ubuntu/Debian)
if [ -x "$(command -v apt-get)" ]; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y cmake build-essential python3-venv python3-tk libx11-dev libgtk-3-dev
else
    echo "Please ensure cmake, build-essential, and python3-tk are installed on your system."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install requirements
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "======================================"
echo "Installation complete!"
echo "Use ./run_linux.sh to start the app."
echo "======================================"
