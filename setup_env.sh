#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project requirements
pip install -r "Research Framework for Optimizing Head-level KV Cache Based on CAKE/requirements.txt"

# Install the CAKE project in editable mode and run its install script
(
    cd "Research Framework for Optimizing Head-level KV Cache Based on CAKE/cakekv-main/cakekv-main" \
    && bash install.sh
)

echo "Setup complete. Activate the environment with 'source venv/bin/activate'."
