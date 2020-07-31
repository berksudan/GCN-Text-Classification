#!/bin/bash

# Change current directory to project directory. 
CURRENT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$CURRENT_PATH" 

# Create virtual environment directory
python3 -m venv venv/

# Activate virtual environment
source venv/bin/activate

# Upgrade Python 
python -m pip install --upgrade pip
python -m pip3 install --upgrade pip

# Check version of pip
# Version must be below 18.XX and compatible with Python 3.4+
pip --version

# Install dependencies
pip3 install -I nltk numpy networkx scipy 
pip3 install -I sklearn torch