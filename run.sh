#!/bin/bash

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

python main.py