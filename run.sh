#!/bin/bash

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python3 -m streamlit run app.py