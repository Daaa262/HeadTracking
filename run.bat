@echo off
SET VENV_DIR=.venv

IF NOT EXIST %VENV_DIR%\Scripts\activate.bat (
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt

python main.py
