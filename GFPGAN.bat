@echo off

:: Activate the conda environment for GFPGAN
CALL "C:\ProgramData\miniconda3\Scripts\activate.bat" GFPGAN

:: Navigate to the GFPGAN directory (Change path accroding to yourself)
cd /D path/to/your/GFPGAN-Web-Interface

:: Run the GFPGAN web interface script
python GFPGAN.py
