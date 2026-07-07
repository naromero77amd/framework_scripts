#!/usr/bin/bash 
# NOTE: Always confirm that Triton is properly installed
#       by doing: pip show triton
pip uninstall -y pytorch-triton-rocm
pip uninstall -y triton
unset CMAKE_PREFIX_PATH
python setup.py clean
pip install -r python/requirements.txt
pip install . # triton 3.4 and later
# pip install python # triton 3.3
