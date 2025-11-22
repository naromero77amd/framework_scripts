#!/usr/bin/bash 
pip uninstall pytorch-triton-rocm
pip uninstall triton -y
pip install -r python/requirements.txt
# pip install -e . # triton 3.4 and later
pip install -e python # triton 3.3
