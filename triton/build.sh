#!/usr/bin/bash 
pip uninstall pytorch-triton-rocm
pip uninstall triton -y
pip install -r python/requirements.txt
pip install .
