#!/bin/bash

apt update && apt-get install screen -y
# /workspace/miniconda3/bin/conda init
/bin/bash/source ~/.bashrc
python -m ipykernel install --user --name arc-agi --display-name "ARC"