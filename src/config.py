from   matplotlib import colors

import os.path as osp
import torch

class Config:
    # I/O
    root            = "../" 
    data_path       = osp.join(root, "data")
    submission_path = osp.join(root, "submissions")
    model_zoo       = osp.join(root, "models")
    experiment      = "baseline-cp"
    
    # 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    CMAP            = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    NORM             = colors.Normalize(vmin=0, vmax=9)
    DEFAULT_BG_VALUE = 7  # (orange not black...from EDA)

    # model vars
    base_llm        = "nvidia/Mistral-NeMo-Minitron-8B-Base"
    model_name      = "simple-arc-solver-cp"
    device          = 'mps'
    dtype           = torch.float16
    max_tokens      = 512
    max_candidates  = 3

    
