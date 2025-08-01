from   matplotlib import colors

import os.path as osp
import torch

class Config:
    # I/O
    root            = "../" 
    data_path       = osp.join(root, "data")
    submission_path = osp.join(root, "submissions")
    model_zoo       = osp.join(root, "models")
    experiment      = "simple-arc-solver" #"llm-fs"
    
    # 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    CMAP            = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    NORM             = colors.Normalize(vmin=0, vmax=9)
    DEFAULT_BG_VALUE = 7  # (orange not black...from EDA)

    # model vars
    base_llm            = "mlx-community/Qwen2.5-14B-Instruct-8bit"
    model_name          = "baseline-cp" #base_llm.split("/")[-1]
    device              = 'mps'
    target_platform     = "mlx"
    dtype               = torch.float16
    max_tokens          = 4096
    temperature         = 0.1
    top_p               = 0.95
    repetition_penalty  = 1.4
    do_sample           = True
    trust_remote_code   = True
    MAX_N_ROWS          = 30


    
