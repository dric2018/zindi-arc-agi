import os.path as osp
import torch

class Config:
    # I/O
    root            = "../" 
    data_path       = osp.join(root, "data")
    submission_path = osp.join(root, "submissions")
    model_zoo       = osp.join(root, "models")
    experiment      = "baseline"
    DEFAULT_BG_VALUE = 7  # (orange not black...from EDA)


    # model vars
    base_llm        = "nvidia/Mistral-NeMo-Minitron-8B-Base"
    model_name      = "simple-arc-solver"
    device          = 'mps'
    dtype           = torch.bfloat16
    max_tokens      = 512
    
