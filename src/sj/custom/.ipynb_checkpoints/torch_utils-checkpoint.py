import torch
import copy

def load_state_dict(model, pretrained_dict):
    # 1. filter out unnecessary keys
    model_state_dict = model.state_dict() 
    for k, v in pretrained_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                model_state_dict[k] = v
    # 2. load state dict
    model.load_state_dict(model_state_dict)
    
    # 3. return
    return model