import torch

SMOOTH = 1e-7

class UnexpectedBehaviorException(Exception):
    def __init__(self, message="An unintended exception occurred."):
        super().__init__(message)

def get_clip(y_pred, y_true, smooth):
    y_true = torch.clamp(y_true, min=smooth)
    y_pred = torch.clamp(y_pred, min=smooth)
    y_true_rev = torch.clamp(1 - y_true, min=smooth)
    y_pred_rev = torch.clamp(1 - y_pred, min=smooth)
    return y_pred, y_true, y_true_rev, y_pred_rev

def get_rev(y_pred, y_true):
    y_pred_rev = 1 - y_pred
    y_true_rev = 1 - y_true
    return y_pred_rev, y_true_rev

def get_one_hot_argmax(class_tensor):
    # seg_tensor: Tensor of shape (B, C)
    class_max_idx_tensor = class_tensor.argmax(1)  # Shape: (B)
    # Create a one-hot encoded tensor
    # Create a tensor of zeros with the same shape as seg_tensor
    one_hot_class_tensor = torch.zeros_like(class_tensor)
    # Scatter 1s along the channel dimension based on the argmax indices
    one_hot_class_tensor.scatter_(1, class_max_idx_tensor.unsqueeze(1), 1)
    return one_hot_class_tensor