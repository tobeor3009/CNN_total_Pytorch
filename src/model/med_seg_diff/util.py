from collections import namedtuple
from functools import partial
import torch
import math

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def const_beta_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.tensor([scale * 0.008] * timesteps, dtype=torch.float32)

def convert_class_label(class_label, device):
    if isinstance(class_label, (list, tuple)):
        class_label = [item.to(device=device, dtype=torch.long)
                        for item in class_label]
    elif class_label is None:
        pass
    else:
        class_label = class_label.to(device=device, dtype=torch.long)
    return class_label

def forward_with_cond_scale(
    model,
    x, t, c,
    x_self_cond,
    class_label,
    cond_scale = 1.,
    rescaled_phi = 0.,
):
    logits = model.forward(x, t, c, x_self_cond, class_label, cond_drop_prob = 0.)

    if cond_scale == 1:
        return logits

    null_logits = model.forward(x, t, c, x_self_cond, class_label, cond_drop_prob = 1.)
    scaled_logits = null_logits + (logits - null_logits) * cond_scale

    if rescaled_phi == 0.:
        return scaled_logits

    std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
    rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

    return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)