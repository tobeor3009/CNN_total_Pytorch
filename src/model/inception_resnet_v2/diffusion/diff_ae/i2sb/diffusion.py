import argparse

from i2sb.diffusion import Diffusion
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from tqdm import tqdm
from i2sb import util
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from .util import unsqueeze_xdim
from pathlib import Path

# in seg, corrupt_img means source image and clean_img is mask
def sample_batch(data, corrupt_method, corrupt_str, device):
    if corrupt_str == "mixture":
        clean_img, corrupt_img = data
        mask = None
    elif "inpaint" in corrupt_str:
        clean_img = data
        with torch.no_grad():
            corrupt_img, mask = corrupt_method(clean_img.to(device))
    else:
        clean_img = data
        with torch.no_grad():
            corrupt_img = corrupt_method(clean_img.to(device))
        mask = None

    # os.makedirs(".debug", exist_ok=True)
    # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
    # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
    # debug()
    x0 = clean_img.detach().to(device)
    x1 = corrupt_img.detach().to(device)
    if mask is not None:
        mask = mask.detach().to(device)
        x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
    cond = x1.detach() if opt.cond_x1 else None

    if opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    assert x0.shape == x1.shape
    
    return x0, x1, mask, cond

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def compute_label(diffusion, step, x0, xt):
    """ Eq 12 """
    std_fwd = diffusion.get_std_fwd(step, xdim=x0.shape[1:])
    label = (xt - x0) / std_fwd
    return label.detach()

def compute_pred_x0(diffusion, step, xt, net_out, clip_denoise=False):
    """ Given network output, recover x0. This should be the inverse of Eq 12 """
    std_fwd = diffusion.get_std_fwd(step, xdim=xt.shape[1:])
    pred_x0 = xt - std_fwd * net_out
    if clip_denoise:
        pred_x0.clamp_(-1., 1.)
    return pred_x0

def ddim_sampling(diffusion, ema, x0, x1, mask, cond, net, nfe=50, ot_ode=False, interval=1000):
    device = next(net.parameters()).device
    with torch.no_grad():
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.

        nfe = nfe or interval - 1
        assert 0 < nfe < interval == len(diffusion.betas)
        steps = util.space_indices(interval, nfe+1)

        x1 = x1.to(device)
        if cond is not None:
            cond = cond.to(device)
        if mask is not None:
            mask = mask.to(device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with ema.average_parameters():
            net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
                out = net(xt, step, cond=cond).pred
                return compute_pred_x0(diffusion, step, xt, out, clip_denoise=True)

            xs, pred_x0 = diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=ot_ode, log_steps=None, verbose=False,
            )
        return xs, pred_x0
    
class DiffusionI2SB(torch.nn.Module):
    def __init__(self, diffusion_model, noise_levels, cond=False):
        super().__init__()
        
        self.diffusion_model = diffusion_model
        self.noise_levels = noise_levels
        self.cond = cond
        
    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)

# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

opt = argparse.Namespace(
    # basic
    seed=0,
    name="test",
    ckpt=None,
    gpu=None,
    n_gpu_per_node=1,
    master_address="localhost",
    node_rank=0,
    num_proc_node=1,

    # SB model
    image_size=256,
    corrupt="mixture",
    t0=1e-4,
    T=1.0,
    interval=1000,
    beta_max=1.0,
    ot_ode=False,
    clip_denoise=False,
    cond_x1=False,
    add_x1_noise=False,

    # optimizer and loss
    batch_size=256,
    microbatch=2,
    num_itr=1_000_000,
    lr=5e-5,
    lr_gamma=0.99,
    lr_step=1000,
    l2_norm=0.0,
    ema=0.99,

    # path and logging
    dataset_dir=Path("/dataset"),
    log_dir=Path(".log"),
    log_writer=None,
    wandb_api_key=None,
    wandb_user=None,
)