import argparse

from i2sb.diffusion import Diffusion
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from tqdm import tqdm
from i2sb import util
import torch
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

def ddim_sampling(diffusion, ema, x0, x1, mask, cond, net, nfe=50, ot_ode=False, interval=1000, clip_denoise=True):
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

        net.eval()

        def pred_x0_fn(xt, step):
            step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
            out = net(xt, step, cond=cond).pred
            return compute_pred_x0(diffusion, step, xt, out, clip_denoise=clip_denoise)

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