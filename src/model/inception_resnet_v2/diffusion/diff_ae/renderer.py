import torch
from torch.cuda import amp
from .model import z_normalize

def render_uncondition(model, x_T, sampler, latent_sampler,
                       clip_latent_noise: bool = False,
                       latent_clip_sample=False,
                       latent_znormalize=True,
                       train_mode=None):
    device = x_T.device
    if train_mode in ["ddpm"]:
        return sampler.sample(model=model, noise=x_T)
    elif train_mode in ["latent_net"]:
        latent_noise = torch.randn(len(x_T), model.feature_channel, device=device)

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=latent_clip_sample,
        )
        if latent_znormalize:
            cond = z_normalize(cond)

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError()


def render_condition(
    model, x_T, sampler,
    x_start=None,
    cond=None,
    train_mode=None
):
    if train_mode in ["autoencoder", "ddpm"]:
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              noise=x_T,
                              model_kwargs={'cond': cond})
    else:
        raise NotImplementedError()
