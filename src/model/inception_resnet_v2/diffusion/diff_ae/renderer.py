import torch
from torch.cuda import amp
from .model import z_normalize
import os
from .diffusion_sampler import GaussianSampler
def render_uncondition(model, x_T, sampler: GaussianSampler, latent_sampler: GaussianSampler,
                       clip_latent_noise: bool = False,
                       latent_clip_sample=False,
                       cond_mean=None,
                       cond_std=None,
                       train_mode=None,
                       clip_denoised=True,
                       image_mask_cat_fn=None,
                       image_mask_split_fn=None):
    device = x_T.device
    if train_mode in ["ddpm"]:
        return sampler.sample(model=model, noise=x_T)
    elif train_mode in ["latent_net", "autoencoder_latent_net"]:
        latent_noise = torch.randn(len(x_T), model.emb_channel, device=device)

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=latent_clip_sample,
        )
        if (cond_mean is not None) and (cond_std is not None):
            cond = cond * cond_std + cond_mean 
        # the diffusion on the model
        return sampler.sample(model=model, image_mask_cat_fn=image_mask_cat_fn, image_mask_split_fn=image_mask_split_fn,
                              noise=x_T, model_kwargs={'cond': cond}, clip_denoised=clip_denoised)
    else:
        raise NotImplementedError()

# def render_uncondition(model, x_T, sampler, latent_sampler,
#                        latent_noise=None,
#                        clip_latent_noise: bool = False,
#                        latent_clip_sample=False,
#                        latent_znormalize=True,
#                        train_mode=None,
#                        clip_denoised=True,
#                        image_mask_cat_fn=None,
#                        image_mask_split_fn=None):
#     device = x_T.device
#     if train_mode in ["ddpm"]:
#         return sampler.sample(model=model, noise=x_T)
#     elif train_mode in ["latent_net", "autoencoder_latent_net"]:
#         if latent_noise is None:
#             latent_noise = torch.randn(len(x_T), model.cond_channel, device=device)

#         if clip_latent_noise:
#             latent_noise = latent_noise.clip(-1, 1)

#         cond = latent_sampler.sample(
#             model=model.latent_net,
#             noise=latent_noise,
#             clip_denoised=latent_clip_sample,
#         )
#         if latent_znormalize:
#             cond = cond * model.conds_std.to(device) + model.conds_mean.to(device)
        
#         # temp_path = os.environ["TEMP_PATH"]
#         # torch.save({"x_T": x_T, "cond":cond}, f"{temp_path}/init_custom.ckpt")
#         # the diffusion on the model
#         return sampler.sample(model=model, image_mask_cat_fn=image_mask_cat_fn, image_mask_split_fn=image_mask_split_fn,
#                               noise=x_T, cond=cond, clip_denoised=clip_denoised)
#     else:
#         raise NotImplementedError()

def render_condition(
    model, x_T, sampler,
    x_start=None,
    cond=None,
    train_mode=None,
    clip_denoised=True,
    image_mask_cat_fn=None,
    image_mask_split_fn=None
):
    if train_mode in ["autoencoder", "autoencoder_latent_net", "latent_net"]:
        if cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model,
                              image_mask_cat_fn=image_mask_cat_fn,
                              image_mask_split_fn=image_mask_split_fn,
                              noise=x_T,
                              model_kwargs={'cond': cond},
                              clip_denoised=clip_denoised)
    else:
        raise NotImplementedError()
