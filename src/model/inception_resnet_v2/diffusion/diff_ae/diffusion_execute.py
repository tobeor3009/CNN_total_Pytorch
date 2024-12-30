import math
import torch
from torch import nn
from torch.amp import autocast
from .resample import UniformSampler
from .diffusion_sampler import GaussianSampler
from .renderer import render_uncondition, render_condition

def _z_normalize(x, target_dim_tuple, eps=1e-5):
    x_mean = x.mean(dim=target_dim_tuple, keepdim=True)
    x_var = x.var(dim=target_dim_tuple, correction=0, keepdim=True)
    x_std = torch.sqrt(x_var + eps)
    x_normalized = (x - x_mean) / x_std    
    return x_normalized, x_mean, x_std

def feature_z_normalize(x, eps=1e-5):
    target_dim_tuple = tuple(range(1, x.ndim))
    x_normalized, x_mean, x_std = _z_normalize(x, target_dim_tuple, eps=eps)
    return x_normalized, x_mean, x_std

def z_normalize(x, eps=1e-5):
    target_dim_tuple = tuple(range(2, x.ndim))
    x_normalized, x_mean, x_std = _z_normalize(x, target_dim_tuple, eps=eps)
    return x_normalized, x_mean, x_std

class AutoEncoder(nn.Module):
    def __init__(self, diffusion_model, train_mode, sample_size=1, img_size=512, img_dim=2,
                 T=1000, T_eval=1000, T_sampler="uniform",
                 beta_scheduler="linear", spaced=True, rescale_timesteps=False,
                 gen_type="ddpm", model_type="autoencoder", model_mean_type="eps", model_var_type="fixed_large", model_loss_type="mse",
                 latent_gen_type="ddim", latent_model_mean_type="eps", latent_model_var_type="fixed_large", latent_model_loss_type="l1",
                 latent_clip_sample=False, latent_znormalize=True,
                 fp16=False, train_pred_xstart_detach=True):
        
        train_mode_list = ["autoencoder", "latent_net", "ddpm"]
        if train_mode == "autoencoder":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is not None, "diffusion_model.encoder need to be set when you training autoencoder"
        elif train_mode == "latent_net":
            latent_net = getattr(diffusion_model, "latent_net", None)
            assert latent_net is not None, "diffusion_model.latent_net need to be set when you training latent_net"
        elif train_mode == "ddpm":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is None, "diffusion_model.encoder need to be None whtn you training ddpm"
        else:
            raise NotImplementedError(f"{train_mode} is not in {train_mode_list}")
        
        ##########################
        latent_model_type = "ddpm"
        ##########################
        self.train_mode = train_mode
        self.diffusion_model = diffusion_model
        self.img_size = getattr(diffusion_model, img_size, None) or img_size
        self.img_dim = img_dim
        self.in_channel = diffusion_model.in_channel
        self.cond_channel = diffusion_model.cond_channel
        self.out_channel = diffusion_model.out_channel
        self.model_type = model_type
        self.latent_clip_sample = latent_clip_sample
        self.latent_znormalize = latent_znormalize
        self.T = T
        self.T_eval = T_eval
        if T_sampler == 'uniform':
            self.T_sampler = UniformSampler(self.T)
        else:
            raise NotImplementedError()
        
        self.common_sampler_kwarg_dict = {
             "beta_scheduler": beta_scheduler,
             "spaced": spaced,
             "rescale_timesteps": rescale_timesteps,
             "gen_type": gen_type,
             "model_type": model_type,
             "model_mean_type": model_mean_type,
             "model_var_type": model_var_type,
             "loss_type": model_loss_type,
             "fp16": fp16,
             "train_pred_xstart_detach": train_pred_xstart_detach
        }
        self.sampler = self.get_sampler(T, self.common_sampler_kwarg_dict)
        self.eval_sampler = self.get_sampler(T_eval, self.common_sampler_kwarg_dict)
        self.common_latent_sampler_kwarg_dict = {
            "beta_scheduler": beta_scheduler,
            "spaced": spaced,
            "rescale_timesteps": rescale_timesteps,
            "gen_type": latent_gen_type,
            "model_type": latent_model_type,
            "model_mean_type": latent_model_mean_type,
            "model_var_type": latent_model_var_type,
            "loss_type": latent_model_loss_type,
            "fp16": fp16,
            "train_pred_xstart_detach": train_pred_xstart_detach
        }

        if train_mode == "latent_net":
            self.latent_sampler = self.get_sampler(T, self.common_latent_sampler_kwarg_dict)
            self.eval_latent_sampler = self.get_sampler(T_eval, self.common_latent_sampler_kwarg_dict)

        # initial variables for consistent sampling
        self.register_buffer('x_T', self.get_noise(sample_size))
    
    def get_noise(self, batch_size):
        image_size_shape = tuple(self.img_size for _ in range(self.img_dim))
        return torch.randn(batch_size, self.in_channel, *image_size_shape)
    
    def get_sampler(self, T, common_sampler_kwarg_dict):
        return GaussianSampler(T=T, **common_sampler_kwarg_dict)
    
    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.get_sampler(T, self.common_sampler_kwarg_dict)
            latent_sampler = self.get_sampler(T_latent, self.common_latent_sampler_kwarg_dict)

        noise = self.get_noise(N).to(device=device)
        
        pred_img = render_uncondition(
            self.diffusion_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            latent_clip_sample=self.latent_clip_sample,
            latent_znormalize=self.latent_znormalize,
            train_mode=self.train_mode
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.eval_latent_sampler
        else:
            sampler = self.get_sampler(T, self.common_sampler_kwarg_dict)
            latent_sampler = self.get_sampler(T, self.common_latent_sampler_kwarg_dict)

        if cond is not None:
            pred_img = render_condition(self.diffusion_model,
                                        noise, sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.diffusion_model,
                                          noise, sampler, latent_sampler,
                                          latent_clip_sample=self.latent_clip_sample,
                                          latent_znormalize=self.latent_znormalize,
                                          train_mode=self.train_mode)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        assert self.model_type == "autoencoder"
        cond = self.diffusion_model.encoder.forward(x)
        return cond
    
    def encode_stochastic(self, x, cond, T=None):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.get_sampler(T, self.common_sampler_kwarg_dict)
        out = sampler.ddim_reverse_sample_loop(self.diffusion_model,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    # Check: remove kwarg ema_model
    def forward(self, noise=None, x_start=None):
        with autocast(False):
            gen = self.eval_sampler.sample(model=self.diffusion_model,
                                           noise=noise,
                                           x_start=x_start)
            return gen
        
    def manipulate(self, x, class_idx, cls_model, manipulate_weight):
        class_mlp_weight = cls_model.classifier.weight
        style_ch = class_mlp_weight.shape[1]
        target_class_weight = cls_model.classifier.weight[class_idx][None, :], dim=1
        target_dim_tuple = tuple(range(1, target_class_weight))
        target_class_weight, *_ = _z_normalize(target_class_weight, target_dim_tuple, eps=1e-5)
        cond = self.encode(x)
        manipulated_cond = cond + manipulate_weight * math.sqrt(style_ch) * target_class_weight
        return manipulated_cond

        