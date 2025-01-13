import math
import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F 
from .resample import UniformSampler
from .diffusion_sampler import GaussianSampler
from .renderer import render_uncondition, render_condition
from src.model.train_util.common import _z_normalize, feature_z_normalize, z_normalize

class AutoEncoder(nn.Module):
    def __init__(self, diffusion_model, train_mode, sample_size=1, img_size=512, img_dim=2,
                 T=1000, T_eval=20, T_sampler="uniform",
                 beta_scheduler="linear", spaced=True, rescale_timesteps=False,
                 gen_type="ddim", model_type="autoencoder", model_mean_type="eps", model_var_type="fixed_large", model_loss_type="mse",
                 latent_gen_type="ddim", latent_model_mean_type="eps", latent_model_var_type="fixed_large", latent_model_loss_type="l1",
                 latent_clip_sample=False, latent_znormalize=True,
                 fp16=False, train_pred_xstart_detach=True):
        super().__init__()
        train_mode_list = ["autoencoder", "latent_net", "ddpm"]
        if train_mode == "autoencoder":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is not None, "diffusion_model.encoder need to be set when you training autoencoder"
        elif train_mode == "latent_net":
            latent_net = getattr(diffusion_model, "latent_net", None)
            assert latent_net is not None, "diffusion_model.latent_net need to be set when you training latent_net"
        elif train_mode == "ddpm":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is None, "diffusion_model.encoder need to be None when you training ddpm"
        else:
            raise NotImplementedError(f"{train_mode} is not in {train_mode_list}")
        
        ##########################
        latent_model_type = "ddpm"
        ##########################
        self.train_mode = train_mode
        self.diffusion_model = diffusion_model
        self.img_size = getattr(diffusion_model, "img_size", None) or img_size
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
        self.sampler = self.get_sampler(T=self.T, T_eval=self.T)
        self.eval_sampler = self.get_sampler(T=self.T, T_eval=self.T_eval)

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
            self.latent_sampler = self.get_latent_sampler(T=self.T, T_eval=self.T)
            self.eval_latent_sampler = self.get_latent_sampler(T=self.T, T_eval=self.T_eval)

        # initial variables for consistent sampling
        # 표준 정규분포 정의 (평균=0, 표준편차=1)
        # 99% 신뢰구간에 해당하는 누적 확률값 (0.5%와 99.5%)
        self.register_buffer('x_T', self.get_noise(sample_size))

    def get_noise(self, batch_size):
        image_size_shape = tuple(self.img_size for _ in range(self.img_dim))
        image_size_shape = (batch_size, self.in_channel, *image_size_shape)
        noise = self.sampler.get_noise_as_shape(image_size_shape, device="cpu")
        return noise

    def get_sampler(self, T, T_eval):
        return GaussianSampler(T=T, T_eval=T_eval, **self.common_sampler_kwarg_dict)
    
    def get_latent_sampler(self, T, T_eval):
        return GaussianSampler(T=T, T_eval=T_eval, **self.common_latent_sampler_kwarg_dict)
    
    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
            latent_sampler = self.get_latent_sampler(T=self.T, T_eval=T_latent)

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

    def render(self, noise, cond=None, T=None, clip_denoised=True):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
        if cond is not None:
            pred_img = render_condition(self.diffusion_model,
                                        noise, sampler,
                                        cond=cond,
                                        train_mode=self.train_mode)
        else:
            pred_img = render_uncondition(self.diffusion_model,
                                          noise, sampler, latent_sampler=None,
                                          latent_clip_sample=self.latent_clip_sample,
                                          latent_znormalize=self.latent_znormalize,
                                          train_mode=self.train_mode)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x):
        assert getattr(self.diffusion_model, "encoder", None) is not None
        cond = self.diffusion_model.encoder.forward(x)
        return cond
    
    def encode_stochastic(self, x, cond, T=None, clip_denoised=True):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
        out = sampler.ddim_reverse_sample_loop(self.diffusion_model,
                                               x,
                                               model_kwargs={'cond': cond},
                                               clip_denoised=clip_denoised)
        return out['sample']

    def forward(self, x_start=None, encoded_feature=None):
        return self.get_loss(x_start=x_start, encoded_feature=encoded_feature)
    # Check: remove kwarg ema_model
    def get_loss(self, x_start=None, encoded_feature=None):
        x_start_device = getattr(x_start, "device", None)
        encoded_feature_device = getattr(x_start, "device", None)
        torch_device = x_start_device or encoded_feature_device
        # with autocast(device_type=torch_device, enabled=False):
        if self.train_mode in ["autoencoder", "ddpm"]:
            assert encoded_feature is None
            t, weight = self.T_sampler.sample(len(x_start), torch_device)
            result_dict = self.sampler.training_losses(model=self.diffusion_model,
                                                        x_start=x_start, t=t)
        elif self.train_mode == "latent_net":
            if encoded_feature is None:
                with torch.no_grad():
                        encoded_feature = self.encode(x_start)
            t, weight = self.T_sampler.sample(len(encoded_feature), torch_device)
            result_dict = self.latent_sampler.training_losses(model=self.diffusion_model.latent_net,
                                                                x_start=encoded_feature, t=t)
        else:
            raise NotImplementedError()
        
        return result_dict['loss']
        
    def manipulate(self, x, class_idx, cls_model, manipulate_weight):
        class_mlp_weight = cls_model.classifier.weight
        style_ch = class_mlp_weight.shape[1]
        target_class_weight = F.normalize(cls_model.classifier.weight[class_idx][None, :], dim=1)
        target_dim_tuple = tuple(range(1, target_class_weight))
        target_class_weight, *_ = _z_normalize(target_class_weight, target_dim_tuple, eps=1e-5)
        cond = self.encode(x)
        manipulated_cond = cond + manipulate_weight * math.sqrt(style_ch) * target_class_weight
        return manipulated_cond

        