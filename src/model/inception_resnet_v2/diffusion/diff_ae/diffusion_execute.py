import math
import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import Dataset
from .resample import UniformSampler
from .diffusion_sampler import GaussianSampler
from .renderer import render_uncondition, render_condition
from src.model.train_util.common import _z_normalize, feature_z_normalize, z_normalize
from tqdm import tqdm

def identity_cat_fn(x, cond):
    return x

def identity_split_fn(x_cated):
    return x_cated, x_cated

class TargetStepDataset(Dataset):
    def __init__(self, dataset, batch_size, target_stepsize):
        self.dataset = dataset
        self.dataset_real_len = len(dataset)
        self.dataset_len = target_stepsize * batch_size
        assert self.dataset_real_len >= len(dataset), "target_stepsize * batch_size is smaller than dataset_real_len"
 
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx = idx % self.dataset_real_len
        return self.dataset[idx]

class LatentDataset(Dataset):
    def __init__(self, batch_cond):
        self.batch_cond = batch_cond

    def __len__(self):
        return len(self.batch_cond)
        
    def __getitem__(self, idx):
        return self.batch_cond[idx]


class AutoEncoder(nn.Module):
    def __init__(self, diffusion_model, train_mode, is_segmentation=False, image_mask_cat_fn=None, image_mask_split_fn=None,
                 sample_size=1, img_size=512, img_dim=2, T=1000, T_eval=20, T_sampler="uniform",
                 beta_scheduler="linear", latent_beta_scheduler="const0.008", spaced=True, rescale_timesteps=False,
                 gen_type="ddim", model_type="autoencoder", model_mean_type="eps", model_var_type="fixed_large", model_loss_type="mse",
                 latent_gen_type="ddim", latent_model_mean_type="eps", latent_model_var_type="fixed_large", latent_model_loss_type="l1",
                 latent_clip_sample=False, latent_znormalize=True,
                 fp16=False, train_pred_xstart_detach=True):
        super().__init__()
        train_mode_list = ["autoencoder", "latent_net", "ddpm", "autoencoder_latent_net"]
        if train_mode == "autoencoder":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is not None, "diffusion_model.encoder need to be set when you training autoencoder"
        elif train_mode in "latent_net":
            latent_net = getattr(diffusion_model, "latent_net", None)
            assert latent_net is not None, "diffusion_model.latent_net need to be set when you training latent_net"
        elif train_mode in "autoencoder_latent_net":
            encoder = getattr(diffusion_model, "encoder", None)
            latent_net = getattr(diffusion_model, "latent_net", None)
            assert encoder is not None, "diffusion_model.encoder need to be set when you training autoencoder_latent_net"
            assert latent_net is not None, "diffusion_model.latent_net need to be set when you training autoencoder_latent_net"
        elif train_mode == "ddpm":
            encoder = getattr(diffusion_model, "encoder", None)
            assert encoder is None, "diffusion_model.encoder need to be None when you training ddpm"
        else:
            raise NotImplementedError(f"{train_mode} is not in {train_mode_list}")
        
        ##########################
        latent_model_type = "ddpm"
        ##########################
        self.diffusion_model = diffusion_model
        self.train_mode = train_mode
        self.is_segmentation = is_segmentation
        
        if is_segmentation:
            image_mask_cat_fn = image_mask_cat_fn or identity_cat_fn
            image_mask_split_fn = image_mask_split_fn or identity_split_fn
            assert callable(image_mask_cat_fn)
            assert callable(image_mask_split_fn)
        else:
            assert image_mask_cat_fn is None, "do not provide image_mask_cat_fn when is_segmentation=True"
            assert image_mask_split_fn is None, "do not provide image_mask_split_fn when is_segmentation=True"

        self.image_mask_cat_fn = image_mask_cat_fn
        self.image_mask_split_fn = image_mask_split_fn
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
        ################## check all varialble time before init process ##################
        
        ##################################################################################
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
             "train_pred_xstart_detach": train_pred_xstart_detach,
        }
        self.sampler = self.get_sampler(T=self.T, T_eval=self.T)
        self.eval_sampler = self.get_sampler(T=self.T, T_eval=self.T_eval)

        self.common_latent_sampler_kwarg_dict = {
            "beta_scheduler": latent_beta_scheduler,
            "spaced": spaced,
            "rescale_timesteps": rescale_timesteps,
            "gen_type": latent_gen_type,
            "model_type": latent_model_type,
            "model_mean_type": latent_model_mean_type,
            "model_var_type": latent_model_var_type,
            "loss_type": latent_model_loss_type,
            "fp16": fp16,
            "train_pred_xstart_detach": train_pred_xstart_detach,
        }
        if train_mode in ["latent_net", "autoencoder_latent_net"]:
            self.latent_sampler = self.get_latent_sampler(T=self.T, T_eval=self.T)
            self.eval_latent_sampler = self.get_latent_sampler(T=self.T, T_eval=self.T_eval)

        # initial variables for consistent sampling
        # 표준 정규분포 정의 (평균=0, 표준편차=1)
        # 99% 신뢰구간에 해당하는 누적 확률값 (0.5%와 99.5%)
        self.register_buffer('x_T', self.get_noise(sample_size))
        
    def compute_cond_mean_std(self, train_dataloader, encode_process_fn=None, device=None, cond_save_path=None):
        device = device or torch.device("cuda")
        
        with torch.no_grad():
            batch_idx = 0
            cond_list = [None for _ in range(len(train_dataloader.dataset))]
            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                if encode_process_fn is not None:
                    cond = encode_process_fn(batch, self.encode, device)
                else:
                    batch = batch.to(device)
                    cond = self.encode(batch)
                cond = cond.cpu()
                for part_cond in cond:
                    cond_list[batch_idx] = part_cond
                    batch_idx += 1

        batch_cond = torch.stack(cond_list, dim=0)
        cond_mean = batch_cond.mean(dim=0, keepdim=True)
        cond_std = batch_cond.std(dim=0, keepdim=True)

        if cond_save_path is not None:
            torch.save({
                "batch_cond": batch_cond,
                "cond_mean": cond_mean,
                "cond_std": cond_std
            }, cond_save_path)

        return batch_cond, cond_mean, cond_std

    def load_cond_into(self, cond_file_path, set_buffer=False):
        cond_dict = torch.load(cond_file_path)
        batch_cond = cond_dict["batch_cond"]
        cond_mean = cond_dict["cond_mean"]
        cond_std = cond_dict["cond_std"]
        if set_buffer:
            self.register_buffer("cond_mean", cond_mean)
            self.register_buffer("cond_std", cond_std)
        return LatentDataset(batch_cond)

    def get_noise(self, batch_size, noise_channel=None):
        noise_channel = noise_channel or self.in_channel
        image_size_shape = tuple(self.img_size for _ in range(self.img_dim))
        image_size_shape = (batch_size, noise_channel, *image_size_shape)
        noise = self.sampler.get_noise_as_shape(image_size_shape, device="cpu")
        return noise

    def get_sampler(self, T, T_eval):
        return GaussianSampler(T=T, T_eval=T_eval, **self.common_sampler_kwarg_dict)
    
    def get_latent_sampler(self, T, T_eval):
        return GaussianSampler(T=T, T_eval=T_eval, **self.common_latent_sampler_kwarg_dict)
    
    def sample(self, N, device, T=None, T_latent=None, clip_denoised=True):
        noise = self.get_noise(N).to(device=device)
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
            latent_sampler = self.get_latent_sampler(T=self.T, T_eval=T_latent)

        if self.latent_znormalize:
            assert (self.cond_mean is not None) and (self.cond_std is not None), "get cond_mean and cond_std first by compute_cond_mean_std class fn"
            cond_mean = self.cond_mean
            cond_std = self.cond_std
        else:
            cond_mean = None
            cond_std = None
        pred_img = render_uncondition(
            self.diffusion_model,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            latent_clip_sample=self.latent_clip_sample,
            cond_mean=cond_mean,
            cond_std=cond_std,
            train_mode=self.train_mode,
            clip_denoised=clip_denoised,
            image_mask_cat_fn=self.image_mask_cat_fn,
            image_mask_split_fn=self.image_mask_split_fn
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def sample_segmentation(self, image, device, T=None, T_latent=None, clip_denoised=True):
        image = image.to(device)
        noise = self.get_noise(len(image)).to(device=device)
        _, noise = self.image_mask_split_fn(noise)
        image_noise = self.image_mask_cat_fn(image, noise)
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
            latent_sampler = self.get_latent_sampler(T=self.T, T_eval=T_latent)

        if self.latent_znormalize:
            assert (self.cond_mean is not None) and (self.cond_std is not None), "get cond_mean and cond_std first by compute_cond_mean_std class fn"
            cond_mean = self.cond_mean
            cond_std = self.cond_std
        else:
            cond_mean = None
            cond_std = None

        pred_img = render_uncondition(
            self.diffusion_model,
            image_noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            latent_clip_sample=self.latent_clip_sample,
            cond_mean=cond_mean,
            cond_std=cond_std,
            train_mode=self.train_mode,
            clip_denoised=clip_denoised,
            image_mask_cat_fn=self.image_mask_cat_fn,
            image_mask_split_fn=self.image_mask_split_fn
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
                                        train_mode=self.train_mode,
                                        clip_denoised=clip_denoised,
                                        image_mask_cat_fn=self.image_mask_cat_fn,
                                        image_mask_split_fn=self.image_mask_split_fn)
        else:
            pred_img = render_uncondition(self.diffusion_model,
                                          noise, sampler, latent_sampler=None,
                                          latent_clip_sample=self.latent_clip_sample,
                                          train_mode=self.train_mode,
                                          clip_denoised=clip_denoised,
                                          image_mask_cat_fn=self.image_mask_cat_fn,
                                          image_mask_split_fn=self.image_mask_split_fn)
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
        out = sampler.ddim_reverse_sample_loop(self.diffusion_model, x,
                                               model_kwargs={'cond': cond},
                                               clip_denoised=clip_denoised)
        return out['sample']

    def encode_stochastic_segmentation(self, image, mask, image_encoded=None, T=None, clip_denoised=True):
        x = self.image_mask_cat_fn(image, mask)
        if image_encoded is None:
            image_encoded = self.diffusion_model.encode(image)
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.get_sampler(T=self.T, T_eval=T)
        out = sampler.ddim_reverse_sample_loop(self.diffusion_model, x,
                                               image_mask_cat_fn=self.image_mask_cat_fn,
                                               image_mask_split_fn=self.image_mask_split_fn,
                                               model_kwargs={'cond': image_encoded},
                                               clip_denoised=clip_denoised)
        return out['sample']


    def forward(self, x_start=None, encoded_feature=None, cond_start=None):
        if self.is_segmentation and (self.train_mode != "latent_net"):
            return self._get_loss_segmentation(x_start=x_start, cond_start=cond_start)
        else:
            return self._get_loss(x_start=x_start, encoded_feature=encoded_feature, cond_start=cond_start)
    
    # Check: remove kwarg ema_model
    def _get_loss(self, x_start=None, encoded_feature=None, cond_start=None):
        exists_encoded_feature = encoded_feature is not None
        exists_cond_start = cond_start is not None
        
        if self.train_mode in ["autoencoder", "autoencoder_latent_net"]:
            assert_info_str = "Either encoded_feature or cond_start should be provided, but not both."
            assert (exists_encoded_feature or exists_cond_start) and not (exists_encoded_feature and exists_cond_start), assert_info_str
        elif self.train_mode in ["latent_net", "ddpm"]:
            assert (exists_encoded_feature and exists_cond_start) is False, "Neither encoded_feature or cond_start should not be provided"

        if cond_start is not None:
            encoded_feature = self.encode(cond_start)
        x_start_device = getattr(x_start, "device", None)
        encoded_feature_device = getattr(encoded_feature, "device", None)
        torch_device = x_start_device or encoded_feature_device
        # with autocast(device_type=torch_device, enabled=False):
        if self.train_mode in ["autoencoder", "ddpm"]:
            t, weight = self.T_sampler.sample(len(x_start), torch_device)
            if encoded_feature is None:
                result_dict = self.sampler.training_losses(model=self.diffusion_model,
                                                            x_start=x_start, t=t)
            else:
                result_dict = self.sampler.training_losses(model=self.diffusion_model,
                                                            x_start=x_start, t=t,
                                                            model_kwargs={'cond': encoded_feature})

        elif self.train_mode == "latent_net":
            t, weight = self.T_sampler.sample(len(x_start), torch_device)
            result_dict = self.latent_sampler.training_losses(model=self.diffusion_model.latent_net,
                                                                x_start=x_start, t=t)
            
        elif self.train_mode == "autoencoder_latent_net":
    
            t, weight = self.T_sampler.sample(len(x_start), torch_device)
            if encoded_feature is None:
                result_dict = self.sampler.training_losses(model=self.diffusion_model,
                                                            x_start=x_start, t=t)
            else:
                result_dict = self.sampler.training_losses(model=self.diffusion_model,
                                                            x_start=x_start, t=t,
                                                            model_kwargs={'cond': encoded_feature})
                t, weight = self.T_sampler.sample(len(encoded_feature), torch_device)
                result_dict_latent = self.latent_sampler.training_losses(model=self.diffusion_model.latent_net,
                                                                        x_start=encoded_feature.detach(), t=t)
            return result_dict['loss'], result_dict_latent['loss']
        else:
            raise NotImplementedError()
        return result_dict['loss']

    def _get_loss_segmentation(self, x_start=None, cond_start=None):
        assert x_start is not None, "get_loss_segmentation requires x_start with image shape like (B, C, H, W)"
        assert cond_start is not None, "get_loss_segmentation requires cond_start with image shape like (B, C, H, W)"

        x_start_device = getattr(x_start, "device", None)
        cond_start_device = getattr(cond_start, "device", None)
        torch_device = x_start_device
        # with autocast(device_type=torch_device, enabled=False):
        if self.train_mode in ["autoencoder", "ddpm"]:
            t, weight = self.T_sampler.sample(len(x_start), torch_device)
            encoded_feature = self.encode(cond_start)
            result_dict = self.sampler.training_losses_segmentation(model=self.diffusion_model,
                                                                    image=x_start, mask=cond_start, image_encoded=encoded_feature, t=t,
                                                                    image_mask_cat_fn=self.image_mask_cat_fn, model_kwargs=None)
        elif self.train_mode == "latent_net":
            raise NotImplementedError()
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

        