from typing import NamedTuple, Callable
from torch.amp import autocast
import numpy as np
import torch
import math
from .choices import ModelType, GenerativeType
from .nn import mean_flat
import os

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_const_betas(const_coef, num_diffusion_timesteps):
    scale = 1000 / num_diffusion_timesteps
    betas = np.array([scale * const_coef] * num_diffusion_timesteps)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_spaced_diffusion_betas(alphas_cumprod, use_timesteps):
    last_alpha_cumprod = 1.0
    new_betas, timestep_map = [], []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            # getting the new betas of the new timesteps
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = np.array(new_betas, dtype=np.float64)
    return new_betas, timestep_map

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_timesteps, spaced):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_name == "cosine":
        betas = betas_for_alpha_bar(num_diffusion_timesteps,
                                    lambda t: torch.cos((t + 0.008) / 1.008 * torch.pi / 2)**2)
    elif schedule_name[:5] == "const":
        const_coef = float(schedule_name[5:])
        betas = get_const_betas(const_coef, num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    betas = betas.astype(np.float64)
    if spaced:
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        betas, timestep_map = get_spaced_diffusion_betas(alphas_cumprod, use_timesteps)
    else:
        timestep_map = None
    return betas, timestep_map

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                  ((mean1 - mean2)**2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                 torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

class DummyModel(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: torch.Tensor

class _WrappedModel:
    """
    converting the supplied t's to the old t's scales.
    """
    def __init__(self, model, timestep_map, rescale_timesteps,
                 original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def forward(self, x, t, t_cond=None, **kwargs):
        """
        Args:
            t: t's with differrent ranges (can be << T due to smaller eval T) need to be converted to the original t's
            t_cond: the same as t but can be of different values
        """
        map_tensor = torch.tensor(self.timestep_map,
                                device=t.device,
                                dtype=t.dtype)
        def do(t):
            new_ts = map_tensor[t]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts

        if t_cond is not None:
            # support t_cond
            t_cond = do(t_cond)

        return self.model(x=x, t=do(t), t_cond=t_cond, **kwargs)

    def __getattr__(self, name):
        # allow for calling the model's methods
        if hasattr(self.model, name):
            func = getattr(self.model, name)
            return func
        raise AttributeError(name)


def get_l1_loss(y_pred, y_true):
    loss = mean_flat((y_pred - y_true).abs())
    return loss

def get_mse_loss(y_pred, y_true):
    loss = mean_flat((y_pred - y_true)**2)
    return loss

class GaussianSampler():
    def __init__(self,
                 T=1000, T_eval=20, beta_scheduler="linear", spaced=True, rescale_timesteps=False,
                 gen_type="ddim", model_type="autoencoder", model_mean_type="eps", model_var_type="fixed_large",
                 loss_type="l1", fp16=False, train_pred_xstart_detach=True, noise_clip_ratio=1, use_truncated_noise=False):
        #########################################################
        if gen_type == "ddpm":
            section_counts = [T_eval]
        elif gen_type == "ddim":
            section_counts = f'ddim{T_eval}'
        else:
            raise NotImplementedError()
        ##################
        self.spaced = spaced
        self.use_timesteps = space_timesteps(T, section_counts)
        self.gen_type = GenerativeType(gen_type)
        self.model_type = ModelType(model_type)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        if self.loss_type == "l1":
            self.loss_fn = get_l1_loss
        elif self.loss_type == "mse":
            self.loss_fn = get_mse_loss
        else:
            raise NotImplementedError()
        
        self.rescale_timesteps = rescale_timesteps
        self.fp16 = fp16
        self.train_pred_xstart_detach = train_pred_xstart_detach
        self.use_truncated_noise = use_truncated_noise
        ################## check all varialble time before init process ##################
        assert self.model_mean_type in ["eps", "eps + x_start"]
        assert self.model_var_type in ["fixed_small", "fixed_large"]
        assert self.loss_type in ["l1", "mse"]
        assert isinstance(self.rescale_timesteps, bool)
        assert isinstance(self.fp16, bool)
        assert isinstance(self.train_pred_xstart_detach, bool)
        ###################################################################################

        betas, self.timestep_map = get_named_beta_schedule(beta_scheduler, T, self.use_timesteps, spaced)
        self.betas = np.array(betas, dtype=np.float64)
        self.original_num_steps = T
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))
        
        if use_truncated_noise:
            self.z_upper_limit = 1.1
            self.z_lower_limit = -1.1
        else:
            normal_dist = torch.distributions.Normal(0, 1)
            noise_clip_ratio = (1 - noise_clip_ratio) / 2
            self.z_upper_limit = normal_dist.icdf(torch.tensor(1 - noise_clip_ratio)).item()
            self.z_lower_limit = normal_dist.icdf(torch.tensor(noise_clip_ratio)).item()

    def _wrap_model(self, model):
        if self.spaced:
            if isinstance(model, _WrappedModel):
                return model
            return _WrappedModel(model, self.timestep_map, self.rescale_timesteps,
                                self.original_num_steps)
        else:
            return model
    
    def truncated_normal_tensor(self, uniform_rand, mean=0, std=1):
        """
        잘린 정규분포(Truncated Normal Distribution)를 주어진 shape으로 생성
        :param mean: 평균
        :param std: 표준편차
        :param lower: 하한
        :param upper: 상한
        :param shape: 출력 텐서의 크기
        :return: 잘린 범위 내에서 샘플링된 PyTorch 텐서
        """
        # CDF 계산
        lower_cdf = 0.5 * (1 + math.erf((self.z_lower_limit - mean) / (std * (2 ** 0.5))))
        upper_cdf = 0.5 * (1 + math.erf((self.z_upper_limit - mean) / (std * (2 ** 0.5))))

        # Uniform 분포에서 샘플링
        u = uniform_rand * (upper_cdf - lower_cdf) + lower_cdf

        # Inverse CDF 변환
        samples = mean + std * (2 ** 0.5) * torch.erfinv(2 * u - 1)
        return samples
    
    def clip_noise_abnormal(self, noise):
        noise = noise.clamp(self.z_lower_limit, self.z_upper_limit)
        return noise
    
    def get_noise_like(self, refer_x):
        if self.use_truncated_noise:
            noise = torch.rand_like(refer_x)
            noise = self.truncated_normal_tensor(noise)
        else:
            noise = torch.randn_like(refer_x)
            noise = self.clip_noise_abnormal(noise)
        return noise

    def get_noise_as_shape(self, shape, device, dtype=torch.float32):
        if self.use_truncated_noise:
            noise = torch.rand(*shape, device=device, dtype=dtype)
            noise = self.truncated_normal_tensor(noise)
        else:
            noise = torch.randn(*shape, device=device, dtype=dtype)
            noise = self.clip_noise_abnormal(noise)
        return noise

    def training_losses(self, model,
                        x_start: torch.Tensor, t: torch.Tensor,
                        model_kwargs=None,
                        noise: torch.Tensor = None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        model = self._wrap_model(model)
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = self.get_noise_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {'x_t': x_t}

        # with autocast(device_type=x_start.device, enabled=self.fp16):
        # x_t is static wrt. to the diffusion process
        model_forward = model.forward(x=x_t.detach(),
                                        t=self._scale_timesteps(t),
                                        x_start=x_start.detach(),
                                        **model_kwargs)
        model_output = model_forward.pred

        _model_output = model_output
        if self.train_pred_xstart_detach:
            _model_output = _model_output.detach()
        # get the pred xstart
        p_mean_var = self.p_mean_variance(
            model=DummyModel(pred=_model_output),
            # gradient goes through x_t
            x=x_t,
            t=t,
            clip_denoised=False)
        terms['pred_xstart'] = p_mean_var['pred_xstart']

        # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        if self.model_mean_type == "eps":
            target = noise
            assert model_output.shape == target.shape == x_start.shape
            terms["eps"] = self.loss_fn(target, model_output)
            terms["loss"] = terms["eps"]
        elif self.model_mean_type == "eps + x_start":
            terms["eps"] = self.loss_fn(noise, model_output)
            terms["recon"] = get_l1_loss(x_start, terms['pred_xstart'])
            terms["loss"] = terms["eps"] * 0.1 + terms["recon"] * 0.9
        elif self.model_mean_type == "eps + consistency":
            x_t_from_noise = self.q_sample(self.get_noise_like(noise), t, noise=noise)
            model_forward_from_noise = model.forward(x=x_t_from_noise.detach(),
                                                    t=self._scale_timesteps(t),
                                                    x_start=x_start.detach(),
                                                    **model_kwargs)
            model_output_from_noise = model_forward_from_noise.pred
            terms["eps"] = self.loss_fn(target, model_output)
            terms["consitency"] = self.loss_fn(model_output_from_noise, model_output)
            terms["loss"] = terms["eps"] * 0.9 + terms["consitency"] * 0.1
        else:
            raise NotImplementedError()
        
        if "vb" in terms:
            # if learning the variance also use the vlb loss
            terms["loss"] = terms["loss"] + terms["vb"]
        return terms

    def training_losses_segmentation(self, model,
                                    image: torch.Tensor, mask: torch.Tensor, image_encoded: torch.Tensor,
                                    t: torch.Tensor, image_mask_cat_fn: Callable,
                                    model_kwargs=None, noise: torch.Tensor = None):
        model = self._wrap_model(model)
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = {'cond': image_encoded}
        
        if noise is None:
            noise = self.get_noise_like(mask)
        
        noise_mask = self.q_sample(mask, t, noise=noise)
        x_start = image_mask_cat_fn(image, mask)
        x_t = image_mask_cat_fn(image, noise_mask)
        terms = {'x_t': x_t}

        # with autocast(device_type=x_start.device, enabled=self.fp16):
        # x_t is static wrt. to the diffusion process
        model_forward = model.forward(x=x_t.detach(),
                                        t=self._scale_timesteps(t),
                                        x_start=x_start.detach(),
                                        **model_kwargs)
        model_output = model_forward.pred
        model_anch_output = model_forward.pred_anch
        _model_output = model_output
        if self.train_pred_xstart_detach:
            _model_output = _model_output.detach()
        
        # get the pred xstart
        p_mean_var = self.p_mean_variance(
            model=DummyModel(pred=_model_output),
            # gradient goes through x_t
            x=noise_mask,
            t=t,
            clip_denoised=False)
        terms['pred_xstart'] = p_mean_var['pred_xstart']

        assert model_output.shape == noise.shape == mask.shape
        if self.model_mean_type == "eps":
            target = noise
            terms["eps"] = self.loss_fn(target, model_output)
            terms["loss"] = terms["eps"]
        elif self.model_mean_type == "eps + x_start":
            terms["loss_noise"] = self.loss_fn(noise, model_output)
            terms["loss_mask"] = self.loss_fn(mask, model_anch_output)
            terms["loss"] = terms["loss_noise"] * 0.1 + terms["loss_mask"] * 0.9

        if "vb" in terms:
            # if learning the variance also use the vlb loss
            terms["loss"] = terms["loss"] + terms["vb"]
        return terms

    def sample(self, model,
               shape=None,
               image_mask_cat_fn=None,
               image_mask_split_fn=None,
               noise=None,
               cond=None,
               x_start=None,
               clip_denoised=True,
               model_kwargs=None,
               progress=False):
        """
        Args:
            x_start: given for the autoencoder
        """
        if model_kwargs is None:
            model_kwargs = {}
            if self.model_type.has_autoenc():
                model_kwargs['x_start'] = x_start
                model_kwargs['cond'] = cond

        if self.gen_type == GenerativeType.ddpm:
            return self.p_sample_loop(model,
                                      shape=shape,
                                      noise=noise,
                                      clip_denoised=clip_denoised,
                                      model_kwargs=model_kwargs,
                                      progress=progress)
        elif self.gen_type == GenerativeType.ddim:
            return self.ddim_sample_loop(model,
                                         shape=shape,
                                         image_mask_cat_fn=image_mask_cat_fn,
                                         image_mask_split_fn=image_mask_split_fn,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress)
        else:
            raise NotImplementedError()
            
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                        x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                            t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = self.get_noise_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t,
                        image_mask_split_fn=None,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        model = self._wrap_model(model)
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B, )
        # with torch.amp.autocast(self.fp16):
        model_forward = model.forward(x=x,
                                        t=self._scale_timesteps(t),
                                        **model_kwargs)
        model_output = model_forward.pred

        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        if self.model_var_type == "fixed_large":
            model_variance = np.append(self.posterior_variance[1], self.betas[1:])
            model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        elif self.model_var_type == "fixed_small":
            model_variance = self.posterior_variance
            model_log_variance = self.posterior_log_variance_clipped
        else:
            raise NotImplementedError()
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            else:
                return self.clip_noise_abnormal(x)
        
        if image_mask_split_fn is not None:
            # x means mask
            image, mask = image_mask_split_fn(x)
            target = mask
        else:
            target = x

        model_variance = _extract_into_tensor(model_variance, t, target.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, target.shape)

        if self.model_mean_type in ["eps", "eps + x_start"]:
            pred_xstart = self._predict_xstart_from_eps(x_t=target, t=t, eps=model_output)
            pred_xstart = process_xstart(pred_xstart)
        else:
            raise NotImplementedError()
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=target, t=t)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == target.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t)

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, scaled_xstart.shape)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        """
        Args:
            scaled_xstart: is supposed to be sqrt(alphacum) * x_0
        """
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.spaced:
            return t
        else:# Scaling is done by the wrapped model.
            if self.rescale_timesteps:
                # scale t to be maxed out at 1000 steps
                return t.float() * (1000.0 / self.num_timesteps)
            return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        cond_fn = self._wrap_model(cond_fn)
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var["mean"].float() +
                    p_mean_var["variance"] * gradient.float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        cond_fn = self._wrap_model(cond_fn)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self, model, x, t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = self.get_noise_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn,
                                              out,
                                              x,
                                              t,
                                              model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * torch.exp(
            0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self, model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self, model,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = self.get_noise_as_shape(shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = torch.tensor([i] * shape[0], device=device)
            t = torch.tensor([i] * len(img), device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self, model, x, t,
        image_mask_split_fn=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model, x, t,
            image_mask_split_fn=image_mask_split_fn,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if image_mask_split_fn is not None:
            # x means mask
            image, mask = image_mask_split_fn(x)
            target = mask
        else:
            target = x
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, target, t,
                                       model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(target, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, target.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, target.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        # if mask_channel is not None:
        #     noise = self.get_noise_like(x[:, -mask_channel:])
        #     barch_size, x_start_channel, *img_shape = x.shape
        #     permute_dim_tuple = tuple((0, 2, 1, *range(3, 3 + len(img_shape))))
        #     assert x_start_channel % mask_channel == 0
        #     multiple_channel = x_start_channel // mask_channel
        #     noise = noise.unsqueeze(2)
        #     noise = noise.expand(barch_size, mask_channel, multiple_channel, *img_shape)
        #     noise = noise.permute(*permute_dim_tuple)
        #     noise = noise.reshape(barch_size, x_start_channel, *img_shape)
        # else:
        #     noise = self.get_noise_like(x)
        noise = self.get_noise_like(out["pred_xstart"])
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_prev) +
                     torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_reverse_sample(
        self, model, x, t,
        image_mask_split_fn=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ? 
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model, x, t,
            image_mask_split_fn=image_mask_split_fn,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if image_mask_split_fn is not None:
            # x means mask
            image, mask = image_mask_split_fn(x)
            target = mask
        else:
            target = x
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, target.shape)
               * target - out["pred_xstart"]) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, target.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t,
                                              target.shape)

        # Equation 12. reversed  (DDIM paper)  (torch.sqrt == torch.sqrt)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_next) +
                     torch.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self, model, x,
        image_mask_cat_fn=None,
        image_mask_split_fn=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.num_timesteps))
        if image_mask_split_fn is not None:
            image, mask = image_mask_split_fn(x)
        else:
            image, mask = None, None
        sample = x
        for time_step in indices:
            t = torch.tensor([time_step] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(model, sample, t=t,
                                               image_mask_split_fn=image_mask_split_fn,
                                               clip_denoised=clip_denoised,
                                               denoised_fn=denoised_fn,
                                               model_kwargs=model_kwargs,
                                               eta=eta)
                if image_mask_cat_fn is not None:
                    # ddim_reverse_sample output reconstruct about mask, not image
                    out['sample'] = image_mask_cat_fn(image, out['sample'])
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }

    def ddim_sample_loop(
        self, model,
        shape=None,
        image_mask_cat_fn=None,
        image_mask_split_fn=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                image_mask_cat_fn=image_mask_cat_fn,
                image_mask_split_fn=image_mask_split_fn,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self, model,
        shape=None,
        image_mask_cat_fn=None,
        image_mask_split_fn=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            pred_image = noise
        else:
            assert isinstance(shape, (tuple, list))
            pred_image = self.get_noise_as_shape(shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        
        if image_mask_split_fn is not None:
            assert image_mask_cat_fn is not None
            image, mask = image_mask_split_fn(noise)
        else:
            image, mask = None, None

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = torch.tensor([i] * len(pred_image), device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model, pred_image, t,
                    image_mask_split_fn=image_mask_split_fn,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                if image_mask_cat_fn is not None:
                    out["sample"] = image_mask_cat_fn(image, out["sample"])
                yield out
                pred_image = out["sample"]

                    
    def _vb_terms_bpd(self, model,
                      x_start,
                      x_t,
                      t,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model,
                                   x_t,
                                   t,
                                   clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"],
                       out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            'model_forward': out['model_forward'],
        }

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size,
                      device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self,
                      model,
                      x_start,
                      clip_denoised=True,
                      model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = self.get_noise_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start)**2))
            eps = self._predict_eps_from_xstart(x_t, t_batch,
                                                out["pred_xstart"])
            mse.append(mean_flat((eps - noise)**2))

        vb = torch.stack(vb, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }