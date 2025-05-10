import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import random
from tqdm import tqdm
from functools import partial

from .util import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from .util import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, const_beta_schedule
from .util import exists, default, identity, identity, extract
from .util import ModelPrediction, forward_with_cond_scale
from einops import rearrange, reduce
DEFAULT_COND_SCALE = 6.
DEFAULT_RESCALED_PHI = 0.7

class MedSegDiff(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        loss_fn=F.mse_loss,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()

#         self.model = model if isinstance(model, Unet) else model.module
        self.model = model

        self.input_img_channels = self.model.input_img_channels
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.image_size = self.model.image_size
        self.objective = objective
        self.loss_fn = loss_fn
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        elif beta_schedule == 'const':
            betas = const_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength
        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        self.timepool = self.get_timepool()
        
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, x_self_cond=None, class_label=None,
                          cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI, clip_x_start=False):
        model_output = forward_with_cond_scale(self.model, x, t, c,
                                               x_self_cond=x_self_cond, class_label=class_label,
                                               cond_scale=cond_scale, rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, c, x_self_cond=None, class_label=None, 
                        cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI, clip_denoised=True):
        preds = self.model_predictions(x, t, c, x_self_cond, class_label, 
                                       cond_scale, rescaled_phi, clip_denoised)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, class_label=None, 
                 cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI, clip_denoised=True):
        b, device = x.size(0), x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, c=c,
                                                                          x_self_cond=x_self_cond, class_label=class_label,
                                                                          cond_scale=cond_scale, rescaled_phi=rescaled_phi,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, class_label,
                      cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI):
        batch, device, dtype = shape[0], self.betas.device, self.betas.dtype

        img = torch.randn(shape, device=device, dtype=dtype)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond, class_label,
                                         cond_scale=cond_scale, rescaled_phi=rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, class_label,
                    cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI, clip_denoised=True):
        batch, device, total_timesteps = shape[0], self.betas.device, self.num_timesteps
        sampling_timesteps, eta, objective = self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img, self_cond, class_label,
                                                             cond_scale=cond_scale, rescaled_phi=rescaled_phi,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, cond_img, class_label, cond_scale=DEFAULT_COND_SCALE, rescaled_phi=DEFAULT_RESCALED_PHI):
        batch_size, device, dtype = cond_img.size(0), self.device, self.dtype
        cond_img = cond_img.to(device=device, dtype=dtype)
        if isinstance(class_label, (list, tuple)):
            class_label = [item.to(device=device, dtype=torch.long)
                           for item in class_label]
        else:
            class_label = class_label.to(device=device, dtype=torch.long)
        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img, class_label,
                         cond_scale, rescaled_phi).float()

    def interpolate(self, x1, x2, classes, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, _ = self.p_sample(img, i, classes)

        return img
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, class_label, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():

                # predicting x_0

                x_self_cond = self.model_predictions(x, t, cond, None, class_label).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, cond, x_self_cond, class_label)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss, t

    def forward(self, img, cond_img, class_label, *args, **kwargs):
        if img.ndim == 3:
            img = rearrange(img, 'b h w -> b 1 h w')

        if cond_img.ndim == 3:
            cond_img = rearrange(cond_img, 'b h w -> b 1 h w')

        device, dtype = self.device, self.dtype
        img = img.to(device=device, dtype=dtype)
        cond_img = cond_img.to(device)
        
        if isinstance(class_label, (list, tuple)):
            class_label = [item.to(device=device, dtype=torch.long)
                           for item in class_label]
        else:
            class_label = class_label.to(device=device, dtype=torch.long)

        b, c, h, w, device, img_size, img_channels, mask_channels = *img.shape, img.device, self.image_size, self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        times = self.get_time(b).to(device=device, dtype=torch.long)
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, cond_img, class_label, *args, **kwargs)
    
    def get_time(self, batch_size):
        left_time_num = len(self.timepool)
        if left_time_num == 0:
            self.timepool = self.get_timepool()
        if left_time_num >= batch_size:
            time = [self.timepool.pop(random.randrange(len(self.timepool)))
                    for _ in range(batch_size)]
        else:
            batch_size_1 = left_time_num
            batch_size_2 = batch_size - batch_size_1
            time_1 = [self.timepool.pop(random.randrange(len(self.timepool)))
                    for _ in range(batch_size_1)]
            self.timepool = self.get_timepool()
            time_2 = [self.timepool.pop(random.randrange(len(self.timepool)))
                    for _ in range(batch_size_2)]
            time = time_1 + time_2
        time = torch.tensor(time)
        return time
    
    def get_timepool(self):
        return list(range(0, self.num_timesteps))