import torch


def compute_gradient_penalty(D, real_samples, fake_samples, mode="2d"):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    device = real_samples.device
    dtype = real_samples.dtype
    if mode == "2d":
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device,
                                                             dtype=dtype)
    elif mode == "3d":
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device,
                                                                dtype=dtype)

    interpolates = (alpha * real_samples + (1 - alpha)
                    * fake_samples).requires_grad_(True)
    _, d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
