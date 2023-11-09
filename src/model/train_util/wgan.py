import torch
import torch.nn.functional as F


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    alpha = torch.rand_like(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha)
                    * fake_samples).requires_grad_(True)
    d_output = D(interpolates)
    if type(d_output) == list:
        _, d_interpolates = d_output
    else:
        d_interpolates = d_output
    fake = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # chanege pradient penalty code for fix gradient's stable when training fp16
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = F.mse_loss(gradients_norm,
                                  torch.ones_like(gradients_norm))
    return gradient_penalty
