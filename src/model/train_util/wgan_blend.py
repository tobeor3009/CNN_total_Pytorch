import torch


# def compute_gradient_penalty_2d(D, real_samples, fake_samples, blend_labels):
#     """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2 using blended images."""
#     device = real_samples.device
#     alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
#     interpolates = (alpha * real_samples + (1 - alpha)
#                     * fake_samples).requires_grad_(True)
#     d_output = D(interpolates)
#     if type(d_output) == list:
#         _, d_interpolates = d_output
#     else:
#         d_interpolates = d_output

#     # We will use only the outputs of the blended regions for calculating the penalty
#     d_interpolates = torch.where(blend_labels, d_interpolates,
#                                  torch.zeros_like(d_interpolates).to(device))

#     fake = torch.ones(d_interpolates.size()).to(device)

#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         retain_graph=True,
#         create_graph=True,
#         only_inputs=True
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

def compute_gradient_penalty_2d(D, real_samples, fake_samples, blend_labels):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2 using blended images."""
    device = real_samples.device
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    blend_labels = F.interpolate(blend_labels.float(), size=real_samples.shape[2:],
                                 mode='bilinear', align_corners=False) > 0.5
    real_samples = torch.where(~blend_labels,
                               real_samples,
                               torch.zeros_like(real_samples))
    fake_samples = torch.where(blend_labels,
                               fake_samples,
                               torch.zeros_like(fake_samples))
    interpolates = (alpha * real_samples + (1 - alpha)
                    * fake_samples).requires_grad_(True)
    d_output = D(interpolates)
    if type(d_output) == list:
        _, d_interpolates = d_output
    else:
        d_interpolates = d_output

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


def get_blend_images_2d(target_images, fake_images, patch_size=64, disc_channel=1):
    # 이미지의 크기 및 패치에 따른 격자 수를 확인
    B, C, H, W = target_images.shape
    grid_size = H // patch_size

    # 레이블 배열 초기화
    label_array = torch.zeros((B, grid_size, grid_size), dtype=bool)
    blended_image = torch.zeros_like(target_images)

    for i in range(grid_size):
        for j in range(grid_size):
            # 0 또는 1을 무작위로 선택하여 타겟 이미지 또는 가짜 이미지 패치를 선택
            choice = torch.randint(0, 2, (B,))
            for b in range(B):
                if choice[b].item() == 0:
                    blended_image[b, :, i * patch_size:(i + 1) * patch_size,
                                  j * patch_size:(j + 1) * patch_size] = target_images[b, :,
                                                                                       i * patch_size:(i + 1) * patch_size,
                                                                                       j * patch_size:(j + 1) * patch_size]
                    label_array[b, i, j] = True
                else:
                    blended_image[b, :, i * patch_size:(i + 1) * patch_size,
                                  j * patch_size:(j + 1) * patch_size] = fake_images[b, :,
                                                                                     i * patch_size:(i + 1) * patch_size,
                                                                                     j * patch_size:(j + 1) * patch_size]
    label_array = label_array[:, None].repeat(1, disc_channel, 1, 1)
    return blended_image, label_array.to(blended_image.device)
