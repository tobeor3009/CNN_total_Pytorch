import torch
import torch.nn.functional as F
import random


def compute_gradient_penalty_blend(D, real_samples, fake_samples, blend_labels):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2 using blended images."""
    inter_mode = "nearest"
    device = real_samples.device
    alpha = torch.rand_like(real_samples)

    blend_labels = F.interpolate(blend_labels.to(device,
                                                 dtype=torch.float32),
                                 size=real_samples.shape[2:],
                                 mode=inter_mode).detach() > 0.5
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


def get_blend_images_2d(target_images, fake_images, patch_size=64, disc_channel=1):
    # 이미지의 크기 및 패치에 따른 격자 수를 확인
    B, C, H, W = target_images.shape
    grid_H = H // patch_size
    grid_W = W // patch_size
    # 레이블 배열 초기화
    label_array = torch.zeros((B, grid_H, grid_W), dtype=bool)
    blended_image = torch.zeros_like(target_images)
    for b in range(B):
        rand = random.random()
        choice = torch.rand([grid_H, grid_W])
        for i in range(grid_H):
            for j in range(grid_W):
                if choice[i, j].item() > rand:
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


def get_blend_images_3d(target_images, fake_images, patch_size=64, disc_channel=1):
    # 이미지의 크기 및 패치에 따른 격자 수를 확인
    B, C, D, H, W = target_images.shape
    grid_size_d = D // patch_size
    grid_size_h = H // patch_size
    grid_size_w = W // patch_size

    # 레이블 배열 초기화
    label_array = torch.zeros((B,
                               grid_size_d,
                               grid_size_h,
                               grid_size_w), dtype=bool)
    blended_image = torch.zeros_like(target_images)
    for b in range(B):
        rand = random.random()
        choice = torch.rand([grid_size_d, grid_size_h, grid_size_w])
        for i in range(grid_size_d):
            for j in range(grid_size_h):
                for k in range(grid_size_w):
                    if choice[i, j, k].item() >= rand:
                        blended_image[b, :,
                                      i * patch_size:(i + 1) * patch_size,
                                      j * patch_size:(j + 1) * patch_size,
                                      k * patch_size:(k + 1) * patch_size] = target_images[b, :,
                                                                                           i * patch_size:(i + 1) * patch_size,
                                                                                           j * patch_size:(j + 1) * patch_size,
                                                                                           k * patch_size:(k + 1) * patch_size]
                        label_array[b, i, j, k] = True
                    else:
                        blended_image[b, :,
                                      i * patch_size:(i + 1) * patch_size,
                                      j * patch_size:(j + 1) * patch_size,
                                      k * patch_size:(k + 1) * patch_size] = fake_images[b, :,
                                                                                         i * patch_size:(i + 1) * patch_size,
                                                                                         j * patch_size:(j + 1) * patch_size,
                                                                                         k * patch_size:(k + 1) * patch_size]

    label_array = label_array[:, None].repeat(1, disc_channel, 1, 1, 1)
    return blended_image, label_array.to(blended_image.device)
