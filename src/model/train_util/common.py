import torch

import torch
import random
from src.util.fold_unfold import fold_nd
def hook_fn(grad, mask, num_dims, patch_size, input_shape):
    """
    후킹 함수로, 그라디언트에 마스크를 적용하여 특정 패치의 그라디언트를 무시하도록 한다.
    
    Args:
    - grad (torch.Tensor): 입력 그라디언트
    - mask (torch.Tensor): 그라디언트를 무시할지 결정하는 마스크
    - num_dims (int): 입력 텐서의 차원 수 (4 또는 5)
    - patch_size (int): 패치 크기
    - input_shape (tuple): 입력 텐서의 원래 크기 (B, C, H, W 또는 B, C, D, H, W)
    
    Returns:
    - torch.Tensor: 마스크가 적용된 그라디언트
    """
    B, C = input_shape[:2]  # 배치 크기와 채널 크기
    if num_dims == 4:
        # 2D 텐서에 대한 그라디언트 후킹
        _, _, H, W = input_shape

        grad_unfold = grad.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        grad_unfold = grad_unfold.contiguous().view(B, C, -1, patch_size, patch_size)
        grad_unfold = grad_unfold * mask[..., None, None]  # 마스크 적용

        grad_fold = grad_unfold.view(B, C, -1, patch_size * patch_size)
        grad_fold = grad_fold.permute(0, 1, 3, 2)
        grad_fold = grad_fold.contiguous().view(B, C * patch_size * patch_size, -1)
        grad_fold = torch.nn.functional.fold(grad_fold, (H, W), kernel_size=patch_size, stride=patch_size)

    elif num_dims == 5:
        # 3D 텐서에 대한 그라디언트 후킹
        _, _, D, H, W = input_shape

        grad_unfold = (
            grad.unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .unfold(4, patch_size, patch_size)
        )
        grad_unfold = grad_unfold.contiguous().view(B, C, -1, patch_size, patch_size, patch_size)
        grad_unfold = grad_unfold * mask[..., None, None, None]  # 마스크 적용

        # 3D 텐서 형태로 fold 복원
        grad_fold = grad_unfold.view(B, C, -1, patch_size ** 3)
        grad_fold = grad_fold.permute(0, 2, 1, 3)
        grad_fold = grad_fold.contiguous().view(-1, C, patch_size ** 3)
        grad_fold = grad_fold.view(-1, C, patch_size, patch_size, patch_size)
        grad_fold = fold_nd(grad_fold, batch_size=B, output_size=D, patch_size=patch_size, stride=patch_size,
                            pad_size=0, img_dim=3, fold_idx=None)
    return grad_fold

def mask_gradient(input_tensor, target_tensor, patch_size=64, ignore_prob=0.95):
    """
    입력 텐서의 그라디언트를 무시하도록 설정하는 함수 (2D/3D 텐서 모두 지원)
    - 2D 텐서: (B, C, H, W)
    - 3D 텐서: (B, C, D, H, W)
    
    Args:
    - input_tensor (torch.Tensor): 입력 텐서 (배치 크기, 채널, 높이, 너비 또는 깊이)
    - target_tensor (torch.Tensor): 그라디언트를 후킹할 대상 텐서
    - patch_size (int): 패치 크기 (기본값 64)
    - ignore_prob (float): 그라디언트를 무시할 확률 (기본값 0.95)
    """
    target_tensor.requires_grad = True
    # 입력 텐서의 차원 수 확인
    num_dims = input_tensor.dim()

    if num_dims not in (4, 5):
        raise ValueError(f"입력 텐서의 차원 수는 4(2D) 또는 5(3D)여야 합니다. 현재 차원: {num_dims}")

    if num_dims == 4:
        # 2D 텐서의 경우: (B, C, H, W)
        B, C, H, W = input_tensor.size()
        input_shape = (B, C, H, W)

        # 2D 텐서를 8x8 또는 지정된 크기의 패치로 분할
        input_unfold = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        input_unfold = input_unfold.contiguous().view(B, C, -1, patch_size, patch_size)

    elif num_dims == 5:
        # 3D 텐서의 경우: (B, C, D, H, W)
        B, C, D, H, W = input_tensor.size()
        input_shape = (B, C, D, H, W)

        # 3D 텐서를 patch_size x patch_size x patch_size 크기의 패치로 분할
        input_unfold = (
            input_tensor.unfold(2, patch_size, patch_size)  # D 방향
            .unfold(3, patch_size, patch_size)  # H 방향
            .unfold(4, patch_size, patch_size)  # W 방향
        )
        input_unfold = input_unfold.contiguous().view(B, C, -1, patch_size, patch_size, patch_size)

    # 배경과 패치 마스크 생성
    tissue_mask = (input_unfold.mean(dim=list(range(-3, 0))) > 0.05).float()  # 마지막 2개 또는 3개 차원에 대해 평균 계산
    background_mask = 1 - tissue_mask

    # 무작위 확률에 따른 마스크 생성
    random_mask = torch.tensor([random.random() > ignore_prob for _ in range(tissue_mask.numel())],
                               device=input_tensor.device, dtype=torch.float32).view_as(tissue_mask)

    # 최종 마스크 생성
    mask = tissue_mask + background_mask * random_mask

    # 후킹 등록
    target_tensor.register_hook(lambda grad: hook_fn(grad, mask, num_dims, patch_size, input_shape))
    
def calculate_threshold(model):
    all_gradients = torch.cat([p.grad.view(-1)
                              for p in model.parameters() if p.grad is not None])
    gradient_mean = torch.mean(all_gradients)
    gradient_std = torch.std(all_gradients)
    threshold = gradient_mean + gradient_std * 1.96
    return threshold


def clip_gradients(model, threshold=None, use_outliers=False):
    if threshold is None:
        threshold = calculate_threshold(model)
    if use_outliers:
        outliers = [p for p in model.parameters()
                    if p.grad is not None and torch.max(torch.abs(p.grad)) > threshold]
        if outliers:
            max_outlier_value = max(
                [torch.max(torch.abs(p.grad)) for p in outliers])
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                            max_outlier_value)
    else:
        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)
