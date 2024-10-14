import torch
import math
from torch.nn import functional as F

def process_batch_split(input_tensor, target_model, batch_size=32):
    data_num = input_tensor.shape[0]
    batch_num = math.ceil(data_num / batch_size)
    pred_tensor = []
    for batch_idx in range(batch_num):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, data_num)
        pred_tensor_part = target_model(input_tensor[start_idx:end_idx])
        if isinstance(pred_tensor_part, list):
            pred_tensor_part = pred_tensor_part[0]
        pred_tensor.append(pred_tensor_part)
    pred_tensor = torch.cat(pred_tensor, axis=0)
    return pred_tensor

def crop_full_drr_tensor(image_tensor, size, patch_size, stride, pad_size):
    
    padded_size = size + pad_size * 2
    # 입력 텐서 크기 확인
    B, C, H, W = image_tensor.shape
    assert C == 2, f"channel must be 2, but image_shape shape is {B, C, H, W}"
    pad_image_tensor = F.pad(image_tensor, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    
    # 각 배치별로 랜덤 크롭 수행
    cropped_tensors = []
    idx_range = range(0, padded_size - patch_size + 1, stride)
    for b in range(B):
        for d_idx in idx_range:
            for h_idx in idx_range:
                for w_idx in idx_range:
                    cropped_lat = pad_image_tensor[b, 0, d_idx:d_idx+patch_size, h_idx:h_idx+patch_size]
                    cropped_ap = pad_image_tensor[b, 1, d_idx:d_idx+patch_size, w_idx:w_idx+patch_size]
                    # 각 배치에 대해 랜덤 위치에서 크롭 수행
                    cropped_image = torch.stack([cropped_lat, cropped_ap], dim=0)
                    cropped_tensors.append(cropped_image)

    # 결과 텐서로 변환 (B * num_crop_repeat, C, crop_h, crop_w)
    cropped_tensor_batch = torch.stack(cropped_tensors, dim=0)
    return cropped_tensor_batch

def crop_drr_tensor(image_tensor, cropped_centers, crop_size=(64, 64)):
    # 입력 텐서 크기 확인
    # real_batch_size, cropped_centers_num, num_crop_repeat = 2, 24, 12
    real_batch_size, C, H, W = image_tensor.shape
    cropped_centers_num = cropped_centers.size(0)
    num_crop_repeat = cropped_centers_num // real_batch_size
    
    crop_h, crop_w = crop_size
    
    assert C == 2, f"channel must be 2, but image_shape shape is {real_batch_size, C, H, W}"
    if H < crop_h or W < crop_w:
        raise ValueError(f"입력 이미지 크기 {image_tensor.shape}가 크롭 크기 {crop_size}보다 작습니다.")

    # 각 배치별로 랜덤 크롭 수행
    cropped_tensors = []
    
    for repeat_idx in range(num_crop_repeat):
        for b in range(real_batch_size):
        # 배치 크기만큼의 랜덤 크롭 위치 설정
            crop_idx = num_crop_repeat * b + repeat_idx
            start_d = cropped_centers[crop_idx, 0]
            start_h = cropped_centers[crop_idx, 1]
            start_w = cropped_centers[crop_idx, 2]
            cropped_lat = image_tensor[b, 0, start_d:start_d+crop_h, start_h:start_h+crop_w]
            cropped_ap = image_tensor[b, 1, start_d:start_d+crop_h, start_w:start_w+crop_w]
            # 각 배치에 대해 랜덤 위치에서 크롭 수행
            cropped_image = torch.stack([cropped_lat, cropped_ap], dim=0)
            cropped_tensors.append(cropped_image)

    # 결과 텐서로 변환 (B * num_crop_repeat, C, crop_h, crop_w)
    cropped_tensor_batch = torch.stack(cropped_tensors, dim=0)
    return cropped_tensor_batch

def random_crop_3d_tensor(image_tensor, crop_size=(64, 64, 64), num_crop_repeat=1):
    """
    3D torch 텐서를 입력으로 받아 각 배치별 랜덤 위치의 크롭 패치를 여러 번 뜯는 함수.

    Args:
        image_tensor (torch.Tensor): 입력 3D 이미지 텐서, 크기 (B, C, D, H, W)
        crop_size (tuple): 크롭할 3D 이미지의 크기 (기본값: (64, 64, 64))
        num_crop_repeat (int): 각 배치당 생성할 크롭 패치 수 (기본값: 1)

    Returns:
        torch.Tensor: 랜덤 크롭 후의 이미지 텐서, 크기 (B * num_crop_repeat, C, crop_size[0], crop_size[1], crop_size[2])
    """
    # 입력 텐서 크기 확인
    B, C, D, H, W = image_tensor.shape
    crop_d, crop_h, crop_w = crop_size

    if D < crop_d or H < crop_h or W < crop_w:
        raise ValueError(f"입력 이미지 크기 {image_tensor.shape}가 크롭 크기 {crop_size}보다 작습니다.")

    # 각 배치별로 랜덤 크롭 수행
    cropped_tensors = []
    cropped_d_centers = []
    cropped_h_centers = []
    cropped_w_centers = []
    
    for _ in range(num_crop_repeat):
        # 배치 크기만큼의 랜덤 크롭 위치 설정
        start_d = torch.randint(0, D - crop_d + 1, (B,))  # shape (B,)
        start_h = torch.randint(0, H - crop_h + 1, (B,))  # shape (B,)
        start_w = torch.randint(0, W - crop_w + 1, (B,))  # shape (B,)
        
        
        for b in range(B):
            # 각 배치에 대해 랜덤 위치에서 크롭 수행
            cropped_image = image_tensor[b, :,
                                         start_d[b]:start_d[b]+crop_d,
                                         start_h[b]:start_h[b]+crop_h,
                                         start_w[b]:start_w[b]+crop_w]
            cropped_tensors.append(cropped_image)
        cropped_d_centers.append(start_d)
        cropped_h_centers.append(start_h)
        cropped_w_centers.append(start_w)
    cropped_d_centers = torch.cat(cropped_d_centers, dim=0)
    cropped_h_centers = torch.cat(cropped_h_centers, dim=0)
    cropped_w_centers = torch.cat(cropped_w_centers, dim=0)
    
    cropped_centers = torch.stack([cropped_d_centers, cropped_h_centers, cropped_w_centers], dim=1)
    
    # 결과 텐서로 변환 (B * num_crop_repeat, C, crop_d, crop_h, crop_w)
    cropped_tensor_batch = torch.stack(cropped_tensors, dim=0)
    return cropped_tensor_batch, cropped_centers