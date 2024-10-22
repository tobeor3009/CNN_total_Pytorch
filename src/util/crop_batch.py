import random
from functools import partial

import numpy as np
import cv2
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF

def rotate_and_crop_with_padding(image, crop_size=256, angle_range=(0, 360)):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"이미지 크기가 {crop_size}x{crop_size}보다 작습니다.")

    # 이미지가 잘리지 않도록 패딩 추가 (대각선 길이 만큼 추가)
    diagonal_length = int(np.sqrt(h**2 + w**2))
    pad_h = (diagonal_length - h) // 2
    pad_w = (diagonal_length - w) // 2
    
    # 이미지 패딩 추가 (회전 후 이미지가 잘리지 않도록 0으로 패딩)
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)

    # 랜덤한 회전 각도 생성
    angle = np.random.uniform(angle_range[0], angle_range[1])

    # 회전 중심을 패딩된 이미지의 중심으로 설정
    padded_h, padded_w = padded_image.shape[:2]
    center = (padded_w // 2, padded_h // 2)

    # 회전 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 회전된 이미지를 위한 새로운 크기 계산
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])
    new_w = int((padded_h * sin_val) + (padded_w * cos_val))
    new_h = int((padded_h * cos_val) + (padded_w * sin_val))

    # 회전 행렬에 이동을 추가하여 회전 후 이미지의 중심이 기존 이미지의 중심에 위치하도록 변환
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # 패딩된 이미지를 회전하여 새로운 이미지 생성
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (new_w, new_h))

    # 회전된 이미지에서 크롭 위치를 랜덤으로 설정
    x_start = np.random.randint(0, new_w - crop_size + 1)
    y_start = np.random.randint(0, new_h - crop_size + 1)

    # 회전된 이미지에서 크롭된 부분을 추출
    cropped_image = rotated_image[y_start:y_start+crop_size, x_start:x_start+crop_size]

    return cropped_image

def get_rand_center_origin_idx(rotate_size, crop_size):
    edge_start = crop_size
    edge_end = rotate_size - crop_size - edge_start + 1
    origin_range = range(edge_start, edge_end)
    return random.choice(origin_range)

def get_rand_edge_origin_idx(rotate_size, crop_size, crop_coef=2.0):
    edge_start = crop_size
    edge_end = rotate_size - crop_size - edge_start + 1
    edge_adj_num = int(round(crop_size / crop_coef))
    
    edge_candi_early = range(crop_size - edge_adj_num, edge_start + 1)
    edge_candi_late = range(edge_end, edge_end + edge_adj_num)
    edge_candi_list = list(edge_candi_early) + list(edge_candi_late)
    return random.choice(edge_candi_list)

def get_rand_center_origin_tuple(rotated_h, rotated_w, crop_size):
    row_idx = get_rand_center_origin_idx(rotated_h, crop_size)
    col_idx = get_rand_center_origin_idx(rotated_w, crop_size)
    return row_idx, col_idx

def get_rand_border_origin_tuple(rotated_h, rotated_w, crop_size, crop_coef=2.0):
    rand_val = random.random()
    if rand_val < 0.475:
        row_idx = get_rand_center_origin_idx(rotated_h, crop_size)
        col_idx = get_rand_edge_origin_idx(rotated_w, crop_size, crop_coef=crop_coef)
    elif rand_val < 0.95:
        row_idx = get_rand_edge_origin_idx(rotated_h, crop_size, crop_coef=crop_coef)
        col_idx = get_rand_center_origin_idx(rotated_w, crop_size)
    else:
        row_idx = get_rand_edge_origin_idx(rotated_h, crop_size, crop_coef=crop_coef)
        col_idx = get_rand_edge_origin_idx(rotated_w, crop_size, crop_coef=crop_coef)
    return row_idx, col_idx
    
def get_angle_0(angle_min, angle_max):
    return 0.0

def rotate_and_crop_tensor(image_tensor, crop_size=128, angle_range=(0, 360), num_crop_repeat=64,
                           center_crop_ratio=0.8, crop_coef=2.0):
    """
    배치 형태의 torch 텐서를 입력으로 받아 랜덤 회전 후, 랜덤 위치의 크롭을 수행하는 함수.
    
    Args:
        image_tensor (torch.Tensor): 입력 이미지 텐서, 크기 (B, C, H, W)
        crop_size (int): 크롭할 이미지의 크기 (기본값: 128)
        angle_range (tuple): 회전 각도의 범위 (기본값: (0, 360)) center_crop 때만 사용되며, border_crop때는 회전하지 않는다.
        num_crop_repeat (int): 각 이미지당 생성할 크롭 횟수 (기본값: 1)
        center_crop_ratio (float): crop 할때 center_crop 과 border_crop을 달리 취급해서 수행하는데, center_crop의 비율을 결정하며, border_crop은 (1 - center_crop) 이다.
        crop_coef (float): border를 crop할때 얼마나 멀리 crop할지 결정하는 ratio. 너무 작으면 padding된 의미없는 부분을 잘라온다. 크면 이미지를 커버하지 못하는 (border 쪽) 부분이 생긴다.
        patch_size가 image_size의 1/4 이면 2.0, 1/8 이면 1.2를 추천한다.
    
    Caution:
        이 함수는 H % crop_size == 0, W % crop_size == 0 를 만족해야 하며, H // crop_size >= 4 가 되도록 한다. W도 마찬가지. H == W 인 경우를 상정하고 설계되었다. 
        바꾸려면 crop_size 과 crop_coef를 tuple로 들어가게 설계해야한다.
    Returns:
        torch.Tensor: 랜덤 회전 및 크롭 후의 이미지 텐서, 크기 (B * num_crop_repeat, C, crop_size, crop_size)
    """
    center_crop_repeat = int(round(num_crop_repeat * center_crop_ratio))
    # 입력 텐서 크기 확인
    B, C, H, W = image_tensor.shape
    if H < crop_size or W < crop_size:
        raise ValueError(f"이미지 크기가 {crop_size}x{crop_size}보다 작습니다.")

    # 이미지가 잘리지 않도록 패딩 추가 (대각선 길이 만큼 추가)
    diagonal_length = int(np.sqrt(H**2 + W**2))
    pad_h = (diagonal_length - H) // 2
    pad_w = (diagonal_length - W) // 2

    # 이미지 패딩 추가 (회전 후 이미지가 잘리지 않도록 0으로 패딩)
    padded_tensor = F.pad(image_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    # 배치 형태의 텐서 처리
    cropped_tensors = []
    for crop_idx in range(num_crop_repeat):
        if crop_idx < center_crop_repeat - 1:
            get_origin_tuple_fn = get_rand_center_origin_tuple
            get_angle_fn = np.random.uniform
        else:
            get_origin_tuple_fn = partial(get_rand_border_origin_tuple, crop_coef=crop_coef)
            get_angle_fn = get_angle_0
        for b in range(B):
            # 랜덤한 각도 생성
            angle = get_angle_fn(*angle_range)

            # 배치 내의 이미지 선택 (C, H, W)
            single_image_tensor = padded_tensor[b]

            # 텐서를 PIL 이미지로 변환하여 회전 수행
            rotated_image = TF.rotate(single_image_tensor, angle=angle, fill=0)
            # 회전된 이미지의 크기
            _, rotated_h, rotated_w = rotated_image.shape
            
            # 랜덤 크롭 위치 설정
            row_idx, col_idx = get_origin_tuple_fn(rotated_h, rotated_w, crop_size)
            # 회전된 이미지에서 크롭
            cropped_image = TF.crop(rotated_image, row_idx, col_idx, crop_size, crop_size)

            # 크롭된 이미지 텐서를 리스트에 추가
            cropped_tensors.append(cropped_image)

    # 결과 텐서로 변환 (B * num_crop_repeat, C, crop_size, crop_size)
    cropped_tensor_batch = torch.stack(cropped_tensors, dim=0)
    return cropped_tensor_batch

def rotate_and_crop_tensor_for_seg(image_tensor, mask_idx_tensor, crop_size=128, 
                                   angle_range=(0, 360), num_crop_repeat=64, center_crop_ratio=0.8, crop_coef=2.0):
    
    concat_tensor = torch.cat([image_tensor, mask_idx_tensor.unsqueeze(1).float()], dim=1)

    concat_patch_tensor = rotate_and_crop_tensor(concat_tensor, crop_size=crop_size, angle_range=angle_range, num_crop_repeat=num_crop_repeat,
                                                 center_crop_ratio=center_crop_ratio, crop_coef=crop_coef)
    image_patch_tensor, mask_idx_patch_tensor = concat_patch_tensor[:, :-1], concat_patch_tensor[:, -1]
    mask_idx_patch_tensor = (mask_idx_patch_tensor > 0.5).long()
    return image_patch_tensor, mask_idx_patch_tensor