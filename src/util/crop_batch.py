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

def rotate_and_crop_tensor(image_tensor, crop_size=128, angle_range=(0, 360), num_crop_repeat=64):
    """
    배치 형태의 torch 텐서를 입력으로 받아 랜덤 회전 후, 랜덤 위치의 크롭을 수행하는 함수.
    
    Args:
        image_tensor (torch.Tensor): 입력 이미지 텐서, 크기 (B, C, 512, 512)
        crop_size (int): 크롭할 이미지의 크기 (기본값: 128)
        angle_range (tuple): 회전 각도의 범위 (기본값: (0, 360))
        num_crop_repeat (int): 각 이미지당 생성할 크롭 횟수 (기본값: 1)
        
    Returns:
        torch.Tensor: 랜덤 회전 및 크롭 후의 이미지 텐서, 크기 (B * num_crop_repeat, C, crop_size, crop_size)
    """
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
    for _ in range(num_crop_repeat):
        for b in range(B):
            # 랜덤한 각도 생성
            angle = np.random.uniform(*angle_range)

            # 배치 내의 이미지 선택 (C, H, W)
            single_image_tensor = padded_tensor[b]

            # 텐서를 PIL 이미지로 변환하여 회전 수행
            rotated_image = TF.rotate(single_image_tensor, angle=angle)

            # 회전된 이미지의 크기
            _, rotated_h, rotated_w = rotated_image.shape
            
            edge_start = crop_size // 2
            # 랜덤 크롭 위치 설정
            top = torch.randint(edge_start, rotated_h - crop_size - edge_start + 1, (1,)).item()
            left = torch.randint(edge_start, rotated_w - crop_size - edge_start + 1, (1,)).item()

            # 회전된 이미지에서 크롭
            cropped_image = TF.crop(rotated_image, top, left, crop_size, crop_size)

            # 크롭된 이미지 텐서를 리스트에 추가
            cropped_tensors.append(cropped_image)

    # 결과 텐서로 변환 (B * num_crop_repeat, C, crop_size, crop_size)
    cropped_tensor_batch = torch.stack(cropped_tensors, dim=0)
    return cropped_tensor_batch