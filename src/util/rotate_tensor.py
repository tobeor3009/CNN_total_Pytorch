import torch
from pytorch3d.transforms import euler_angles_to_matrix
import numpy as np

def _apply_rotation_to_single_channel(single_channel_tensor, rotated_coords):
    """
    회전된 좌표를 이용하여 하나의 채널에 대한 회전을 수행하는 함수.
    
    Args:
        single_channel_tensor (torch.Tensor): 입력 3D 채널 텐서, 크기 (D, H, W)
        rotated_coords (torch.Tensor): 회전된 좌표 텐서, 크기 (D, H, W, 3)
        
    Returns:
        torch.Tensor: 회전된 채널 텐서, 크기 (D, H, W)
    """
    D, H, W = single_channel_tensor.shape
    rotated_image = torch.zeros_like(single_channel_tensor)

    # 좌표값을 정수 인덱스로 변환하여 사용 (nearest neighbor 접근)
    rotated_coords = ((rotated_coords + 1) * 0.5 * torch.tensor([D, H, W])).long()  # [-1, 1] -> [0, D], [0, H], [0, W]

    # 좌표가 유효한 범위 내에 있는지 확인
    valid_mask = (
        (rotated_coords[..., 0] >= 0) & (rotated_coords[..., 0] < D) &
        (rotated_coords[..., 1] >= 0) & (rotated_coords[..., 1] < H) &
        (rotated_coords[..., 2] >= 0) & (rotated_coords[..., 2] < W)
    )

    # 유효한 좌표만 업데이트
    rotated_image[rotated_coords[valid_mask][..., 0],
                  rotated_coords[valid_mask][..., 1],
                  rotated_coords[valid_mask][..., 2]] = single_channel_tensor[valid_mask]

    return rotated_image

def convert_angle_to_radian(angle):
    return angle * np.pi / 180.0

def get_angle(angle_range, angle_x, angle_y, angle_z):
    if angle_x is None:
        angle_x = np.random.uniform(*angle_range)
    if angle_y is None:
        angle_y = np.random.uniform(*angle_range)
    if angle_z is None:
        angle_z = np.random.uniform(*angle_range)
    
    return angle_x, angle_y, angle_z


def rotate_3d_tensor(image_tensor, angle_range=(0, 360), 
                     angle_x=None, angle_y=None, angle_z=None, do_pad=True):
    """
    3D torch 텐서를 입력으로 받아 X, Y, Z 축을 기준으로 랜덤 회전시키는 함수.

    Args:
        image_tensor (torch.Tensor): 입력 3D 이미지 텐서, 크기 (B, C, D, H, W)
        angle_range (tuple): 각 축별 회전 각도의 범위 (기본값: (0, 360))

    Returns:
        torch.Tensor: 랜덤 회전 후의 이미지 텐서, 크기 (B, C, D, H, W)
    """
    B, C, D, H, W = image_tensor.shape
    if do_pad:
        diag_len = int(np.ceil(np.sqrt(D**2 + H**2 + W**2)))
        pad_d = (diag_len - D) // 2
        pad_h = (diag_len - H) // 2
        pad_w = (diag_len - W) // 2
    else:
        pad_d, pad_h, pad_w = 0, 0, 0
    # 이미지 패딩 추가 (회전 후 이미지가 잘리지 않도록 0으로 패딩)
    padded_tensor = F.pad(image_tensor, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='constant', value=0)
    _, _, padded_D, padded_H, padded_W = padded_tensor.shape
    
    # 각 배치에 대해 회전 수행
    rotated_tensors = []
    
    for b in range(B):
        angle_x, angle_y, angle_z = get_angle(angle_range, angle_x, angle_y, angle_z)
        # 랜덤한 각도 생성 (X, Y, Z 축 각각) - degree 단위를 radian으로 변환
        angle_x = convert_angle_to_radian(angle_x) # X 축 회전 (radian)
        angle_y = convert_angle_to_radian(angle_y) # Y 축 회전 (radian)
        angle_z = convert_angle_to_radian(angle_z) # Z 축 회전 (radian)
        # Euler angles to rotation matrix (X -> Y -> Z 순서로 회전)
        rotation_matrix = euler_angles_to_matrix(torch.tensor([angle_x, angle_y, angle_z]), convention="XYZ")

        # 좌표계 생성 및 회전 수행
        d_range = torch.linspace(-1, 1, padded_D)
        h_range = torch.linspace(-1, 1, padded_H)
        w_range = torch.linspace(-1, 1, padded_W)
        d, h, w = torch.meshgrid(d_range, h_range, w_range, indexing='ij')
        # 좌표를 쌓아서 (D * H * W, 3) 형태로 변경
        original_coords = torch.stack([d.flatten(), h.flatten(), w.flatten()], dim=1).to(image_tensor.device)  # (D*H*W, 3)
        # 회전 적용
        rotated_coords = torch.matmul(original_coords, rotation_matrix.T)  # (D*H*W, 3)

        # 좌표를 (D, H, W, 3) 형태로 변경하여 텐서로 변환
        rotated_coords = rotated_coords.reshape(padded_D, padded_H, padded_W, 3)
        # 회전된 좌표를 이용해 샘플링 없이 직접적인 회전 수행
        # 각 좌표를 이용하여 텐서를 회전시키기 위한 논리를 작성해야 함 (ex: nearest neighbor 방식)

        # 새로운 좌표를 이용해 회전된 이미지 텐서 생성
        rotated_image = torch.zeros_like(padded_tensor[b])  # 새로운 빈 텐서 생성 (C, D, H, W)

        # 좌표 변환을 통해 직접 회전된 이미지를 채우기
        for c in range(C):
            rotated_image[c] = _apply_rotation_to_single_channel(padded_tensor[b, c], rotated_coords)

        rotated_tensors.append(rotated_image)

    # 배치 형태로 변환하여 반환
    rotated_tensor_batch = torch.stack(rotated_tensors, dim=0)
    return rotated_tensor_batch