import os
import cv2
import json
import numpy as np
import importlib.util

if importlib.util.find_spec("cupy") is not None:
    import cupy as cp
    print("✅ CuPy is installed.")
else:
    print("❌ CuPy is NOT installed.")

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])

def RGB_to_LAB(image_array, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    arr = image_array[:, :, ::-1]
    arr = arr / 255
    mask = arr > 0.04045
    arr[mask] = compat_cp.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    arr = arr @ xyz_from_rgb.T
    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    mask = arr > 0.008856
    mask_x, mask_y, mask_z = mask[..., 0], mask[..., 1], mask[..., 2]

    arr_converted = compat_cp.zeros_like(arr)
    arr_converted[mask] = compat_cp.cbrt(arr[mask])
    arr_converted[~mask] = 7.787 * arr[~mask] + (16 / 116)

    x_converted, y_converted, z_converted = arr_converted[...,
                                                          0], arr_converted[..., 1], arr_converted[..., 2]

    L = compat_cp.zeros_like(y)

    # Nonlinear distortion and linear transformation
    L[mask_y] = 116 * compat_cp.cbrt(y[mask_y]) - 16
    L[~mask_y] = 903.3 * y[~mask_y]
    L *= 2.55
    # if want to see this formula, go to https://docs.opencv.org/3.4.15/de/d25/imgproc_color_conversions.html RGB <-> CIELab
    a = 500 * (x_converted - y_converted) + 128
    b = 200 * (y_converted - z_converted) + 128

    return compat_cp.round(compat_cp.stack([L, a, b], axis=-1)).astype("uint8")


def get_tissue_mask(I, luminosity_threshold=0.8, use_gpu=False):
    #     I_LAB = RGB_to_LAB(I, use_gpu=use_gpu)
    L = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)[:, :, 0].astype("float16")
    L = L / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
#     if mask.sum() == 0:
#         raise Exception("Empty tissue mask computed")

    return mask


def RGB_to_OD(I, use_gpu=False):

    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    I[I == 0] = 1
    return -1 * compat_cp.log(I / 255)


def OD_to_RGB(OD, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, 'Negative optical density'
    return (255 * compat_cp.exp(-1 * OD)).astype(np.uint8)

def get_random_index_permutation_range(image_shape, patch_stride):
    row_index_range = range(0, image_shape[0], patch_stride)
    col_index_range = range(0, image_shape[1], patch_stride)

    random_permutation_len = len(row_index_range) * len(col_index_range)
    random_permutation = np.zeros((random_permutation_len, 2), dtype="int32")

    for row_index, row_num in enumerate(row_index_range):
        for col_index, col_num in enumerate(col_index_range):
            current_index = row_index * len(col_index_range) + col_index 
            random_permutation[current_index] = (row_num, col_num)
    np.random.shuffle(random_permutation)
    return random_permutation