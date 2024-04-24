import cv2
import albumentations as A
import albumentations.pytorch
import numpy as np

base_augmentation_policy_dict = {
    "positional": True,
    "noise": True,
    "elastic": True,
    "brightness_contrast": True,
    "color": True,
    "to_jpeg": True
}

################### Resize Constant ###################
INTER_DICT = {
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST
}
################### Augment Constant ###################
positional_transform = A.OneOf([
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.RandomRotate90(p=1)
], p=1)

noise_transform = A.OneOf([
    A.Blur(blur_limit=(2, 2), p=1),
    A.GaussNoise(var_limit=(0.01, 5), p=1),
], p=1)

elastic_tranform = A.ElasticTransform(always_apply=False, p=0.5,
                                      alpha=0.20000000298023224, sigma=3.359999895095825,
                                      alpha_affine=2.009999990463257, interpolation=1, border_mode=1,
                                      value=(0, 0, 0), mask_value=None, approximate=False)

brightness_value = 0.1
brightness_contrast_transform = A.OneOf([
    A.RandomBrightnessContrast(
        brightness_limit=(-brightness_value, brightness_value),
        contrast_limit=(-brightness_value, brightness_value), p=1),
], p=1)

color_transform = A.OneOf([
    A.ISONoise(always_apply=False, p=0.5,
               intensity=(0.05000000074505806, 0.12999999523162842),
               color_shift=(0.009999999776482582, 0.26999998092651367)),
    A.HueSaturationValue(p=0.1)
], p=1)

hist_transform = A.OneOf([
    A.CLAHE(always_apply=True, p=0.5,
            clip_limit=(1, 15), tile_grid_size=(8, 8)),
], p=1)

to_jpeg_transform = A.ImageCompression(quality_lower=99,
                                       quality_upper=100,
                                       p=1)

to_tensor_transform = albumentations.pytorch.transforms.ToTensorV2()
################### Preprocess Constant ###################


def get_resized_array(image_array, target_size, interpolation):
    image_resized_array = image_array
    if target_size is not None:
        image_resized_array = cv2.resize(src=image_array,
                                         dsize=target_size,
                                         interpolation=INTER_DICT[interpolation]
                                         )
    return image_resized_array


def get_augumented_array(image_array, augmentation_proba, augmentation_policy_dict):
    final_transform_list = []
    if augmentation_policy_dict["positional"] is True:
        final_transform_list.append(positional_transform)
    if augmentation_policy_dict["noise"] is True:
        final_transform_list.append(noise_transform)
    if augmentation_policy_dict["elastic"] is True:
        final_transform_list.append(elastic_tranform)
    if augmentation_policy_dict["brightness_contrast"] is True:
        final_transform_list.append(brightness_contrast_transform)
    if augmentation_policy_dict["color"] is True:
        final_transform_list.append(color_transform)
    if augmentation_policy_dict["to_jpeg"] is True:
        final_transform_list.append(to_jpeg_transform)

    final_transform = A.Compose(final_transform_list, p=augmentation_proba)

    if augmentation_proba == 0:
        return image_array
    else:
        return final_transform(image=image_array)['image']


def get_seg_augumented_array(image_array, mask_array,
                             augmentation_proba, augmentation_policy_dict):
    final_transform_list = []
    if augmentation_policy_dict["positional"] is True:
        final_transform_list.append(positional_transform)
    if augmentation_policy_dict["noise"] is True:
        final_transform_list.append(noise_transform)
    if augmentation_policy_dict["elastic"] is True:
        final_transform_list.append(elastic_tranform)
    if augmentation_policy_dict["brightness_contrast"] is True:
        final_transform_list.append(brightness_contrast_transform)
    if augmentation_policy_dict["color"] is True:
        final_transform_list.append(color_transform)
    if augmentation_policy_dict["to_jpeg"] is True:
        final_transform_list.append(to_jpeg_transform)

    final_transform = A.Compose(final_transform_list, p=augmentation_proba)

    if augmentation_proba == 0:
        return image_array, mask_array
    else:
        transformed = final_transform(image=image_array, mask=mask_array)
        return transformed['image'], transformed['mask']


def get_preprocessed_array(image_array, preprocess_input):
    if preprocess_input is None:
        return image_array
    elif preprocess_input == "-1~1":
        return (image_array / 127.5) - 1
    elif preprocess_input == "0~1":
        return image_array / 255
    else:
        return preprocess_input(image_array)
