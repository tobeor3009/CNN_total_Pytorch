import cv2
import albumentations as A
import numpy as np

base_aug_policy_dict = {
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
    "cubic": cv2.INTER_CUBIC
}
################### Augment Constant ###################
positional_transform = A.OneOf([
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.RandomRotate90(p=1)
], p=0.5)

noise_transform = A.OneOf([
    A.Blur(blur_limit=(2, 2), p=1),
    A.GaussNoise(var_limit=(0.01, 5), p=1),
], p=0.5)

elastic_tranform = A.ElasticTransform(p=0.5)

brightness_value = 0.2
brightness_contrast_transform = A.OneOf([
    A.RandomBrightnessContrast(
        brightness_limit=(-brightness_value, brightness_value), contrast_limit=(-brightness_value, brightness_value), p=1),
], p=0.5)

color_transform = A.OneOf([
    A.ChannelShuffle(p=1),
    A.ToGray(p=1),
    A.ToSepia(p=1),
], p=0.5)

to_jpeg_transform = A.ImageCompression(
    quality_lower=99, quality_upper=100, p=0.5)
################### Preprocess Constant ###################


def get_resized_array(image_array, target_size, interpolation):
    image_resized_array = cv2.resize(src=image_array,
                                     dsize=target_size,
                                     interpolation=INTER_DICT[interpolation]
                                     )
    if len(image_resized_array.shape) == 2:
        image_resized_array = np.expand_dims(image_resized_array, axis=-1)
    return image_resized_array


def get_augumented_array(image_array, argumentation_proba, argumentation_policy_dict):
    final_transform_list = []
    if argumentation_policy_dict["positional"] is True:
        final_transform_list.append(positional_transform)
    if argumentation_policy_dict["noise"] is True:
        final_transform_list.append(noise_transform)
    if argumentation_policy_dict["elastic"] is True:
        final_transform_list.append(elastic_tranform)
    if argumentation_policy_dict["brightness_contrast"] is True:
        final_transform_list.append(brightness_contrast_transform)
    if argumentation_policy_dict["color"] is True:
        final_transform_list.append(color_transform)
    if argumentation_policy_dict["to_jpeg"] is True:
        final_transform_list.append(to_jpeg_transform)

    final_transform = A.Compose(
        final_transform_list, p=argumentation_proba)

    if argumentation_proba == 0:
        return image_array
    else:
        return final_transform(image=image_array)['image']


def get_preprocessed_array(image_array, preprocess_input):
    if preprocess_input is None:
        return image_array
    elif preprocess_input == "-1~1":
        return (image_array / 127.5) - 1
    elif preprocess_input == "mask":
        return image_array / 255
    else:
        return preprocess_input(image_array)
